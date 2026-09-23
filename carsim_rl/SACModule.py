"""SAC networks, replay buffer, and update logic for the CarSim trainer."""

import collections
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class ReplayBuffer:
    """Fixed-size buffer that stores transition tuples for off-policy updates."""

    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones


# Step 5: Utility to softly copy parameters between networks
@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak-averages the parameters of the target network towards the source."""
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


# Step 6: Actor network that outputs a squashed Gaussian policy
class PolicyNetwork(nn.Module):
    """Gaussian policy network with tanh squashing for bounded action spaces."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int, action_scale: np.ndarray):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_dim, act_dim)
        self.log_std_layer = nn.Linear(hidden_dim, act_dim)
        
        # Store action scaling parameters as buffers (not trainable)
        self.register_buffer("action_scale", torch.tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.zeros(act_dim, dtype=torch.float32))

    def forward(self, state: torch.Tensor):
        """Returns mean and log_std for the Gaussian distribution."""
        hidden = self.net(state)
        mu = self.mu_layer(hidden)
        log_std = self.log_std_layer(hidden)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, state: torch.Tensor):
        """Samples an action using the reparameterization trick.
        
        Returns:
            action: Scaled action for the environment
            log_prob: Log probability of the action
            mean_action: Deterministic action (for evaluation)
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()
        
        # Sample from normal distribution
        normal = torch.distributions.Normal(mu, std)
        x_t = normal.rsample()  # Reparameterization trick
        
        # Apply tanh squashing
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        
        # Calculate log probability with change of variables formula
        log_prob = normal.log_prob(x_t)
        # Enforcing action bounds (tanh)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        # Deterministic action for evaluation
        mean = torch.tanh(mu) * self.action_scale + self.action_bias
        
        return action, log_prob, mean


# Step 7: Q-network (critic) for value estimation
class QNetwork(nn.Module):
    """Q-network that estimates action-value function."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """Returns Q(s, a)."""
        x = torch.cat([state, action], dim=1)
        return self.net(x)


# Step 8: SAC Agent class that encapsulates all training logic
class SACAgent:
    """Soft Actor-Critic agent."""
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_scale: np.ndarray,
        device: torch.device,
        gamma: float = 0.99,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        hidden_size: int = 256,
        tau: float = 5e-3,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha
        
        # Initialize networks
        self.policy_net = PolicyNetwork(obs_dim, act_dim, hidden_size, action_scale).to(device)
        
        self.q1_net = QNetwork(obs_dim, act_dim, hidden_size).to(device)
        self.q2_net = QNetwork(obs_dim, act_dim, hidden_size).to(device)
        
        self.q1_target_net = QNetwork(obs_dim, act_dim, hidden_size).to(device)
        self.q2_target_net = QNetwork(obs_dim, act_dim, hidden_size).to(device)
        self.q1_target_net.load_state_dict(self.q1_net.state_dict())
        self.q2_target_net.load_state_dict(self.q2_net.state_dict())
        
        # Freeze target networks
        for param in self.q1_target_net.parameters():
            param.requires_grad = False
        for param in self.q2_target_net.parameters():
            param.requires_grad = False
        
        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr_actor)
        self.q1_optimizer = torch.optim.Adam(self.q1_net.parameters(), lr=lr_critic)
        self.q2_optimizer = torch.optim.Adam(self.q2_net.parameters(), lr=lr_critic)
        
        # Entropy temperature
        if auto_alpha:
            self.target_entropy = -float(act_dim)
            self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32, 
                                         device=device, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
        else:
            self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32, device=device)
            self.alpha_optimizer = None
            self.target_entropy = None
    
    @torch.no_grad()
    def select_action(self, state: np.ndarray, deterministic: bool = False):
        """Select action from the policy."""
        state_v = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if deterministic:
            _, _, action_v = self.policy_net.sample(state_v)
        else:
            action_v, _, _ = self.policy_net.sample(state_v)
        return action_v.squeeze(0).cpu().numpy()
    
    def update(self, batch):
        """Perform one step of gradient descent on the networks."""
        states, actions, rewards, next_states, dones = batch
        
        states_v = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_v = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards_v = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_v = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_v = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        alpha = self.log_alpha.exp().detach()
        
        # ============================================================
        # Update Q-networks (critics)
        # ============================================================
        with torch.no_grad():
            # Sample actions from current policy for next states
            next_actions_v, next_log_probs_v, _ = self.policy_net.sample(next_states_v)
            
            # Compute target Q-values using target networks
            next_q1_v = self.q1_target_net(next_states_v, next_actions_v)
            next_q2_v = self.q2_target_net(next_states_v, next_actions_v)
            next_q_v = torch.min(next_q1_v, next_q2_v)
            
            # Target with entropy regularization
            target_q_v = rewards_v + self.gamma * (1.0 - dones_v) * (
                next_q_v - alpha * next_log_probs_v
            )
        
        # Current Q estimates
        q1_v = self.q1_net(states_v, actions_v)
        q2_v = self.q2_net(states_v, actions_v)
        
        # Q-network losses (MSE)
        q1_loss = F.mse_loss(q1_v, target_q_v)
        q2_loss = F.mse_loss(q2_v, target_q_v)
        
        # Update Q-networks
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # ============================================================
        # Update policy network (actor)
        # ============================================================
        # Sample new actions from current policy
        new_actions_v, log_probs_v, _ = self.policy_net.sample(states_v)
        
        # Compute Q-values for new actions
        q1_new_v = self.q1_net(states_v, new_actions_v)
        q2_new_v = self.q2_net(states_v, new_actions_v)
        min_q_new_v = torch.min(q1_new_v, q2_new_v)
        
        # Policy loss: maximize Q - alpha * log_prob
        policy_loss = (alpha * log_probs_v - min_q_new_v).mean()
        
        # Update policy
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        # ============================================================
        # Update entropy temperature (alpha)
        # ============================================================
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs_v.detach() + self.target_entropy)).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            alpha_loss = torch.tensor(0.0)
        
        # ============================================================
        # Soft update of target networks
        # ============================================================
        soft_update(self.q1_target_net, self.q1_net, self.tau)
        soft_update(self.q2_target_net, self.q2_net, self.tau)
        
        # Return losses for logging
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': alpha.item(),
            'entropy': -log_probs_v.mean().item(),
        }
    
    def save(self, path: str):
        """Save agent's networks."""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'q1_state_dict': self.q1_net.state_dict(),
            'q2_state_dict': self.q2_net.state_dict(),
            'q1_target_state_dict': self.q1_target_net.state_dict(),
            'q2_target_state_dict': self.q2_target_net.state_dict(),
            'log_alpha': self.log_alpha,
        }, path)
    
    def load(self, path: str):
        """Load agent's networks."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.q1_net.load_state_dict(checkpoint['q1_state_dict'])
        self.q2_net.load_state_dict(checkpoint['q2_state_dict'])
        self.q1_target_net.load_state_dict(checkpoint['q1_target_state_dict'])
        self.q2_target_net.load_state_dict(checkpoint['q2_target_state_dict'])
        self.log_alpha = checkpoint['log_alpha']
