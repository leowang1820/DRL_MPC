"""Differentiable mirror of Reference7DOF, NOT a replacement physical model.

All tire coefficients, force-derived load memory and Euler clamps are retained.
Float64 is required by the training workflow. No parameters are identified here.
At the origin the norm uses sqrt(clamp(sum of squares, EPS**2)) so autograd has
finite zero subgradient; its values match the reference max(hypot(...), EPS).
"""

import torch

from ..model import EPS, TireFit, VehicleParameters


def rhs(state, controls, mu, memory):
    """Batched (...,7), (...,6), (...,), (...,2) -> derivative, memory, invalid.

    controls = actual FL/FR steer, then FL/FR/RL/RR signed net torque.
    Arbitrary leading batch dimensions are supported, no measurements injected.
    """
    v, tire = VehicleParameters(), TireFit()
    vx, vy, r = state[..., 0], state[..., 1], state[..., 2]
    wheel_x = state.new_tensor([v.lf_m, v.lf_m, -v.lr_m, -v.lr_m])
    wheel_y = state.new_tensor([v.track_front_m/2, -v.track_front_m/2,
                                v.track_rear_m/2, -v.track_rear_m/2])
    delta = torch.cat([controls[..., :2], torch.zeros_like(controls[..., :2])], dim=-1)
    co, si = torch.cos(delta), torch.sin(delta)
    ax, ay = memory[..., 0], memory[..., 1]
    front = (v.mass_kg*v.gravity_m_s2*v.lr_m-v.mass_kg*ax*v.cg_height_m)/v.wheelbase_m
    rear = (v.mass_kg*v.gravity_m_s2*v.lf_m+v.mass_kg*ax*v.cg_height_m)/v.wheelbase_m
    ft = v.mass_kg*ay*v.cg_height_m*v.lr_m/(v.wheelbase_m*v.track_front_m)
    rt = v.mass_kg*ay*v.cg_height_m*v.lf_m/(v.wheelbase_m*v.track_rear_m)
    raw_fz = torch.stack([front/2-ft, front/2+ft, rear/2-rt, rear/2+rt], dim=-1)
    fz = raw_fz.clamp_min(100.)
    vxb, vyb = vx[..., None]-r[..., None]*wheel_y, vy[..., None]+r[..., None]*wheel_x
    vxw = vxb*co+vyb*si
    slip = (state[..., 3:]*v.wheel_radius_m-vxw)/vxw.abs().clamp_min(.1)
    alpha = torch.atan2(vyb, vxb.clamp_min(.1))-delta
    z = fz/1000.
    a, b = tire.longitudinal, tire.lateral
    friction = mu[..., None]
    dx = friction*(a[1]*z**2+a[2]*z)
    bx = ((a[3]*z**2+a[4]*z)*torch.exp(-a[5]*z))/(a[0]*dx+EPS)
    ex = a[6]*z**2+a[7]*z+a[8]
    dy = friction*(b[1]*z**2+b[2]*z)
    by = (b[3]*torch.sin(2*torch.atan(z/b[4]))*(1-b[5]*tire.camber_fit))/(b[0]*dy+EPS)
    ey = b[6]*z+b[7]
    bxk, bya = bx*slip, by*torch.rad2deg(alpha)
    fx0 = 1000*dx*torch.sin(a[0]*torch.atan(bxk-ex*(bxk-torch.atan(bxk))))
    fy0 = -1000*dy*torch.sin(b[0]*torch.atan(bya-ey*(bya-torch.atan(bya))))
    denominator = (1+slip).abs().clamp_min(1e-5)
    sx, sy = (slip/denominator).abs(), (torch.tan(alpha)/denominator).abs()
    norm = (sx.square()+sy.square()).clamp_min(EPS**2).sqrt()
    fx, fy = sx/norm*fx0, sy/norm*fy0
    # Reference uses hypot + EPS (not a changed force law).
    # vector_norm provides a finite subgradient at exactly zero.
    force_norm = torch.linalg.vector_norm(torch.stack([fx, fy], dim=-1), dim=-1)
    scale = torch.minimum(torch.ones_like(fx), friction*fz/(force_norm+EPS))
    fx, fy = fx*scale, fy*scale
    fxb, fyb = fx*co-fy*si, fx*si+fy*co
    tire_ax, tire_ay = fxb.sum(-1)/v.mass_kg, fyb.sum(-1)/v.mass_kg
    moment = (wheel_x*fyb-wheel_y*fxb).sum(-1)
    derivative = torch.cat([
        torch.stack([tire_ax+r*vy, tire_ay-100*torch.exp(-vy**2)/v.mass_kg*vy-r*vx,
                     moment/v.yaw_inertia_kg_m2], dim=-1),
        (controls[..., 2:]-fx*v.wheel_radius_m)/v.wheel_inertia_kg_m2], dim=-1)
    invalid = ((raw_fz < 100).any(-1) | (vxb < .1).any(-1) | (vxw.abs() < .1).any(-1)
               | (vx <= 0) | (vxw <= 0).any(-1) | (dx < 0).any(-1) | (dy < 0).any(-1))
    return derivative, torch.stack([tire_ax, tire_ay], dim=-1), invalid


def euler_step(state, derivative, dt):
    raw = state+dt*derivative
    invalid = (raw[..., 0] < .1) | (raw[..., 3:] < -1).any(-1)
    x = torch.cat([raw[..., :1].clamp_min(.1), raw[..., 1:3], raw[..., 3:].clamp_min(-1)], dim=-1)
    return x, invalid
