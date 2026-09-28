"""Differentiable physics equivalence and no-teacher-forcing sequence tests."""

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from dynamics7dof.model import Controls, LoadMemory, Reference7DOF
from dynamics7dof.learning.ensemble import ResidualEnsemble, fit_scaler, make_members
from dynamics7dof.learning.online import OnlineShadow
from dynamics7dof.learning.rollout import (TorchRollout, batch_for, horizon_profile, prepare_windows,
                                          train_rollout, window_indices)
from dynamics7dof.model import STATE_NAMES
from dynamics7dof.learning.symmetry import fit_population
from dynamics7dof.learning.torch_physics import rhs, euler_step


class RolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self, zero=False):
        rng = np.random.default_rng(27)
        features = rng.normal(size=(100, 16))
        targets = rng.normal(size=(100, 7))*.1
        m = ResidualEnsemble(make_members(2, 8), fit_scaler(*fit_population(features, targets, 'mirror')),
                             symmetry='mirror')
        m.scaler['lower'][:] = -10000
        m.scaler['upper'][:] = 10000
        if zero:
            with torch.no_grad():
                for p in m.members.parameters():
                    p.zero_()
            m.refresh()
        return m

    def test_torch_rhs_matches_numpy_random_states_and_memory(self):
        rng = np.random.default_rng(13)
        for _ in range(25):
            x = np.r_[rng.uniform(5, 20), rng.uniform(-.5, .5), rng.uniform(-.1, .1), rng.uniform(20, 50, 4)]
            u = np.r_[rng.uniform(-.05, .05, 2), rng.uniform(-50, 50, 4)]
            mem, mu = rng.uniform(-2, 2, 2), rng.uniform(.3, 1.)
            a = Reference7DOF().evaluate(x, Controls(u[:2], u[2:]), mu=mu, load_memory=LoadMemory(*mem))
            b, memory, invalid = rhs(*(torch.tensor(v, dtype=torch.float64) for v in (x, u, mu, mem)))
            np.testing.assert_allclose(b.numpy(), a.derivative, rtol=1e-10, atol=1e-8)
            np.testing.assert_allclose(memory.numpy(), [a.next_load_memory.ax_m_s2,a.next_load_memory.ay_m_s2], atol=1e-10)
            self.assertEqual(bool(invalid), bool(a.flags))

    def test_zero_slip_gradients_are_finite(self):
        x = torch.tensor([10.,0.,0.,*[10/.353]*4], dtype=torch.float64, requires_grad=True)
        u = torch.zeros(6, dtype=torch.float64)
        f, memory, _ = rhs(x, u, torch.tensor(.5), torch.zeros(2,dtype=torch.float64))
        (f.square().sum()+memory.square().sum()).backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_physics_state_and_memory_gradcheck(self):
        x = torch.tensor([11., .23, .031, 31.1,31.6,31.3,31.8], dtype=torch.float64,requires_grad=True)
        m = torch.tensor([.2,-.4],dtype=torch.float64,requires_grad=True)
        u = torch.tensor([.021,.026,10.,20.,-30.,-40.],dtype=torch.float64)
        def fun(x, m):
            f, mem, _ = rhs(x,u,torch.tensor(.5,dtype=torch.float64),m)
            return torch.cat([f,mem])
        self.assertTrue(torch.autograd.gradcheck(fun,(x,m),eps=1e-6,atol=2e-4,rtol=2e-4))

    def test_zero_network_rollout_is_reference_not_teacher_forced(self):
        model = self.model(zero=True); engine = TorchRollout(model)
        x = np.array([11., .03, .01, *[11/.353]*4])
        u = np.array([.012,.014,10.,20.,-2.,-3.])
        px = engine.tensor(np.tile(x,(2,1,1)))
        pu = engine.tensor(np.tile(u,(2,1,12,1)))
        predictions, bad = engine.predict(px,engine.tensor(np.zeros((2,1,2))),pu,engine.tensor(np.full((2,1),.5)))
        memory = LoadMemory(0,0)
        expected = []
        for _ in range(12):
            step = Reference7DOF().predict(x,Controls(u[:2],u[2:]),mu=.5,load_memory=memory,horizon_s=.001,substeps=2)
            x,memory=step.state,step.load_memory
            expected.append(x.copy())
        np.testing.assert_allclose(predictions[0,0].detach().numpy(), expected, atol=1e-10)
        self.assertFalse(bad.any())

    def test_member_residual_matches_numpy_member_with_projection(self):
        model=self.model(); engine=TorchRollout(model)
        z=np.random.default_rng(6).normal(size=(2,5,16))
        actual=engine.residual(engine.tensor(z),engine.stacked_weights()).detach().numpy()
        model.refresh()
        for i in range(2):
            expected=model.predict(z[i])['members'][i]
            expected=np.clip(expected,-model.scaler['correction_limit'],model.scaler['correction_limit'])
            np.testing.assert_allclose(actual[i],expected,atol=1e-12)

    def test_multi_step_parameter_gradient_matches_finite_difference(self):
        model=self.model(zero=True); engine=TorchRollout(model)
        # Finite differences must be away from abs/max zero-slip kinks. The
        # separate equilibrium test checks finite subgradients at that kink.
        x=engine.tensor(np.tile([11.,.23,.031,31.1,31.6,31.3,31.8],(2,1,1)))
        u=engine.tensor(np.tile([.021,.026,10.,20.,-30.,-40.],(2,1,5,1)))
        memory=engine.tensor(np.tile([.2,-.4],(2,1,1))); mu=engine.tensor(np.full((2,1),.5))
        def loss():
            p,_=engine.predict(x,memory,u,mu)
            return p[0,0,-1,0]
        parameter=model.members[0][-1].bias
        loss().backward(); grad=float(parameter.grad[0]); eps=1e-5
        with torch.no_grad():
            parameter[0]+=eps; plus=float(loss())
            parameter[0]-=2*eps; minus=float(loss())
            parameter[0]+=eps
        self.assertAlmostEqual(grad,(plus-minus)/(2*eps),delta=2e-6)
        self.assertGreater(abs(grad),1e-5)

    def test_windows_never_bridge_anchor_targets_or_bad_timestamps(self):
        t=np.arange(3001)*.001
        fit=window_indices(t,np.ones(len(t),bool),100,25)
        hold=window_indices(t,np.ones(len(t),bool),100,25,anchor=True)
        fit_rows=set((fit[:,None]+np.arange(101)).ravel())
        hold_rows=set((hold[:,None]+np.arange(101)).ravel())
        self.assertFalse(fit_rows & hold_rows)
        self.assertGreater(len(hold),3)
        t[501:]+=.001
        changed=window_indices(t,np.ones(len(t),bool),100,25)
        self.assertFalse(any(i<=500<i+100 for i in changed))

    def test_rollout_checkpoint_prevents_derivative_only_online_updates(self):
        with tempfile.TemporaryDirectory() as d:
            m=self.model(); m.metadata['derivative_only_online_adaptation_allowed']=False
            x=np.zeros((30,16)); y=np.zeros((30,7)); path=Path(d)/'m.pt'
            m.save(path,x,y,x,y)
            self.assertFalse(OnlineShadow(path,adapt=False).adapt)
            with self.assertRaisesRegex(ValueError,'observation-only'):
                OnlineShadow(path,adapt=True)

    def test_window_preparation_never_reads_development_capture(self):
        m=self.model(); m.metadata.update(dataset='unused',dataset_sha256='dataset-hash')
        t=np.arange(2801)*.001
        x=np.tile([10.,0.,0.,*[10/.353]*4],(len(t),1))
        data={k:x[:,j] for j,k in enumerate(STATE_NAMES)}
        data.update(solver_time_s=t,return_code=np.zeros(len(t)),solver_done=np.zeros(len(t)),
                    front_steer_fl_rad=np.zeros(len(t)),front_steer_fr_rad=np.zeros(len(t)))
        for wheel in ('fl','fr','rl','rr'):
            for kind in ('drive','brake'):
                data[f'{kind}_{wheel}_nm']=np.zeros(len(t))
        manifest={'samples_sha256':'dataset-hash','episodes':[
            dict(split='train',capture='train-only',source_sha256={},id=0),
            dict(split='validation',capture='DO-NOT-READ',source_sha256={'metadata.json':'wrong'},id=1)]}
        meta=dict(scenario_mu_declared=.5,contract={'settings':{'MU_ROAD_CONSTANT':.5}})
        with patch('dynamics7dof.learning.rollout.load_dataset',return_value=({},manifest)), \
                patch('dynamics7dof.learning.rollout.load_capture',return_value=(meta,data)) as load:
            fit,hold,sources=prepare_windows(m,horizon=50,stride=25)
        self.assertEqual(load.call_count,1)
        self.assertEqual(str(load.call_args.args[0]),'train-only')
        self.assertEqual(len(sources),1)
        self.assertGreater(len(fit['x']),10)
        self.assertGreater(len(hold['x']),3)
        np.testing.assert_allclose(fit['truth'][:,0,:],fit['x'],atol=0)

    def test_profile_is_endpoint_member_metric_and_exclusive_output(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); model=self.model(zero=True); path=root/'base.pt'; model.save(path)
            x=np.tile([10.,0.,0.,*[10/.353]*4],(4,1))
            windows=dict(x=x,memory=np.zeros((4,2)),u=np.zeros((4,2,6)),
                         truth=np.tile(x[:,None,:],(1,2,1)),mu=np.full(4,.5),scale=np.ones((4,7)),block=np.arange(4))
            with patch('dynamics7dof.learning.rollout.prepare_windows',return_value=(windows,windows,[])):
                report=horizon_profile(path,root/'profile',horizon=2)
            self.assertEqual(report['rows'][0]['horizon_s'],.002)
            self.assertEqual(report['rows'][0]['member_window_count'],8)
            np.testing.assert_allclose(report['rows'][0]['learned_member_rmse_si'],0,atol=1e-10)
            with self.assertRaises(ValueError):
                horizon_profile(path,root/'profile',horizon=2)

    def test_tiny_training_saves_compatible_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); m=self.model(zero=True)
            path=root/'base.pt'; x=np.zeros((30,16)); y=np.zeros((30,7));m.save(path,x,y,x,y)
            states=np.tile([10.,0.,0.,*[10/.353]*4],(20,1))
            truth=np.tile(states[:,None,:],(1,2,1));truth[:,:,0]+=.0001
            windows=dict(x=states,memory=np.zeros((20,2)),u=np.zeros((20,2,6)),
                         truth=truth,mu=np.full(20,.5),scale=np.ones((20,7)),block=np.arange(20))
            with patch('dynamics7dof.learning.rollout.prepare_windows',return_value=(windows,windows,[])):
                r=train_rollout(path,root/'new',horizon=2,updates=2,batch_size=2)
            loaded,_=ResidualEnsemble.load(root/'new/ensemble.pt')
            self.assertEqual(loaded.metadata['training_objective'],'truncated_simulation_error')
            self.assertFalse(r['automatic_deployment'])


if __name__ == '__main__':
    unittest.main()
