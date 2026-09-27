"""Engineering verifier regression tests; no CarSim installation or DLL needed."""

from dataclasses import replace
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dynamics7dof.model import Controls, LoadMemory, Reference7DOF, STATE_NAMES, VehicleParameters
from dynamics7dof.physics_check import (
    initial_total_vehicle_profile, read_echo_parameters, shadow_diagnostics,
    sha, timing_and_torque, wheel_torque_imbalance,
)


class PhysicsCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='physics_check_')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def echo(self):
        return {'M_TL':2126.4,'IZZ_TL':5067.232381,'LX_AXLE(1)':0,'LX_AXLE(2)':3160,
                'LX_CG_TL':1290.030098,'H_CG_TL':549.7827314,
                **{f'ISPIN({a},{s})':3.23 for a,s in ((1,1),(1,2),(2,1),(2,2))}}

    def synthetic(self, turning=False):
        t=np.r_[0.,.0005+np.arange(250)*.001]
        state=np.r_[10.,0.,0.,np.full(4,10/.353)]
        controls=Controls([.01,.01] if turning else [0,0], [20]*4 if turning else [0]*4)
        model=Reference7DOF()
        memory=LoadMemory(0,0)
        states=[state.copy()]; memories=[memory]
        for dt in np.diff(t):
            p=model.predict(state,controls,mu=.5,load_memory=memory,horizon_s=float(dt),
                            substeps=max(1,int(np.ceil(dt/.0005-1e-8))))
            state,memory=p.state,p.load_memory
            states.append(state.copy()); memories.append(memory)
        x=np.array(states)
        data={k:x[:,i] for i,k in enumerate(STATE_NAMES)}
        data.update(solver_time_s=t,wrapper_time_s=np.arange(len(t))*.001,
                    return_code=np.zeros(len(t)),solver_done=np.zeros(len(t)),
                    front_steer_fl_rad=np.full(len(t),controls.front_steer_rad[0]),
                    front_steer_fr_rad=np.full(len(t),controls.front_steer_rad[1]),
                    ax_carsim_m_s2=np.array([m.ax_m_s2 for m in memories]),
                    ay_carsim_m_s2=np.array([m.ay_m_s2 for m in memories]))
        for i,w in enumerate(('fl','fr','rl','rr')):
            data[f'drive_{w}_nm']=np.full(len(t),controls.wheel_net_torque_nm[i])
            data[f'brake_{w}_nm']=np.zeros(len(t))
        return {'scenario_mu_declared':.5},data

    def independent_evidence(self):
        """Analytic time-dependent evidence, not generated with the 7DOF model."""
        t=np.r_[0.,.0005+np.arange(1200)*.001]
        ts=np.arange(1200)*.001
        data=dict(solver_time_s=t,wrapper_time_s=np.arange(len(t))*.001,
                  vx_m_s=10-.2*t,x_origin_m=10*t-.1*t*t,
                  front_steer_fl_rad=.001*t,return_code=np.zeros(len(t)),
                  solver_done=np.zeros(len(t)),brake_pressure_command_mpa=np.ones(len(t)))
        for w in ('fl','fr','rl','rr'):
            data[f'omega_{w}_rad_s']=(10-.2*t)/.353
            data[f'drive_{w}_nm']=np.zeros(len(t))
            data[f'brake_{w}_nm']=np.full(len(t),-100.)
        units={'Vx':'km/h','Xo':'m','Steer_L1':'deg',**{f'Fx_{w}':'N' for w in ('L1','R1','L2','R2')}}
        header={'VsChannelGroup':{'Channels':[{'Name Aliases':[k],'Units':v} for k,v in units.items()]}}
        (self.root/'LastRun.vs').write_text(json.dumps(header),encoding='utf-8')
        (self.root/'LastRun_all.par').write_text('synthetic fixture',encoding='utf-8')
        echo=self.echo()
        echo.update({f'RRE({a},{s})':353 for a,s in ((1,1),(1,2),(2,1),(2,2))})
        (self.root/'LastRun_echo.par').write_text('\n'.join(f'! {k} {v}' for k,v in echo.items()),encoding='utf-8')
        manifest={p.name:{'sha256':sha(p)} for p in self.root.iterdir()}
        (self.root/'source_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        with (self.root/'LastRun.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=['Time',*units])
            writer.writeheader()
            for tt in ts:
                row=dict(Time=tt,Vx=(10-.2*tt)*3.6,Xo=10*tt-.1*tt*tt,Steer_L1=np.degrees(.001*tt))
                row.update({f'Fx_{w}':(-100-3.23*(-.2/.353))/.353 for w in ('L1','R1','L2','R2')})
                writer.writerow(row)
        return {'contract':{'parameter_sha256':sha(self.root/'LastRun_all.par')}},data

    def test_signed_torque_balance_and_alternative_signs(self):
        drive=np.array([50.,20.]); brake=np.array([-200.,-100.])
        alpha=np.array([-3.,-2.]); radius=.353; inertia=3.23
        fx=(drive+brake-inertia*alpha)/radius
        good=wheel_torque_imbalance(drive,brake,fx,alpha,radius=radius,inertia=inertia)
        np.testing.assert_allclose(good,0,atol=1e-12)
        wrong=wheel_torque_imbalance(drive,brake,fx,alpha,radius=radius,inertia=inertia,brake_multiplier=-1)
        np.testing.assert_allclose(wrong,-2*brake)
        ignored=wheel_torque_imbalance(drive,brake,fx,alpha,radius=radius,inertia=inertia,brake_multiplier=0)
        np.testing.assert_allclose(ignored,-brake)

    def test_independent_clock_and_torque_check(self):
        meta,data=self.independent_evidence()
        report,profile=timing_and_torque(meta,data,self.root)
        self.assertEqual(report['clock_first_interval_s'],.0005)
        self.assertLess(report['alignment']['Vx']['rms_by_shift_s']['0'],1e-10)
        self.assertGreater(report['torques']['fl']['alternative_to_plus_min_rmse_ratio'],1000)
        self.assertEqual(profile.wheel_inertia_kg_m2,3.23)

    def test_extra_half_step_shift_fails_alignment(self):
        meta,data=self.independent_evidence()
        data['solver_time_s']=data['wrapper_time_s'].copy()
        with self.assertRaisesRegex(ValueError,'timing is inconsistent'):
            timing_and_torque(meta,data,self.root)

    def test_unsigned_brake_rejected_in_analytic_balance(self):
        meta,data=self.independent_evidence(); data['brake_fl_nm']*=-1
        with self.assertRaisesRegex(ValueError,'torque sign'):
            timing_and_torque(meta,data,self.root)

    def test_evidence_mutation_and_parameter_mismatch_rejected(self):
        meta,data=self.independent_evidence()
        meta['contract']['parameter_sha256']='not the same vehicle'
        with self.assertRaisesRegex(ValueError,'parameters differ'):
            timing_and_torque(meta,data,self.root)
        (self.root/'LastRun_echo.par').write_text('mutated',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'snapshot hash mismatch'):
            timing_and_torque(meta,data,self.root)

    def test_invalid_wheel_parameters_rejected(self):
        for r,j in [(0,2),(.3,-2),(float('nan'),2),(.3,float('inf'))]:
            with self.assertRaises(ValueError):
                wheel_torque_imbalance(0,0,0,0,radius=r,inertia=j)

    def test_echo_parser_reads_calculated_and_indexed_values(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'echo.par'
            p.write_bytes(b'! M_TL 2126.4 ; kg\n  ! ISPIN(1,1) 3.23 ; kg-m^2\nM_SU 1820\n! unrelated prose\n')
            v=read_echo_parameters(p)
        self.assertEqual(v['M_TL'],2126.4)
        self.assertEqual(v['ISPIN(1,1)'],3.23)
        self.assertEqual(v['M_SU'],1820)

    def test_proxy_is_explicit_and_original_parameters_unchanged(self):
        p=initial_total_vehicle_profile(self.echo())
        self.assertEqual(p.mass_kg,2126.4)
        self.assertAlmostEqual(p.wheelbase_m,3.16)
        self.assertEqual(p.wheel_inertia_kg_m2,3.23)
        self.assertEqual(VehicleParameters().mass_kg,2026.)
        self.assertEqual(VehicleParameters().wheel_inertia_kg_m2,2.)

    def test_unequal_inertias_rejected(self):
        e=self.echo(); e['ISPIN(2,2)']=4
        with self.assertRaises(ValueError):
            initial_total_vehicle_profile(e)

    def test_equilibrium_and_first_half_step(self):
        meta,data=self.synthetic()
        before=data['solver_time_s'].copy()
        r=shadow_diagnostics(meta,data,VehicleParameters())
        self.assertEqual(r['time_basis'],'observed_solver_T')
        self.assertLess(max(r['policies']['carried']['state_rmse'].values()),1e-12)
        self.assertEqual(r['fifty_ms_measured_future_input_replay']['windows'],1)
        np.testing.assert_array_equal(before,data['solver_time_s'])

    def test_carried_memory_matches_generated_trajectory(self):
        meta,data=self.synthetic(turning=True)
        r=shadow_diagnostics(meta,data,VehicleParameters())
        self.assertLess(max(r['policies']['carried']['state_rmse'].values()),1e-12)
        self.assertGreater(max(r['policies']['zero_each_step']['state_rmse'].values()),1e-8)
        self.assertGreater(r['policies']['zero_each_step']['raw_fz_difference_vs_carried_rmse_n']['fl'],1)
        self.assertEqual(r['policies']['carried']['raw_fz_difference_vs_carried_rmse_n']['fl'],0)

    def test_partial_clock_is_not_silently_replaced(self):
        meta,data=self.synthetic(); data['solver_time_s'][10]=np.nan
        with self.assertRaisesRegex(ValueError,'solver clock'):
            shadow_diagnostics(meta,data,VehicleParameters())

    def test_legacy_clock_explicitly_marked(self):
        meta,data=self.synthetic(); del data['solver_time_s']
        r=shadow_diagnostics(meta,data,VehicleParameters())
        self.assertIn('no_absolute_clock_claim',r['time_basis'])
        self.assertTrue(r['not_training_labels'])

    def test_nonmonotonic_and_wrong_timestep_rejected(self):
        meta,data=self.synthetic(); data['solver_time_s'][20]=data['solver_time_s'][19]
        with self.assertRaisesRegex(ValueError,'increasing'):
            shadow_diagnostics(meta,data,VehicleParameters())
        meta,data=self.synthetic(); data['solver_time_s']*=2
        with self.assertRaisesRegex(ValueError,'1 ms'):
            shadow_diagnostics(meta,data,VehicleParameters())

    def test_bad_state_and_stride_rejected(self):
        meta,data=self.synthetic()
        with self.assertRaises(ValueError):
            shadow_diagnostics(meta,data,VehicleParameters(),stride=0)
        data['vx_m_s'][30]=np.nan
        with self.assertRaisesRegex(ValueError,'Nonfinite state'):
            shadow_diagnostics(meta,data,VehicleParameters())

    def test_error_window_excluded_from_multistep_replay(self):
        meta,data=self.synthetic(); data['return_code'][125]=1
        r=shadow_diagnostics(meta,data,VehicleParameters())
        self.assertEqual(r['fifty_ms_measured_future_input_replay'],{})

    def test_coarse_step_flags_retained_in_refinement_check(self):
        meta,data=self.synthetic()
        predict=Reference7DOF.predict
        def flagged(model,*args,**kwargs):
            result=predict(model,*args,**kwargs)
            if kwargs['substeps']==2:
                return replace(result,flags=('synthetic_coarse_flag',))
            return result
        with patch.object(Reference7DOF,'predict',flagged):
            r=shadow_diagnostics(meta,data,VehicleParameters())
        self.assertEqual(r['fifty_ms_measured_future_input_replay']['flagged_windows'],1)


if __name__=='__main__':
    unittest.main()
