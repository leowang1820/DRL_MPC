# -*- coding: utf-8 -*-
"""MATLAB NMPC 三通道参考动力学的无 DLL 单元测试。"""

import math
import unittest

import numpy as np

from wm_recovery.nmpc_reference import (
    NMPCReferenceParams, nmpc_force_balance, nmpc_accelerations,
    rk4_nominal_step, NMPC_Z_NAMES, NMPC_Y_NAMES,
)


class NMPCReferenceTests(unittest.TestCase):
    def test_matlab_physical_parameters(self):
        p = NMPCReferenceParams()
        self.assertAlmostEqual(p.mass_kg, 2053.8)
        self.assertAlmostEqual(p.lr_m, 1.895)
        self.assertEqual(len(NMPC_Z_NAMES), 8)
        self.assertEqual(len(NMPC_Y_NAMES), 3)
        p.validate()

    def test_zero_torque_zero_angle_zero_lateral_state(self):
        p = NMPCReferenceParams()
        z = np.array([0.0,0.0,16.0,0.0,0.0,0.0,0.5,0.8])
        np.testing.assert_allclose(nmpc_accelerations(z,p), np.zeros(3), atol=1e-11)
        np.testing.assert_allclose(rk4_nominal_step(z,0.01,p), z[:3], atol=1e-11)

    def test_only_external_yaw_moment(self):
        p = NMPCReferenceParams()
        z = np.array([0.0,0.0,10.0,0.0,1250.0,0.0,0.5,0.8])
        deriv = nmpc_accelerations(z,p)
        self.assertAlmostEqual(deriv[0],0.0,places=11)
        self.assertAlmostEqual(deriv[1],1250/p.inertia_z_kgm2,places=12)
        self.assertAlmostEqual(deriv[2],0.0,places=11)

    def test_pure_longitudinal_force_at_zero_angle(self):
        p = NMPCReferenceParams()
        z = np.array([0.0,0.0,18.0,0.0,0.0,1000.0,0.4,0.8])
        deriv=nmpc_accelerations(z,p)
        self.assertAlmostEqual(deriv[0],0.0,places=11)
        self.assertAlmostEqual(deriv[1],0.0,places=11)
        self.assertAlmostEqual(deriv[2],1000.0/p.mass_kg,places=11)
        stepped=rk4_nominal_step(z,dt_s=0.01,params=p)
        self.assertAlmostEqual(stepped[2],18.0 + 0.01*1000.0/p.mass_kg,places=10)

    def test_independent_scalar_matlab_translation(self):
        """独立按 MATLAB 顺序写一遍标量公式，防止参考实现符号/常量写错。"""
        z=[1.2,0.15,20.0,0.08,250.0,3000.0,0.7,0.8]
        vy,r,vx,delta,mz,fx_all,lam,mu=z
        lf, lr, L, g, m, Iz, h = 1.265, 3.16-1.265, 3.16, 9.81, 2053.8, 4095.0, 0.59
        fx_f=fx_all*lam
        fx_r=fx_all*(1.0-lam)
        fz_f=(m*g*lr-h*fx_all)/L
        fz_r=(m*g*lf+h*fx_all)/L
        fz_f=1000 + 0.5*((fz_f-1000)+math.sqrt((fz_f-1000)**2+10000))
        fz_r=1000 + 0.5*((fz_r-1000)+math.sqrt((fz_r-1000)**2+10000))
        lim_f=mu*fz_f
        lim_r=mu*fz_r
        mf=lim_f**2 - fx_f**2
        mr=lim_r**2 - fx_r**2
        mf_sm=0.5*(mf+math.sqrt(mf**2+1e6))
        mr_sm=0.5*(mr+math.sqrt(mr**2+1e6))
        cap_f=math.sqrt(mf_sm)+0.001
        cap_r=math.sqrt(mr_sm)+0.001
        kyf=8735.0825*(fz_f/2000)-59.6128*(fz_f/2000)**2
        kyr=8735.0825*(fz_r/2000)-59.6128*(fz_r/2000)**2
        vs=math.sqrt(vx**2+0.01)
        af=delta-(vy+lf*r)/vs
        ar=(lr*r-vy)/vs
        fyf=cap_f*math.tanh((2*kyf*af)/cap_f)
        fyr=cap_r*math.tanh((2*kyr*ar)/cap_r)
        sum_fx=fx_f*math.cos(delta)-fyf*math.sin(delta)+fx_r
        sum_fy=fyf*math.cos(delta)+fx_f*math.sin(delta)+fyr
        sum_mz=lf*(fyf*math.cos(delta)+fx_f*math.sin(delta))-lr*fyr+mz
        expected=[-vx*r+sum_fy/m, sum_mz/Iz, sum_fx/m+vy*r]
        np.testing.assert_allclose(nmpc_accelerations(z),expected, rtol=0,atol=3e-12)
        forces=nmpc_force_balance(z)
        self.assertAlmostEqual(forces["fy_front_n"],fyf,places=10)
        self.assertAlmostEqual(forces["sum_mz_nm"],sum_mz,places=10)

    def test_batch_vectorization_matches_scalar_and_rk4_steps(self):
        rows = np.array([
            [0.5,0.03,11.1,0.04,40,0,0.5,0.8],
            [1.0,0.09,16.6,0.1,-200,2000,0.7,0.8],
            [-0.3,-0.01,22.2,-0.03,100,-500,-0.2,0.8],
        ])
        dt=np.array([0.01,0.01,0.02])
        bulk=nmpc_accelerations(rows)
        per=np.stack([nmpc_accelerations(row) for row in rows])
        np.testing.assert_allclose(bulk,per,rtol=0,atol=1e-12)
        stepped=rk4_nominal_step(rows,dt_s=dt,substeps=4)
        per_step=np.stack([
            rk4_nominal_step(row,float(h),substeps=4) for row,h in zip(rows,dt)
        ])
        np.testing.assert_allclose(stepped,per_step,rtol=0,atol=1e-12)

    def test_invalid_inputs_and_interval(self):
        for bad in [np.zeros((2,7)),np.zeros((0,8)),np.zeros((2,2,8))]:
            with self.assertRaises(ValueError):
                nmpc_accelerations(bad)
        z=np.array([0,0,10,0,0,0,0.5,0.8],dtype=float)
        z[3]=np.nan
        with self.assertRaises(ValueError):
            nmpc_accelerations(z)
        z[3]=0
        z[-1]=-0.2
        with self.assertRaises(ValueError):
            nmpc_accelerations(z)
        z[-1]=0.8
        with self.assertRaises(ValueError):
            rk4_nominal_step(z,-0.01)
        with self.assertRaises(ValueError):
            rk4_nominal_step(z,0.01,substeps=0)
        with self.assertRaises(ValueError):
            rk4_nominal_step(np.tile(z,(2,1)),np.array([0.01]))


if __name__ == "__main__":
    unittest.main()
