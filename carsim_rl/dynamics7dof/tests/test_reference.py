"""Deterministic reference/invariant tests, not a CarSim accuracy benchmark."""

import unittest

import numpy as np

from dynamics7dof import Controls, LoadMemory, Reference7DOF, TireFit, VehicleParameters


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.model = Reference7DOF()
        self.v = self.model.vehicle
        self.memory = LoadMemory(0.0, 0.0)
        self.controls = Controls([0.0, 0.0], [0.0] * 4)
        self.x = np.r_[40.0 / 3.6, 0.0, 0.0,
                       np.full(4, (40.0 / 3.6) / self.v.wheel_radius_m)]

    def evaluate(self, state=None, controls=None, mu=0.5, memory=None):
        return self.model.evaluate(
            self.x if state is None else state,
            self.controls if controls is None else controls,
            mu=mu, load_memory=self.memory if memory is None else memory,
        )

    def test_user_parameters_and_fit_preserved(self):
        self.assertEqual(self.v.mass_kg, 2026.0)
        self.assertAlmostEqual(self.v.wheelbase_m, 3.16)
        self.assertEqual(self.v.yaw_inertia_kg_m2, 4095.0)
        self.assertEqual(self.v.wheel_inertia_kg_m2, 2.0)
        self.assertEqual(self.v.wheel_radius_m, 0.353)
        self.assertEqual(self.model.tire.camber_fit, 1.9554)

    def test_straight_free_rolling_is_equilibrium_in_reference(self):
        # No aerodynamic/rolling resistance is present in the source model.
        np.testing.assert_allclose(self.evaluate().derivative, 0.0, atol=1e-10)
        result = self.model.predict(self.x, self.controls, mu=0.5,
                                    load_memory=self.memory, horizon_s=0.05)
        np.testing.assert_allclose(result.state, self.x, atol=1e-10)
        self.assertEqual(result.flags, ())

    def test_static_load_and_front_rear_distribution(self):
        e = self.evaluate()
        total = self.v.mass_kg * self.v.gravity_m_s2
        self.assertAlmostEqual(e.raw_fz_n.sum(), total)
        self.assertAlmostEqual(e.fz_n[:2].sum(), total * self.v.lr_m / self.v.wheelbase_m)
        self.assertAlmostEqual(e.fz_n[2:].sum(), total * self.v.lf_m / self.v.wheelbase_m)

    def test_load_transfer_direction_and_unclamped_sum(self):
        static = self.evaluate()
        e = self.evaluate(memory=LoadMemory(1.0, 2.0))
        self.assertLess(e.raw_fz_n[:2].sum(), static.raw_fz_n[:2].sum())
        self.assertGreater(e.raw_fz_n[2:].sum(), static.raw_fz_n[2:].sum())
        self.assertLess(e.raw_fz_n[0], e.raw_fz_n[1])
        self.assertLess(e.raw_fz_n[2], e.raw_fz_n[3])
        self.assertAlmostEqual(e.raw_fz_n.sum(), static.raw_fz_n.sum())

    def test_original_lateral_fit_numeric_value(self):
        _, fy = self.model.tire_forces(np.zeros(4), np.full(4, np.deg2rad(1.0)),
                                       np.full(4, 4000.0), mu=1.0)
        # Independently recomputed from the supplied MATLAB coefficients.
        np.testing.assert_allclose(fy, -1079.5269466020852, rtol=1e-12)

    def test_known_small_slip_attenuation_is_preserved_not_fixed(self):
        fx, fy = self.model.tire_forces(np.full(4, 0.001), np.full(4, 0.001),
                                       np.full(4, 4000.0), mu=1.0)
        np.testing.assert_allclose(fx, 70.68088698766009, rtol=1e-12)
        np.testing.assert_allclose(fy, -44.70224222697554, rtol=1e-12)

    def test_friction_bound_and_per_wheel_mu(self):
        mu = np.array([0.0, 0.2, 0.5, 1.0])
        fz = np.full(4, 4000.0)
        fx, fy = self.model.tire_forces([0.2] * 4, [0.1] * 4, fz, mu=mu)
        self.assertTrue(np.all(np.hypot(fx, fy) <= mu * fz + 1e-10))
        self.assertEqual(fx[0], 0.0)
        self.assertEqual(fy[0], 0.0)

    def test_positive_steer_generates_positive_yaw(self):
        e = self.evaluate(controls=Controls([0.02, 0.02], [0.0] * 4))
        self.assertGreater(e.derivative[2], 0.0)
        self.assertGreater(e.sum_fy_body_n, 0.0)

    def test_mirrored_turns(self):
        results = []
        for sign in (1, -1):
            c = Controls([sign * 0.02] * 2, [0.0] * 4)
            results.append(self.model.predict(self.x, c, mu=0.5, load_memory=self.memory,
                                               horizon_s=0.05).state)
        left, right = results
        np.testing.assert_allclose(left[:3], right[:3] * [1, -1, -1], atol=1e-10)
        np.testing.assert_allclose(left[3:], right[[4, 3, 6, 5]], atol=1e-10)

    def test_expanded_matlab_yaw_moment_matches_cross_products(self):
        c = Controls([0.03, 0.02], [100.0, 90.0, 80.0, 70.0])
        x = self.x.copy()
        x[1:3] = [0.2, 0.1]
        e = self.evaluate(x, c)
        a, b, d = self.v.lf_m, self.v.lr_m, self.v.track_front_m
        dl, dr = c.front_steer_rad
        fl, fr, rl, rr = e.fx_tire_n
        yl, yr, zl, zr = e.fy_tire_n
        expected = (d/2 * (fr*np.cos(dr)-fl*np.cos(dl)+yl*np.sin(dl)-yr*np.sin(dr))
                    + a * (fr*np.sin(dr)+fl*np.sin(dl)+yr*np.cos(dr)+yl*np.cos(dl))
                    + d/2 * (rr-rl) - b * (zr+zl))
        self.assertAlmostEqual(e.yaw_moment_nm, expected, places=9)

    def test_wheel_torque_balance_and_acceleration_semantics(self):
        c = Controls([0.01, 0.02], [100.0, 200.0, -30.0, -40.0])
        x = self.x.copy()
        x[1:3] = [0.3, 0.2]
        e = self.evaluate(x, c)
        np.testing.assert_allclose(e.derivative[3:],
            (c.wheel_net_torque_nm-e.fx_tire_n*self.v.wheel_radius_m)/self.v.wheel_inertia_kg_m2)
        self.assertAlmostEqual(e.derivative[0], e.next_load_memory.ax_m_s2 + x[2]*x[1])
        expected_vy_dot = (e.next_load_memory.ay_m_s2 - x[2]*x[0]
                           - 100*np.exp(-x[1]**2)/self.v.mass_kg*x[1])
        self.assertAlmostEqual(e.derivative[1], expected_vy_dot)

    def test_force_timestamp_and_one_substep_euler(self):
        c = Controls([0.02] * 2, [0.0] * 4)
        initial_eval = self.evaluate(controls=c)
        result = self.model.predict(self.x, c, mu=0.5, load_memory=self.memory,
                                    horizon_s=0.001, substeps=1)
        np.testing.assert_allclose(result.state, self.x + 0.001*initial_eval.derivative)
        np.testing.assert_array_equal(result.last_force_state, self.x)
        self.assertEqual(result.force_sample_time_s, 0.0)
        longer = self.model.predict(self.x, c, mu=0.5, load_memory=self.memory,
                                    horizon_s=0.05, substeps=100)
        self.assertEqual(longer.substep_s, 0.0005)
        self.assertAlmostEqual(longer.force_sample_time_s, 0.0495)

    def test_split_rollout_matches_only_when_memory_carried(self):
        c = Controls([0.02] * 2, [0.0] * 4)
        whole = self.model.predict(self.x, c, mu=0.5, load_memory=self.memory,
                                   horizon_s=0.05, substeps=100)
        first = self.model.predict(self.x, c, mu=0.5, load_memory=self.memory,
                                   horizon_s=0.025, substeps=50)
        second = self.model.predict(first.state, c, mu=0.5, load_memory=first.load_memory,
                                    horizon_s=0.025, substeps=50)
        np.testing.assert_allclose(whole.state, second.state, atol=1e-12)

    def test_acceleration_memory_changes_same_state_prediction(self):
        c = Controls([0.02] * 2, [0.0] * 4)
        one = self.evaluate(controls=c)
        two = self.evaluate(controls=c, memory=LoadMemory(2.0, 3.0))
        self.assertGreater(np.max(np.abs(one.derivative-two.derivative)), 1e-5)

    def test_load_floor_and_state_clamps_are_exposed(self):
        e = self.evaluate(memory=LoadMemory(0.0, 20.0))
        self.assertIn("legacy_load_floor_active", e.flags)
        self.assertIn("contact_loss_outside_model_domain", e.flags)
        self.assertGreater(e.fz_n.sum(), e.raw_fz_n.sum())
        x = np.r_[np.zeros(3), [-3.0] * 4]
        result = self.model.predict(x, self.controls, mu=0.0, load_memory=self.memory,
                                    horizon_s=0.001, substeps=1)
        self.assertIn("legacy_vx_clamp_active", result.flags)
        self.assertIn("legacy_wheel_speed_clamp_active", result.flags)
        self.assertEqual(result.state[0], 0.1)
        np.testing.assert_array_equal(result.state[3:], -np.ones(4))

    def test_inputs_are_not_mutated_and_column_states_work(self):
        before = self.x.copy()
        self.model.predict(self.x.reshape(7, 1), self.controls, mu=0.5,
                           load_memory=self.memory, horizon_s=0.05)
        np.testing.assert_array_equal(self.x, before)
        self.assertFalse(self.controls.front_steer_rad.flags.writeable)

    def test_explicit_mu_and_load_memory_required(self):
        with self.assertRaises(TypeError):
            self.model.evaluate(self.x, self.controls, load_memory=self.memory)
        with self.assertRaises(TypeError):
            self.model.evaluate(self.x, self.controls, mu=0.5)

    def test_invalid_inputs_fail_fast(self):
        for mu in (-1, float("nan"), [0.5, 0.5]):
            with self.subTest(mu=mu), self.assertRaises(ValueError):
                self.evaluate(mu=mu)
        with self.assertRaises(ValueError):
            self.evaluate(state=np.ones((7, 7)))
        with self.assertRaises(ValueError):
            self.evaluate(state=[float("nan")] * 7)
        with self.assertRaises(ValueError):
            Controls([0, 90], [0] * 4)
        with self.assertRaises(ValueError):
            VehicleParameters(mass_kg=0)
        with self.assertRaises(ValueError):
            TireFit(camber_fit=float("nan"))
        for steps in (0, -1, 1.5, True):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                self.model.predict(self.x, self.controls, mu=0.5, load_memory=self.memory,
                                   horizon_s=0.05, substeps=steps)
        with self.assertRaises(ValueError):
            self.model.predict(self.x, self.controls, mu=0.5, load_memory=self.memory,
                               horizon_s=0.0)


if __name__ == "__main__":
    unittest.main()
