import math
import unittest

from analysis import conditional_survival, constant_hazard, tracking_decomposition


class TrackingTests(unittest.TestCase):
    def test_planar_rmse_identity_includes_lateral_bias(self):
        result = tracking_decomposition([0.6, 1.0], [0.1, 0.3], 1.0)
        self.assertAlmostEqual(result["forward_bias"], -0.2)
        self.assertAlmostEqual(result["lateral_bias"], 0.2)
        self.assertAlmostEqual(result["ripple"] ** 2, 0.05)
        self.assertAlmostEqual(result["planar_rmse"] ** 2, 0.13)
        self.assertAlmostEqual(result["planar_rmse"] ** 2,
                               result["forward_bias"] ** 2 + result["lateral_bias"] ** 2 + result["ripple"] ** 2)

    def test_invalid_velocity_input(self):
        for vx, vy, command in [([], [], 1), ([1], [], 1), ([math.nan], [0], 1), ([1], [0], math.inf)]:
            with self.assertRaises(ValueError):
                tracking_decomposition(vx, vy, command)


class HazardTests(unittest.TestCase):
    def test_point_estimate_and_conditional_survival(self):
        hazard = constant_hazard(5, 100.0)
        self.assertAlmostEqual(hazard, 0.05)
        self.assertAlmostEqual(conditional_survival(hazard, 20.0), math.exp(-1))
        self.assertEqual(conditional_survival(constant_hazard(0, 100), 20), 1.0)

    def test_invalid_hazard_input(self):
        for deaths, exposure in [(-1, 1), (1.5, 1), (True, 1), (1, 0), (1, math.inf)]:
            with self.assertRaises(ValueError):
                constant_hazard(deaths, exposure)
        with self.assertRaises(ValueError):
            conditional_survival(-0.1, 1)


if __name__ == "__main__":
    unittest.main()
