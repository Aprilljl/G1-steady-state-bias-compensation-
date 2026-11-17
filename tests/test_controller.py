import math
import unittest

from controller import CycleIntegralController


class CycleIntegralControllerTests(unittest.TestCase):
    def test_waits_for_activation_and_keeps_state(self):
        controller = CycleIntegralController(0.2, 0.0, 2.0)
        self.assertEqual(controller.update(1.0, 0.5, 1.99), 1.0)
        self.assertEqual(controller.integral, 0.0)
        self.assertAlmostEqual(controller.update(1.0, 0.5, 2.0), 1.1)
        self.assertAlmostEqual(controller.update(1.0, 0.5, 2.8), 1.2)

    def test_upper_saturation_does_not_wind_up(self):
        controller = CycleIntegralController(1.0, 0.0, 1.5)
        self.assertEqual(controller.update(1.0, 0.0, 2.0), 1.5)
        self.assertEqual(controller.update(1.0, 0.0, 2.8), 1.5)
        self.assertAlmostEqual(controller.update(1.0, 1.2, 3.6), 1.3)

    def test_lower_saturation_and_reference_above_cap(self):
        controller = CycleIntegralController(1.0, 0.0, 1.75)
        self.assertEqual(controller.update(1.0, 3.0, 2.0), 0.0)
        controller.reset()
        self.assertEqual(controller.update(2.0, 2.0, 2.8), 1.75)

    def test_reset_clears_integral(self):
        controller = CycleIntegralController(0.2058, 0.0, 2.0)
        controller.update(1.0, 0.5, 2.0)
        controller.reset()
        self.assertEqual(controller.integral, 0.0)

    def test_invalid_boundary_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            CycleIntegralController(-1.0, 0.0, 2.0)
        with self.assertRaises(ValueError):
            CycleIntegralController(0.2, 2.0, 1.0)
        controller = CycleIntegralController(0.2, 0.0, 2.0)
        with self.assertRaises(ValueError):
            controller.update(1.0, math.nan, 2.0)
        with self.assertRaises(ValueError):
            controller.update(1.0, 0.5, -1.0)


if __name__ == "__main__":
    unittest.main()
