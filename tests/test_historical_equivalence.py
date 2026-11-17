import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "historical"))
sys.path.insert(0, str(ROOT / "historical" / "thesis_eval" / "mujoco_g1_12dof"))

from controller import CycleIntegralController
from sp_control import OuterLoop


class HistoricalEquivalenceTests(unittest.TestCase):
    def test_fixed_reference_cycle_updates_match_historical_outer_loop(self):
        new = CycleIntegralController(0.2058, 0.0, 2.0)
        historical = OuterLoop(dict(type="outer", gamma=0.2058, u_min=0.0, u_max=2.0, t_on_s=2.0))
        historical.reset(1.0)
        for step in range(1, 281):
            time_s = step * 0.02
            velocity = 0.6 if step <= 160 else 1.3
            output = historical.observe(velocity, step, time_s)
            if output is not None:
                expected = new.update(1.0, historical.log[-1][1], time_s)
                self.assertAlmostEqual(output, expected)


if __name__ == "__main__":
    unittest.main()
