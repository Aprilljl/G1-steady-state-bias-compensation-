"""Checks for release documentation scope and safety wording."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReleaseDocumentationTests(unittest.TestCase):
    def test_index_describes_retained_scope(self):
        text = (ROOT / "docs/experiment_index.md").read_text(encoding="utf-8")
        for phrase in ("离线复算数据未收录", "单训练 seed", "真值速度", "RMSE", "Floquet", "Isaac Gym", "simulation_gait_sequence.png"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)
        self.assertTrue((ROOT / "historical/gait_illustration/simulation_gait_sequence.png").is_file())
        self.assertNotIn("recompute_release.py", text)

    def test_licenses_and_rights_boundaries(self):
        text = (ROOT / "DATA_LICENSE.md").read_text(encoding="utf-8")
        for phrase in ("CC BY 4.0", "F/G/R checkpoints", "owns and has the right to license", "third-party", "BSD 3-Clause", "repository includes these checkpoints"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)
        for path in ("LICENSE", "historical/LICENSE", "historical/isaacgym_release/overlay/LICENSE", "historical/isaacgym_release/overlay/legged_gym/LICENSE"):
            self.assertTrue((ROOT / path).is_file(), path)
        for policy in ("F", "G", "R"):
            self.assertTrue((ROOT / "checkpoints" / policy / "model_10000.pt").is_file())

    def test_release_docs_do_not_contain_absolute_windows_paths(self):
        for relative in ("docs/experiment_index.md", "docs/release_checklist.md", "THIRD_PARTY_NOTICES.md"):
            self.assertNotRegex((ROOT / relative).read_text(encoding="utf-8"), r"[A-Za-z]:[\\/]")


if __name__ == "__main__":
    unittest.main()
