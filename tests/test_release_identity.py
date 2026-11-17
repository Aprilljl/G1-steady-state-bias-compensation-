import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReleaseIdentityTests(unittest.TestCase):
    def test_project_identity_and_scientific_caveats_are_explicit(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        provenance = (ROOT / "PROVENANCE.md").read_text(encoding="utf-8")
        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")

        self.assertIn("g1-steady-state-bias-compensation", readme)
        self.assertIn("Steady-State Bias Compensation for Learned Bipedal Locomotion", citation)
        self.assertIn("unpublished", citation)
        self.assertIn("original SP-A3 periodic-orbit precheck failed", provenance)
        self.assertIn("FTLE/return-map association with fall rate is not causal evidence", provenance)
        self.assertIn("steady bias does not prove absence of integral action", provenance)
        self.assertIn("final constant command approximation", provenance)
        self.assertIn("g1-steady-state-bias-compensation", readme)

    def test_bsd_license_and_authorization_boundaries(self):
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        for phrase in ("BSD 3-Clause License", "Copyright (c) 2026, Jinlong Liu",
                       "1. Redistributions of source code",
                       "2. Redistributions in binary form",
                       "3. Neither the name of the copyright holder",
                       "specific prior written permission",
                       "THIS SOFTWARE IS PROVIDED", "AS IS"):
            self.assertIn(phrase, license_text)
        for name in ("README.md", "PROVENANCE.md"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertIn("BSD 3-Clause", text)
            self.assertIn("authorized public disclosure", text)
            self.assertIn("reuse terms", text)

    def test_readme_documentation_links_exist(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        expected_docs = (
            "docs/isaacgym_reproduction.md",
            "docs/mujoco_reproduction.md",
            "PROVENANCE.md",
            "DATA_LICENSE.md",
        )
        for relative_path in expected_docs:
            self.assertIn(relative_path, readme)
            self.assertTrue((ROOT / relative_path).is_file(), relative_path)

    def test_private_absolute_paths_are_not_published_in_release_docs(self):
        for name in ("README.md", "PROVENANCE.md", "CITATION.cff"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertNotIn("D:\\", text)
            self.assertNotIn("C:\\Users\\", text)


if __name__ == "__main__":
    unittest.main()
