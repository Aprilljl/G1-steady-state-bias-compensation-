import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "historical" / "isaacgym_release"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IsaacGymReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = RELEASE / "configs" / "short_paper_isaacgym.yaml"
        cls.scenarios = load_module("release_scenarios", RELEASE / "scenarios.py")
        cls.arch = load_module("release_arch", RELEASE / "policy_arch.py")

    def _registry(self, tmp):
        path = Path(tmp) / "registry.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["matched_run", "final_checkpoint", "final_sha256"])
            writer.writeheader()
            writer.writerow({"matched_run": "Mar11_13-47-46_", "final_checkpoint": "F.pt", "final_sha256": "f" * 64})
        return path

    def test_design_preserves_registered_windows_and_grids(self):
        import yaml
        cfg = yaml.safe_load(self.config.read_text(encoding="utf-8"))
        self.assertEqual(cfg["experiments"]["SP-A2_speed_hazard_plane"]["episode_length_s"], 60.0)
        self.assertEqual(cfg["experiments"]["SP-B1g_static_map_isaac"]["episode_length_s"], 30.0)
        self.assertEqual(len(cfg["experiments"]["SP-A2_speed_hazard_plane"]["grid"]["command_vx"]), 13)
        self.assertEqual(cfg["experiments"]["SP-B1g_static_map_isaac"]["grid"]["payload_kg"], [0, 3, 5, 8, 10, 15])

    def test_expansion_maps_policy_and_condition_counts(self):
        with tempfile.TemporaryDirectory() as d:
            registry = self._registry(d)
            a2 = self.scenarios.expand(self.config, registry, Path(d) / "out", ["SP-A2"])
            b1 = self.scenarios.expand(self.config, registry, Path(d) / "out", ["SP-B1g"])
        self.assertEqual(len(a2), 1)
        self.assertEqual(len(a2[0]["envs"]), 13 * 100)
        self.assertEqual(a2[0]["episode_length_s"], 60.0)
        self.assertEqual(len(b1), 1)
        self.assertEqual(len(b1[0]["envs"]), 6 * 9 * 20)
        self.assertTrue(all(e["terrain"]["name"] == "plane" for e in b1))
        self.assertEqual(a2[0]["policy_id"], "Mar11_13-47-46_")

    def test_missing_registry_is_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            registry = Path(d) / "registry.csv"
            registry.write_text("matched_run,final_checkpoint,final_sha256\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                self.scenarios.expand(self.config, registry, Path(d) / "out", ["SP-A2"])
        self.assertIn("policies not found", str(ctx.exception))

    def test_paper_mlp_rejects_upstream_lstm(self):
        self.assertTrue(self.arch.validate_architecture({
            "policy_class_name": "ActorCritic", "actor_hidden_dims": "[512, 256, 128]", "rnn_type": ""}))
        with self.assertRaises(ValueError):
            self.arch.validate_architecture({
                "policy_class_name": "ActorCriticRecurrent", "actor_hidden_dims": "[32]", "rnn_type": "lstm"})

    def test_manifest_records_unverified_boundaries(self):
        manifest = json.loads((RELEASE / "source_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["private_source"]["commit"], "8df0dfc8669620b5ee058dbe117fe92b653aafc6")
        self.assertIn("Isaac Gym is absent on the Windows host", manifest["unverified"])
        self.assertIn("private_source/legged_gym/scripts/eval_batch.py", manifest["private_patch_closure"])
        self.assertEqual(len(manifest["source_hashes_sha256_raw_bytes"]), 9)

    def test_runtime_boundary_is_import_safe_and_explicit(self):
        runtime = load_module("release_eval_batch", RELEASE / "eval_batch.py")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "spec.json"
            path.write_text(json.dumps({
                "launch_id": "x", "policy_id": "F", "checkpoint": "x.pt",
                "checkpoint_sha256": "a" * 64,
                "terrain": {"mesh_type": "plane"}, "episode_length_s": 60.0, "envs": []}), encoding="utf-8")
            spec = runtime.validate_spec(path)
        self.assertEqual(spec["policy_id"], "F")
        with self.assertRaises(RuntimeError):
            runtime.main()

    def test_overlay_manifest_and_real_entrypoint_are_present(self):
        manifest = json.loads((RELEASE / "overlay_manifest.json").read_text(encoding="utf-8"))
        for relative, record in manifest["files"].items():
            path = RELEASE / "overlay" / relative
            self.assertTrue(path.is_file(), relative)
            import hashlib
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["private_sha256"], relative)
        text = (RELEASE / "overlay" / "legged_gym" / "scripts" / "eval_batch.py").read_text(encoding="utf-8")
        self.assertIn("Batched offline evaluation", text)
        self.assertIn("import isaacgym", text)

    def test_documentation_identifies_overlay_and_stub(self):
        manifest = json.loads((RELEASE / "source_manifest.json").read_text(encoding="utf-8"))
        evaluator = next(item for item in manifest["selected_sources"]
                         if item["source"] == "legged_gym/scripts/eval_batch.py")
        self.assertEqual(evaluator["destination"], "overlay/legged_gym/scripts/eval_batch.py")
        self.assertTrue((RELEASE / evaluator["destination"]).is_file())
        readme = (RELEASE / "README.md").read_text(encoding="utf-8")
        self.assertIn(evaluator["destination"], readme)
        self.assertIn("offline stub", readme)
        self.assertIn("No Isaac Gym execution", readme)
        self.assertNotIn("not included", readme)
        overlay_readme = RELEASE / "overlay" / "README.md"
        self.assertIn("../../../docs/isaacgym_reproduction.md", overlay_readme.read_text(encoding="utf-8"))
        self.assertTrue((overlay_readme.parent / "../../../docs/isaacgym_reproduction.md").is_file())


if __name__ == "__main__":
    unittest.main()
