import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "historical"))
import g1_runner
from thesis_eval.registry_builder import build_registry, write_registry
from thesis_eval.registry import load_registry
from thesis_eval import scenarios


class RegistryAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="g1 registry space ")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        original = json.loads((ROOT / "model_manifest.json").read_text(encoding="utf-8"))
        self.manifest = {}
        self.files = {}
        for policy, record in original.items():
            candidate = self.path / (policy + " synthetic bytes.txt")
            payload = ("synthetic fixture only: " + policy).encode()
            candidate.write_bytes(payload)
            self.files[policy] = candidate
            self.manifest[policy] = dict(record, sha256=hashlib.sha256(payload).hexdigest())
        self.manifest_path = self.path / "fixture.json"
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_each_policy_and_real_registry_schema(self):
        for policy in "FGR":
            rows = build_registry(self.manifest_path, {policy: self.files[policy]})
            output = self.path / (policy + ".csv")
            write_registry(rows, output)
            reg = load_registry(output)
            run = self.manifest[policy]["run"]
            self.assertEqual(set(reg), {run})
            self.assertEqual(reg[run]["final_checkpoint"], str(self.files[policy].resolve()))
            self.assertEqual(reg[run]["final_sha256"], self.manifest[policy]["sha256"])
            with self.assertRaises(FileExistsError):
                write_registry(rows, output)

    def test_scenario_expansion_and_bad_bytes(self):
        rows = build_registry(self.manifest_path, {"F": self.files["F"]})
        output = write_registry(rows, self.path / "registry.csv")
        config = scenarios.load_config(str(g1_runner.CONFIGS / "short_paper_mujoco.yaml"))
        specs = scenarios.expand(config, load_registry(output), str(self.path / "not-created"), only=["SP0_smoke_mujoco"])
        self.assertEqual(len(specs), 1)
        self.assertEqual(len(specs[0]["envs"]), 4)
        self.assertFalse((self.path / "not-created").exists())
        self.files["F"].write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            build_registry(self.manifest_path, {"F": self.files["F"]})
        with self.assertRaises(FileNotFoundError):
            build_registry(self.manifest_path, {"G": self.path / "missing"})

    def test_cli_refuses_synthetic_checkpoint_without_writing(self):
        out = self.path / "never.csv"
        process = subprocess.run([sys.executable, "-B", str(ROOT / "g1_runner.py"), "registry", "--f",
                                  str(self.files["F"]), "--out", str(out)], capture_output=True, text=True)
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(out.exists())

    def test_dangerous_paths_and_gates(self):
        for path in (ROOT, ROOT / "historical" / "eval_results", self.path):
            with self.assertRaises(ValueError):
                g1_runner.output_path(path)
        with self.assertRaises(ValueError):
            g1_runner.plan("R", "S1-hazard")
        with self.assertRaises(ValueError):
            g1_runner.plan("G", "control-smoke")
        runner, config, name = g1_runner.plan("G", "S2-step")
        self.assertEqual(runner.name, "sp_control.py")
        self.assertEqual(config.name, "short_paper_control_G.yaml")
        self.assertEqual(name, "SP-B2_step_id")
        with patch("g1_runner.build_registry", return_value=[{}]), patch("g1_runner.subprocess.call") as call:
            rc = g1_runner.main(["run", "--policy", "F", "--f", "fixture", "--out", str(self.path / "new")])
            self.assertNotEqual(rc, 0)
            call.assert_not_called()
            self.assertFalse((self.path / "new").exists())

    def test_registry_output_parent_and_existing_file_are_protected(self):
        with self.assertRaises(ValueError):
            g1_runner.output_path(self.path / "new" / "nested")
        existing = self.path / "existing.csv"
        existing.write_text("sentinel", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            write_registry([], existing)
        self.assertEqual(existing.read_text(encoding="utf-8"), "sentinel")

    def test_control_wrapper_dispatches_to_fresh_child_without_force(self):
        gym = self.path / "gym"
        scene = gym / "resources" / "robots" / "g1_description" / "scene.xml"
        scene.parent.mkdir(parents=True)
        scene.write_text("fixture", encoding="utf-8")
        output = self.path / "new-output"

        def dispatch(command, cwd):
            child = Path(command[command.index("--out_root") + 1])
            self.assertEqual(child, output / "results")
            self.assertTrue(output.is_dir())
            self.assertFalse(child.exists())
            self.assertNotIn("--force", command)
            self.assertEqual(command[command.index("--only") + 1], "SP0_smoke_control")
            return 0

        with patch.dict("os.environ", {"UNITREE_RL_GYM_ROOT": str(gym)}), \
             patch("g1_runner.build_registry", return_value=[]), \
             patch("g1_runner.inspect_dependencies", return_value=[]), \
             patch("g1_runner.subprocess.call", side_effect=dispatch):
            rc = g1_runner.main(["run", "--policy", "F", "--experiment", "control-smoke", "--f", "fixture",
                                 "--out", str(output), "--trust-checkpoint", "--confirm-run"])
        self.assertEqual(rc, 0)

    def test_merge_helpers(self):
        from thesis_eval.run_eval import _write_launches_csv, merge_experiment
        directory = self.path / "experiment"
        (directory / "episodes").mkdir(parents=True)
        for i in range(2):
            with (directory / "episodes" / (str(i) + ".csv")).open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["id", "value"])
                writer.writerow([i, i + 1])
        self.assertEqual(merge_experiment(directory), 2)
        row = dict(launch_id="id", policy="F", terrain="plane", chunk=0, n_envs=2, status="ok", seconds=1)
        _write_launches_csv(directory, [row])
        _write_launches_csv(directory, [dict(row, status="skipped")])
        with (directory / "launches.csv").open(newline="") as stream:
            self.assertEqual(next(csv.DictReader(stream))["status"], "ok")


if __name__ == "__main__":
    unittest.main()
