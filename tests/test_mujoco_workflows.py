"""Fast permanent checks for the guarded historical MuJoCo workflow."""
import unittest
import tempfile
import subprocess
import shutil
import os
from unittest.mock import patch
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import g1_runner


class MujocoWorkflowTests(unittest.TestCase):
    def test_historical_experiment_mapping_is_explicit(self):
        expected = {
            ("F", "mujoco-smoke"): "short_paper_mujoco.yaml",
            ("F", "control-smoke"): "short_paper_control.yaml",
            ("G", "mujoco-smoke"): "short_paper_mujoco_G.yaml",
            ("G", "S2-step"): "short_paper_control_G.yaml",
        }
        for (policy, experiment), filename in expected.items():
            runner, config, _ = g1_runner.plan(policy, experiment)
            self.assertTrue(runner.is_file())
            self.assertEqual(config.name, filename)
            self.assertTrue(config.is_file())

    def test_unavailable_round_three_and_R_are_blocked(self):
        for policy in ("R",):
            with self.assertRaises(ValueError):
                g1_runner.plan(policy, "S1-hazard")
        with self.assertRaises(ValueError):
            g1_runner.plan("F", "S3-round3")

    def test_smoke_config_keeps_tiny_protocol(self):
        config = (g1_runner.CONFIGS / "short_paper_mujoco.yaml").read_text(encoding="utf-8")
        self.assertIn("SP0_smoke_mujoco:", config)
        self.assertIn("episodes: 2", config)
        self.assertIn("episode_length_s: 30.0", config)
        self.assertIn("command_vx: [1.0, 2.4]", config)

    @unittest.skipUnless(os.environ.get("RUN_MUJOCO_SMOKE") == "1", "set RUN_MUJOCO_SMOKE=1")
    def test_control_standin_and_existing_output_refusal(self):
        upstream = Path(os.environ.get("UNITREE_RL_GYM_ROOT", str(ROOT.parent / "unitree_rl_gym"))).expanduser()
        motion = upstream / "deploy" / "pre_train" / "g1" / "motion.pt"
        xml = upstream / "resources" / "robots" / "g1_description" / "scene.xml"
        if g1_runner.inspect_dependencies() or not motion.is_file() or not xml.is_file():
            self.skipTest("MuJoCo dependencies or upstream scene/stand-in unavailable")
        with tempfile.TemporaryDirectory(prefix="control-standin-") as tmp:
            model = Path(tmp) / "motion.pt"
            shutil.copyfile(motion, model)
            out = Path(tmp) / "output"
            cmd = [sys.executable, "-B", str(g1_runner.RUNNERS / "sp_control.py"),
                   "--config", str(g1_runner.CONFIGS / "short_paper_control.yaml"),
                   "--local_jit", str(model), "--out_root", str(out),
                   "--only", "SP0_smoke_control", "--episodes", "1", "--workers", "1"]
            first = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            result = out / "_jit_standin" / "SP0_smoke_control" / "episodes_all.csv"
            self.assertTrue(result.is_file())
            before = result.read_bytes()
            second = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("output directory exists", second.stderr)
            self.assertEqual(result.read_bytes(), before)

    def test_floquet_rejects_stateful_policy_for_each_job(self):
        import importlib
        floquet = importlib.import_module("historical.thesis_eval.mujoco_g1_12dof.sp_floquet")

        policy = object()
        jobs = [dict(payload=0.0, cmd=1.0, episode=i, policy=policy) for i in range(2)]
        with patch.object(floquet.C5, "Ch5Sim", return_value=object()), \
             patch.object(floquet, "load_policy", return_value=policy) as load, \
             patch.object(floquet, "is_stateless", return_value=False) as check, \
             patch.object(floquet, "analyse_condition") as analyse:
            floquet._W.clear()
            self.addCleanup(floquet._W.clear)
            for job in jobs:
                result = floquet.run_job(job)
                self.assertIn("error", result)
                self.assertIn("stateful policy", result["error"])
            self.assertNotIn("policy", floquet._W)
            self.assertEqual(load.call_count, 2)
            self.assertEqual(check.call_count, 2)
            analyse.assert_not_called()

    def test_historical_control_and_followup_design_mapping(self):
        control = (g1_runner.CONFIGS / "short_paper_control.yaml").read_text(encoding="utf-8")
        followup = (g1_runner.CONFIGS / "short_paper_followup.yaml").read_text(encoding="utf-8")
        for name in ("SP-B3_compensation:", "SP-B4_gain_sweep:", "SP-B5_payload_step:",
                     "SP-B6_estimator:", "SP-B7_cap:"):
            self.assertIn(name, control)
        for name in ("SP-E1_gain_onset:", "SP-E2a_b3_15kg:", "SP-E2b_b5_15kg:",
                     "SP-E2c_b7_15kg:", "SP-E2d_b7_none:"):
            self.assertIn(name, followup)
        self.assertIn('"15": 0.75', (g1_runner.ROOT / "historical" / "eval_results" /
                                     "short_paper_design" / "sp_design.json").read_text(encoding="utf-8"))
        self.assertIn('"15": 2.0', (g1_runner.ROOT / "historical" / "eval_results" /
                                    "short_paper_design" / "sp_design_v2.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
