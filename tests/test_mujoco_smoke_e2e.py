"""Permanent, tiny local MuJoCo smoke coverage using an explicit stand-in only."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "historical" / "thesis_eval"
RUNNER = HIST / "mujoco_g1_12dof" / "ch5_mujoco.py"
CONFIG = HIST / "mujoco_g1_12dof" / "configs" / "short_paper_mujoco.yaml"
UPSTREAM = Path(os.environ.get("UNITREE_RL_GYM_ROOT", str(ROOT.parent / "unitree_rl_gym"))).expanduser()
XML = UPSTREAM / "resources" / "robots" / "g1_description" / "scene.xml"
MOTION = UPSTREAM / "deploy" / "pre_train" / "g1" / "motion.pt"
RUN_SMOKE = os.environ.get("RUN_MUJOCO_SMOKE") == "1"


@unittest.skipUnless(RUN_SMOKE, "set RUN_MUJOCO_SMOKE=1 to run the opt-in MuJoCo stand-in smoke")
@unittest.skipUnless(MOTION.is_file(), "upstream G1 TorchScript stand-in is not present")
class MujocoSmokeE2ETest(unittest.TestCase):
    def test_two_tiny_episodes_write_explicit_standin_metadata(self):
        with tempfile.TemporaryDirectory(prefix="cep-mujoco-smoke-") as tmp:
            out = Path(tmp) / "results"
            command = [
                sys.executable, "-B", str(RUNNER), "--config", str(CONFIG),
                "--local_jit", str(MOTION), "--xml", str(XML), "--out_root", str(out),
                "--only", "SP0_smoke_mujoco", "--workers", "1", "--episodes", "1",
            ]
            ascii_motion = Path(tmp) / "motion.pt"
            shutil.copyfile(MOTION, ascii_motion)
            command[command.index(str(MOTION))] = str(ascii_motion)
            process = subprocess.run(command, cwd=str(HIST), capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            experiment = out / "_jit_standin" / "SP0_smoke_mujoco"
            episodes = experiment / "episodes_all.csv"
            self.assertTrue(episodes.is_file())
            self.assertEqual(len(episodes.read_text(encoding="utf-8").splitlines()), 3)
            metadata = next((experiment / "meta").glob("*.json"))
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            self.assertTrue(payload["jit_standin"])
            self.assertTrue(payload["policy_id_in_csv"].endswith("@jit_standin"))
            self.assertEqual(payload["model_xml"], str(XML.resolve()))
            self.assertEqual(payload["model_xml_sha256"], hashlib.sha256(XML.read_bytes()).hexdigest())
            import mujoco
            self.assertEqual(payload["mujoco_version"], mujoco.__version__)

    def test_missing_xml_exits_nonzero_without_output(self):
        with tempfile.TemporaryDirectory(prefix="cep-mujoco-smoke-fail-") as tmp:
            out = Path(tmp) / "results"
            missing = Path(tmp) / "missing.xml"
            ascii_motion = Path(tmp) / "motion.pt"
            shutil.copyfile(MOTION, ascii_motion)
            process = subprocess.run([
                sys.executable, "-B", str(RUNNER), "--config", str(CONFIG),
                "--local_jit", str(ascii_motion), "--xml", str(missing), "--out_root", str(out),
                "--only", "SP0_smoke_mujoco", "--workers", "1", "--episodes", "1",
            ], cwd=str(HIST), capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
