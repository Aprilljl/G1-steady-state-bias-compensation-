import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from check_model import verify


class ModelManifestTests(unittest.TestCase):
    def test_identity_records(self):
        records = json.loads((ROOT / "model_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual({k: v["run"] for k, v in records.items()}, {
            "F": "Mar11_13-47-46_", "G": "Mar11_17-23-39_", "R": "Mar11_13-45-41_"})
        self.assertEqual(len({v["sha256"] for v in records.values()}), 3)
        for record in records.values():
            self.assertEqual(record["filename"], "model_10000.pt")
            self.assertEqual(len(bytes.fromhex(record["sha256"])), 32)

    def test_bundled_checkpoint_hashes(self):
        records = json.loads((ROOT / "model_manifest.json").read_text(encoding="utf-8"))
        for policy, record in records.items():
            with self.subTest(policy=policy):
                path = ROOT / "checkpoints" / policy / record["filename"]
                self.assertTrue(verify(path, record["sha256"])[1])

    def test_hash_and_missing_without_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "model_10000.pt"
            candidate.write_bytes(b"not a pickle or model")
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            self.assertEqual(verify(candidate, digest), (digest, True))
            self.assertEqual(verify(candidate, "0" * 64), (digest, False))
            command = [sys.executable, "-B", str(ROOT / "check_model.py"), "F"]
            mismatch = subprocess.run(command + [str(candidate)], capture_output=True, text=True)
            self.assertEqual(mismatch.returncode, 1)
            self.assertIn("MISMATCH", mismatch.stdout)
            missing = subprocess.run(command + [str(candidate) + ".missing"], capture_output=True, text=True)
            self.assertEqual(missing.returncode, 2)
            self.assertIn("Cannot read checkpoint", missing.stdout)
            listing = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(listing.returncode, 0)
            self.assertIn("unverified", listing.stdout)


if __name__ == "__main__":
    unittest.main()
