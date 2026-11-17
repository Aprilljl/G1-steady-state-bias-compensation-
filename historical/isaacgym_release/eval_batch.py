"""Paper-specific Isaac Gym execution contract (offline import-safe stub).

This module is an offline boundary: it validates a launch specification without
importing Isaac Gym or loading a checkpoint. The separate overlay contains the
historical evaluator and environment code; its runtime closure and compatibility
remain unverified. See source_manifest.json.
"""
import json
from pathlib import Path

REQUIRED_SPEC_KEYS = ("launch_id", "policy_id", "checkpoint", "checkpoint_sha256",
                      "terrain", "episode_length_s", "envs")


def validate_spec(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("launch spec not found: %s" % path)
    spec = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_SPEC_KEYS if key not in spec]
    if missing:
        raise ValueError("launch spec missing keys: %s" % missing)
    if not spec["checkpoint_sha256"]:
        raise ValueError("checkpoint_sha256 is required; no guessed checkpoint is accepted")
    if spec["terrain"].get("mesh_type") != "plane":
        raise ValueError("this selected release only permits the registered plane Isaac Gym plan")
    return spec


def main(*_args, **_kwargs):
    raise RuntimeError(
        "Isaac Gym runtime is not bundled: bind this validated spec to the "
        "authorized private source eval_batch.py and exact environment patch "
        "on a licensed Isaac Gym host"
    )


if __name__ == "__main__":
    main()
