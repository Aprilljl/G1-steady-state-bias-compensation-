"""Read-only expansion of the historical Isaac Gym plan; never starts a simulator."""
import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HIST = ROOT / "historical" / "thesis_eval"
CONFIG = Path(__file__).parent / "configs" / "short_paper_isaacgym.yaml"


def expand(config_path, registry_path, out_root, only=None):
    spec = importlib.util.spec_from_file_location("historical_scenarios", HIST / "scenarios.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with open(registry_path, encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    registry = {}
    for row in rows:
        key = row.get("matched_run")
        if not key or key in registry or not row.get("final_checkpoint") or not row.get("final_sha256"):
            raise ValueError("registry must have unique matched_run, final_checkpoint and final_sha256")
        registry[key] = row
    return module.expand(module.load_config(config_path), registry, str(out_root), only=only)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--registry", required=True)
    parser.add_argument("--out-root", required=True, help="hypothetical output root; never created")
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args(argv)
    try:
        specs = expand(args.config, args.registry, args.out_root, args.only)
    except (ImportError, OSError, ValueError, KeyError) as exc:
        print("offline plan error: %s" % exc, file=sys.stderr)
        return 2
    print(json.dumps(specs, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
