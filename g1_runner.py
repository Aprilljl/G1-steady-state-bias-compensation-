"""Read-only planning and explicitly authorized historical MuJoCo launches."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
HIST = ROOT / "historical"
sys.path.insert(0, str(HIST))
from thesis_eval.registry_builder import build_registry, write_registry

CONFIGS = ROOT / "historical" / "thesis_eval" / "mujoco_g1_12dof" / "configs"
RUNNERS = ROOT / "historical" / "thesis_eval" / "mujoco_g1_12dof"
# Exact experiment names: no prefix expansion, no arbitrary configs, no later-stage bypass.
EXPERIMENTS = {
    "mujoco-smoke": ("ch5_mujoco.py", "short_paper_mujoco", "SP0_smoke_mujoco"),
    "control-smoke": ("sp_control.py", "short_paper_control", "SP0_smoke_control"),
    "S1-hazard": ("ch5_mujoco.py", "short_paper_mujoco", "SP-A1_speed_hazard"),
    "S1-static": ("ch5_mujoco.py", "short_paper_mujoco", "SP-B1_static_map"),
    "S2-step": ("sp_control.py", "short_paper_control", "SP-B2_step_id"),
}

# Historical configs use the same experiment names but different policy registries.
# Keep the mapping explicit: no guessed filenames or accidental cross-policy runs.
POLICY_CONFIGS = {
    "F": {
        "mujoco-smoke": "short_paper_mujoco.yaml",
        "control-smoke": "short_paper_control.yaml",
        "S1-hazard": "short_paper_mujoco.yaml",
        "S1-static": "short_paper_mujoco.yaml",
        "S2-step": "short_paper_control.yaml",
    },
    "G": {
        "mujoco-smoke": "short_paper_mujoco_G.yaml",
        "S1-hazard": "short_paper_mujoco_G.yaml",
        "S1-static": "short_paper_mujoco_G.yaml",
        "S2-step": "short_paper_control_G.yaml",
    },
    "R": {},
}


def plan(policy, experiment):
    if policy not in POLICY_CONFIGS:
        raise ValueError("policy must be one of F, G, R")
    if experiment not in EXPERIMENTS:
        raise ValueError("unknown experiment: %s" % experiment)
    config_name = POLICY_CONFIGS[policy].get(experiment)
    if config_name is None:
        if policy == "R":
            raise ValueError("R is available for registry only: no matching archived short-paper MuJoCo config")
        raise ValueError("%s has no archived configuration for %s" % (policy, experiment))
    runner, _, name = EXPERIMENTS[experiment]
    return RUNNERS / runner, CONFIGS / config_name, name


def inspect_dependencies():
    return [name for name in ("numpy", "torch", "mujoco", "yaml") if importlib.util.find_spec(name) is None]


def output_path(path):
    output = Path(path).expanduser().resolve()
    # Historical source/data, project root, existing directories/files, and ancestors
    # are never valid output targets. Explicitly require a fresh directory.
    if output.exists() or output == ROOT or ROOT in output.parents or output in ROOT.parents:
        raise ValueError("output must be a new directory outside this project")
    if not output.parent.is_dir():
        raise ValueError("output parent directory must already exist")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("check", "registry", "run"):
        p = sub.add_parser(action)
        for policy in ("F", "G", "R"):
            p.add_argument("--" + policy.lower(), help="candidate checkpoint path")
        if action == "registry":
            p.add_argument("--out", required=True, help="new CSV path; never overwrites")
        if action in ("check", "run"):
            p.add_argument("--policy", choices=("F", "G", "R"), required=True)
            p.add_argument("--experiment", choices=tuple(EXPERIMENTS), default="mujoco-smoke")
        if action == "run":
            p.add_argument("--out", required=True, help="new directory outside project")
            p.add_argument("--trust-checkpoint", action="store_true", help="assert trusted provenance; torch.load can execute code")
            p.add_argument("--confirm-run", action="store_true", help="explicitly authorize simulator execution")
    args = parser.parse_args(argv)
    paths = {p: getattr(args, p.lower()) for p in ("F", "G", "R") if getattr(args, p.lower())}
    try:
        if args.action in ("check", "run"):
            runner, config, experiment = plan(args.policy, args.experiment)
            if set(paths) != {args.policy}:
                raise ValueError("provide exactly the selected policy checkpoint path")
        rows = build_registry(ROOT / "model_manifest.json", paths)
        print("Verified SHA-256 only; this does not establish trusted provenance or safe deserialization.")
        if args.action == "registry":
            path = Path(args.out).expanduser().resolve()
            # Exclusive create: also protects checkpoint and existing source files.
            with path.open("x", encoding="utf-8-sig", newline="") as stream:
                import csv
                from thesis_eval.registry_builder import COLUMNS
                writer = csv.DictWriter(stream, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            print("Registry: %s (contains absolute local paths; review before sharing)" % path)
            return 0
        missing = inspect_dependencies()
        gym_root = Path(os.environ.get("UNITREE_RL_GYM_ROOT", str(ROOT.parent / "unitree_rl_gym"))).expanduser().resolve()
        xml = gym_root / "resources" / "robots" / "g1_description" / "scene.xml"
        print("Plan: %s %s %s; config=%s" % (args.policy, args.experiment, experiment, config))
        print("Missing Python modules: %s" % (", ".join(missing) if missing else "none"))
        print("Upstream scene: %s (%s)" % (xml, "present" if xml.is_file() else "missing"))
        print("Later-stage design gates are not implemented by this launcher; S3 and later are blocked.")
        if args.action == "check":
            return 0 if not missing and xml.is_file() else 2
        if not args.trust_checkpoint or not args.confirm_run:
            raise ValueError("run requires both --trust-checkpoint and --confirm-run")
        if missing or not xml.is_file():
            raise ValueError("missing dependencies or upstream scene; see checks above")
        output = output_path(args.out)
        # mkdir is exclusive; no pre-existing output can be overwritten or merged.
        output.mkdir()
        runner_output = output / "results"
        with tempfile.TemporaryDirectory(prefix="g1-registry-") as temp:
            registry = write_registry(rows, Path(temp) / "registry.csv")
            command = [sys.executable, "-B", str(runner), "--config", str(config), "--registry", str(registry),
                       "--out_root", str(runner_output), "--only", experiment, "--workers", "1"]
            print("Launching: " + subprocess.list2cmdline(command), flush=True)
            return subprocess.call(command, cwd=str(HIST))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
