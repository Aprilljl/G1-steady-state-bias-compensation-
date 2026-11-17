"""Drive batched policy evaluation: expand scenarios -> run eval_batch.py per launch -> merge CSVs.

    python thesis_eval/run_eval.py --config thesis_eval/configs/chapter5.yaml \
        --registry eval_results/registry.csv --out_root eval_results/chapter5 \
        [--only E5-1 E5-2] [--dry_run] [--force] [--python python] [--task g1]
    python thesis_eval/run_eval.py --out_root eval_results/chapter5 --summary   # merge only
"""
import argparse
import csv
import glob
import os
import re
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

LAUNCH_FIELDS = ["launch_id", "policy", "terrain", "chunk", "n_envs", "status", "seconds"]


def merge_csvs(csv_paths, out_path):
    """Concatenate CSVs; header is the union of columns in first-appearance order."""
    columns, tables = [], []
    for p in csv_paths:
        with open(p, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            for c in (reader.fieldnames or []):
                if c not in columns:
                    columns.append(c)
        tables.append(rows)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="")
        writer.writeheader()
        n = 0
        for rows in tables:
            writer.writerows(rows)
            n += len(rows)
    return n


def merge_experiment(exp_dir):
    paths = sorted(glob.glob(os.path.join(exp_dir, "episodes", "*.csv")))
    if not paths:
        return None
    out = os.path.join(exp_dir, "episodes_all.csv")
    return merge_csvs(paths, out)


def _run_launch(spec, spec_path, python, task, log_path):
    """Run eval_batch.py for one spec; tee combined output to log_path. Returns exit code."""
    env = dict(os.environ)
    env["EVAL_SPEC"] = spec_path
    cmd = [python, os.path.join("legged_gym", "scripts", "eval_batch.py"), "--task", task, "--headless"]
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        try:
            proc = subprocess.Popen(cmd, env=env, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, universal_newlines=True,
                                    encoding="utf-8", errors="replace")
        except OSError as e:
            log.write("failed to start: %s\n" % e)
            return -1
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


def _write_launches_csv(exp_dir, rows):
    """Update launches.csv; 'skipped' rows never overwrite an earlier record."""
    path = os.path.join(exp_dir, "launches.csv")
    old = {}
    if os.path.exists(path):
        with open(path, "r", newline="", encoding="utf-8") as f:
            old = {r["launch_id"]: r for r in csv.DictReader(f)}
    for r in rows:
        if r["status"] == "skipped" and r["launch_id"] in old:
            continue
        old[r["launch_id"]] = r
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LAUNCH_FIELDS, restval="")
        w.writeheader()
        w.writerows(old[k] for k in sorted(old))


def _list_experiments(out_root, only):
    if not os.path.isdir(out_root):
        return []
    names = sorted(d for d in os.listdir(out_root)
                   if os.path.isdir(os.path.join(out_root, d, "episodes")))
    if only:
        names = [n for n in names if any(n == o or n.startswith(o) for o in only)]
    return names


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--registry")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--python", default="python")
    ap.add_argument("--task", default="g1")
    ap.add_argument("--summary", action="store_true", help="only merge episodes/*.csv per experiment")
    args = ap.parse_args(argv)

    out_root = os.path.abspath(args.out_root)

    if args.summary:
        for name in _list_experiments(out_root, args.only):
            n = merge_experiment(os.path.join(out_root, name))
            print("%s: merged %s rows" % (name, n))
        return 0

    if not args.config or not args.registry:
        ap.error("--config and --registry are required unless --summary")

    from thesis_eval import scenarios
    from thesis_eval.registry import load_registry

    config = scenarios.load_config(args.config)
    registry = load_registry(args.registry)
    specs = scenarios.expand(config, registry, out_root, only=args.only)
    spec_paths = scenarios.write_specs(specs, out_root)
    print("expanded %d launches, %d envs in total" % (len(specs), sum(len(s["envs"]) for s in specs)))

    if args.dry_run:
        for s, p in zip(specs, spec_paths):
            print("  %s  envs=%d  -> %s" % (s["launch_id"], len(s["envs"]), p))
        return 0

    rows_by_exp, results = {}, []
    for spec, spec_path in zip(specs, spec_paths):
        exp_dir = os.path.join(out_root, spec["experiment"])
        chunk = int(re.search(r"__c(\d+)$", spec["launch_id"]).group(1))
        row = {"launch_id": spec["launch_id"], "policy": spec["policy_id"],
               "terrain": spec["terrain"]["name"], "chunk": chunk,
               "n_envs": len(spec["envs"]), "status": "", "seconds": 0.0}
        if os.path.exists(spec["out_csv"]) and not args.force:
            row["status"] = "skipped"
        else:
            print("=== running %s (%d envs)" % (spec["launch_id"], len(spec["envs"])))
            t0 = time.time()
            if args.force and os.path.exists(spec["out_csv"]):
                os.remove(spec["out_csv"])
            log_path = os.path.join(exp_dir, "logs", spec["launch_id"] + ".log")
            code = _run_launch(spec, spec_path, args.python, args.task, log_path)
            row["seconds"] = round(time.time() - t0, 1)
            row["status"] = "ok" if code == 0 and os.path.exists(spec["out_csv"]) else "failed"
        rows_by_exp.setdefault(spec["experiment"], []).append(row)
        results.append(row)

    for exp, rows in rows_by_exp.items():
        exp_dir = os.path.join(out_root, exp)
        _write_launches_csv(exp_dir, rows)
        n = merge_experiment(exp_dir)
        print("%s: episodes_all.csv rows=%s" % (exp, n))

    print("\n%-70s %-8s %8s" % ("launch_id", "status", "seconds"))
    for r in results:
        print("%-70s %-8s %8.1f" % (r["launch_id"], r["status"], r["seconds"]))
    n_failed = sum(r["status"] == "failed" for r in results)
    print("ok=%d skipped=%d failed=%d" % (sum(r["status"] == "ok" for r in results),
                                          sum(r["status"] == "skipped" for r in results), n_failed))
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
