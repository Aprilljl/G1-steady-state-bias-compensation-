"""Map list.txt run names (e.g. Mar09_16-01-25_) to semantically named log folders.

The run name is the local wall-clock time at which the log dir was created; the
tfevents file name carries the unix epoch written a few seconds later.
"""
import argparse
import csv
import datetime
import json
import os
import re
import sys

try:
    from thesis_eval.policy_arch import infer_arch, load_checkpoint, file_sha256
except ImportError:  # executed as a script: python thesis_eval/registry.py
    from policy_arch import infer_arch, load_checkpoint, file_sha256

RUN_RE = re.compile(r"[A-Z][a-z]{2}\d{2}_\d{2}-\d{2}-\d{2}_")
EVENT_RE = re.compile(r"^events\.out\.tfevents\.(\d+)\.")
CKPT_RE = re.compile(r"^model_(\d+)\.pt$")
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

COLUMNS = [
    "folder", "folder_path", "event_epoch", "event_local_time", "matched_run", "match_delta_s",
    "list_sections", "list_descriptions", "label_txt", "n_checkpoints", "min_iter", "max_iter",
    "final_checkpoint", "final_sha256", "policy_class_name", "actor_hidden_dims", "rnn_type",
    "rnn_hidden_size", "arch_error",
]


def parse_list_txt(path):
    """Return a list of (section, description, run_name) in file order."""
    entries = []
    section = ""
    expect_header = False
    with open(path, encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if set(line) == {"#"}:
                expect_header = True
                continue
            if expect_header:
                section = line
                expect_header = False
            for run in RUN_RE.findall(line):
                desc = " ".join(RUN_RE.sub(" ", line).split())
                entries.append((section, desc, run))
    return entries


def run_to_datetime(run, year):
    mon, day = MONTHS.index(run[:3]) + 1, int(run[3:5])
    hh, mm, ss = int(run[6:8]), int(run[9:11]), int(run[12:14])
    return datetime.datetime(year, mon, day, hh, mm, ss)


def event_local_time(epoch, tz_offset_hours):
    return datetime.datetime.utcfromtimestamp(epoch) + datetime.timedelta(hours=tz_offset_hours)


def match_run(local_time, runs, tolerance_s):
    """Best run with -2 <= (event - run) <= tolerance; returns (run, delta) or (None, None)."""
    best = (None, None)
    for run in runs:
        # the run name has no year, so try neighbouring years to survive new-year wraparound
        for year in (local_time.year - 1, local_time.year, local_time.year + 1):
            try:
                delta = (local_time - run_to_datetime(run, year)).total_seconds()
            except ValueError:  # e.g. Feb 29 in a non-leap year
                continue
            if -2 <= delta <= tolerance_s and (best[0] is None or abs(delta) < abs(best[1])):
                best = (run, delta)
    return best


def find_run_folders(logs_root):
    """Folders with a tfevents file, searching logs_root/* and logs_root/*/*."""
    found = []
    for name in sorted(os.listdir(logs_root)):
        sub = os.path.join(logs_root, name)
        if not os.path.isdir(sub):
            continue
        if any(EVENT_RE.match(n) for n in os.listdir(sub)):
            found.append(sub)
            continue
        for name2 in sorted(os.listdir(sub)):
            sub2 = os.path.join(sub, name2)
            if os.path.isdir(sub2) and any(EVENT_RE.match(n) for n in os.listdir(sub2)):
                found.append(sub2)
    return found


def describe_folder(folder, logs_root, args):
    names = os.listdir(folder)
    epochs = [int(EVENT_RE.match(n).group(1)) for n in names if EVENT_RE.match(n)]
    epoch = min(epochs)
    ckpts = sorted((int(CKPT_RE.match(n).group(1)), n) for n in names if CKPT_RE.match(n))
    row = dict.fromkeys(COLUMNS, "")
    row.update({
        "folder": os.path.relpath(folder, logs_root).replace(os.sep, "/"),
        "folder_path": os.path.abspath(folder),
        "event_epoch": epoch,
        "event_local_time": event_local_time(epoch, args.tz_offset_hours).strftime("%Y-%m-%d %H:%M:%S"),
        "label_txt": " | ".join(sorted(n for n in names if n.lower().endswith(".txt"))),
        "n_checkpoints": len(ckpts),
    })
    if ckpts:
        final = os.path.abspath(os.path.join(folder, ckpts[-1][1]))
        row.update({"min_iter": ckpts[0][0], "max_iter": ckpts[-1][0], "final_checkpoint": final})
        if args.hash:
            row["final_sha256"] = file_sha256(final)
        try:
            _, state_dict = load_checkpoint(final, map_location="cpu")
            arch = infer_arch(state_dict)
            row.update({
                "policy_class_name": arch["policy_class_name"],
                "actor_hidden_dims": json.dumps(arch["actor_hidden_dims"]),
                "rnn_type": arch["rnn_type"] or "",
                "rnn_hidden_size": "" if arch["rnn_hidden_size"] is None else arch["rnn_hidden_size"],
            })
        except Exception as e:  # keep the registry usable even if one checkpoint is broken
            row["arch_error"] = "%s: %s" % (type(e).__name__, e)
    else:
        row["arch_error"] = "no model_*.pt found"
    return row


def build_registry(logs_root, list_txt, args):
    entries = parse_list_txt(list_txt)
    run_names = []
    for _, _, run in entries:
        if run not in run_names:
            run_names.append(run)

    rows = []
    for folder in find_run_folders(logs_root):
        row = describe_folder(folder, logs_root, args)
        local = datetime.datetime.strptime(row["event_local_time"], "%Y-%m-%d %H:%M:%S")
        run, delta = match_run(local, run_names, args.tolerance_s)
        if run is not None:
            secs, descs = [], []
            for s, d, r in entries:
                if r == run:
                    if s not in secs:
                        secs.append(s)
                    descs.append(d)
            row.update({"matched_run": run, "match_delta_s": "%g" % delta,
                        "list_sections": " | ".join(secs), "list_descriptions": " | ".join(descs)})
        rows.append(row)
    return rows, run_names


def write_csv(rows, out_path):
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows, run_names):
    by_run = {}
    for r in rows:
        if r["matched_run"]:
            by_run.setdefault(r["matched_run"], []).append(r["folder"])
    print("folders found: %d, matched: %d, list runs: %d"
          % (len(rows), sum(1 for r in rows if r["matched_run"]), len(run_names)))
    print("list.txt runs with NO matching folder:")
    missing = [r for r in run_names if r not in by_run]
    for r in missing:
        print("  " + r)
    if not missing:
        print("  (none)")
    print("folders NOT matched to any list run:")
    unmatched = [r["folder"] for r in rows if not r["matched_run"]]
    for f in unmatched:
        print("  " + f)
    if not unmatched:
        print("  (none)")
    print("runs matched by more than one folder:")
    dup = [(k, v) for k, v in sorted(by_run.items()) if len(v) > 1]
    for k, v in dup:
        print("  %s -> %s" % (k, ", ".join(v)))
    if not dup:
        print("  (none)")
    bad = [r["folder"] for r in rows if r["arch_error"]]
    if bad:
        print("folders with arch_error: " + ", ".join(bad))


def load_registry(csv_path):
    """run_name -> row dict. First matching folder wins; a row with final_sha256 is preferred."""
    registry = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            run = row["matched_run"]
            if not run:
                continue
            if run not in registry or (row["final_sha256"] and not registry[run]["final_sha256"]):
                registry[run] = row
    return registry


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logs_root", required=True)
    p.add_argument("--list_txt", required=True)
    p.add_argument("--out", default="eval_results/registry.csv")
    p.add_argument("--tz_offset_hours", type=float, default=8)
    p.add_argument("--tolerance_s", type=float, default=10)
    p.add_argument("--hash", action="store_true", help="compute sha256 of final checkpoints")
    args = p.parse_args()

    rows, run_names = build_registry(os.path.expanduser(args.logs_root), args.list_txt, args)
    write_csv(rows, args.out)
    print("wrote %s (%d rows)" % (args.out, len(rows)))
    print_summary(rows, run_names)


if __name__ == "__main__":
    sys.exit(main())
