"""CSV helpers required by the archived MuJoCo runner (not an Isaac Gym runner)."""
import csv
from pathlib import Path

FIELDS = ["launch_id", "policy", "terrain", "chunk", "n_envs", "status", "seconds"]


def _write_launches_csv(exp_dir, rows):
    path = Path(exp_dir) / "launches.csv"
    old = {}
    if path.exists():
        with path.open(encoding="utf-8", newline="") as stream:
            old = {row["launch_id"]: row for row in csv.DictReader(stream)}
    for row in rows:
        if row["status"] != "skipped" or row["launch_id"] not in old:
            old[row["launch_id"]] = row
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, restval="")
        writer.writeheader()
        writer.writerows(old[key] for key in sorted(old))


def merge_experiment(exp_dir):
    paths = sorted((Path(exp_dir) / "episodes").glob("*.csv"))
    if not paths:
        return None
    columns, tables = [], []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            for column in reader.fieldnames or []:
                if column not in columns:
                    columns.append(column)
            tables.append(rows)
    with (Path(exp_dir) / "episodes_all.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, restval="")
        writer.writeheader()
        for rows in tables:
            writer.writerows(rows)
    return sum(map(len, tables))
