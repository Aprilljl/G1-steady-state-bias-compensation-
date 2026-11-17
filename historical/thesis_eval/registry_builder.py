"""Strict checkpoint registry builder and verifier for the historical MuJoCo runner.

This module writes the runner's real CSV schema: ``matched_run`` is the policy
id consumed by ``thesis_eval.scenarios.expand`` and ``final_checkpoint`` plus
``final_sha256`` are the fields consumed before any torch deserialization.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

COLUMNS = [
    "folder", "folder_path", "event_epoch", "event_local_time", "matched_run", "match_delta_s",
    "list_sections", "list_descriptions", "label_txt", "n_checkpoints", "min_iter", "max_iter",
    "final_checkpoint", "final_sha256", "policy_class_name", "actor_hidden_dims", "rnn_type",
    "rnn_hidden_size", "arch_error",
]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path, expected):
    """Hash bytes only; never imports torch or deserializes a checkpoint."""
    actual = file_sha256(path)
    return actual, actual == expected


def _normalise_policy(policy):
    if policy not in {"F", "G", "R"}:
        raise ValueError("policy must be one of F, G, R")
    return policy


def _record(manifest, policy, checkpoint):
    policy = _normalise_policy(policy)
    record = manifest[policy]
    expected = record["sha256"]
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("checkpoint not found: %s" % path)
    actual, matches = verify_checkpoint(path, expected)
    if not matches:
        raise ValueError("SHA-256 mismatch for %s: expected %s, got %s" % (policy, expected, actual))
    # Registry paths are absolute so the historical runner cannot silently use a
    # different cwd. No model bytes are copied or loaded.
    return {
        "folder": str(path.parent),
        "folder_path": str(path.parent),
        "matched_run": record["run"],
        "final_checkpoint": str(path),
        "final_sha256": actual,
        "policy_class_name": "ActorCritic",
        "actor_hidden_dims": json.dumps([512, 256, 128]),
        "rnn_type": "",
        "rnn_hidden_size": "",
        "arch_error": "registered architecture; checkpoint not deserialized by builder",
        "label_txt": "",
        "n_checkpoints": "1",
        "min_iter": "10000",
        "max_iter": "10000",
        "event_epoch": "",
        "event_local_time": "",
        "match_delta_s": "",
        "list_sections": "",
        "list_descriptions": "",
    }


def build_registry(manifest_path, paths):
    """Return a CSV-row list for selected policy:path pairs.

    ``paths`` is a mapping with exactly the selected policy keys; missing keys
    are rejected rather than converted to guessed or stand-in files.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not paths:
        raise ValueError("select at least one policy checkpoint")
    unknown = set(paths) - {"F", "G", "R"}
    if unknown:
        raise ValueError("unknown policies: %s" % sorted(unknown))
    return [_record(manifest, policy, paths[policy]) for policy in ("F", "G", "R") if policy in paths]


def write_registry(rows, output):
    output = Path(output).expanduser().resolve()
    with output.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return output
