"""Inspect archived model identities and hash candidate bytes without loading them."""
import argparse
import hashlib
import json
from pathlib import Path

MANIFEST = Path(__file__).resolve().with_name("model_manifest.json")


def verify(path, expected):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest(), digest.hexdigest() == expected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", choices=("F", "G", "R"))
    parser.add_argument("path", nargs="?", help="candidate checkpoint; omitted to show recorded identity")
    args = parser.parse_args(argv)
    record = json.loads(MANIFEST.read_text(encoding="utf-8"))[args.policy]
    print(json.dumps(record, indent=2))
    if args.path is None:
        print("No checkpoint supplied; identity and architecture remain unverified.")
        return 0
    try:
        actual, matches = verify(args.path, record["sha256"])
    except OSError as exc:
        print("Cannot read checkpoint: %s" % exc)
        return 2
    print("Candidate SHA-256: " + actual)
    print("Hash " + ("MATCH" if matches else "MISMATCH"))
    print("Hash matching does not establish safe deserialization or provenance; do not load unknown weights.")
    return 0 if matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
