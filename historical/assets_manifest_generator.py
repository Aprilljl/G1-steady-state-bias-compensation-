"""Inventory the G1 scene closure; never overwrite an existing file."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

BASE = Path("resources/robots/g1_description")
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "unitree_rl_gym"


def inventory(root):
    root = Path(root).resolve()
    base = (root / BASE).resolve()
    pending = [base / "scene.xml"]
    xmls, meshes = set(), set()
    while pending:
        path = pending.pop().resolve()
        path.relative_to(base)
        if path in xmls:
            continue
        xmls.add(path)
        tree = ET.parse(path)
        compiler = tree.find("compiler")
        meshdir = compiler.get("meshdir", "") if compiler is not None else ""
        for element in tree.iter("include"):
            pending.append(path.parent / element.attrib["file"])
        for element in tree.iter("mesh"):
            if "file" in element.attrib:
                mesh = (path.parent / meshdir / element.attrib["file"]).resolve()
                mesh.relative_to(base)
                meshes.add(mesh)
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(xmls | meshes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--comparison-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = inventory(args.source_root)
    if args.comparison_root is not None and records != inventory(args.comparison_root):
        raise ValueError("source and comparison closures differ")
    result = {"entrypoint": (BASE / "scene.xml").as_posix(),
              "transitive_files_sha256": records,
              "comparison_performed": args.comparison_root is not None}
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"inventoried {len(records)} files")


if __name__ == "__main__":
    main()
