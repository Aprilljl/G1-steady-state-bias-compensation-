"""Manifest-backed, byte-level checks for the published G1 scene closure."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import pytest

MANIFEST = Path(__file__).parents[1] / "historical" / "assets_manifest.json"


def configured_private_root() -> Path | None:
    """Return the private comparison tree only when explicitly opted in."""
    value = os.environ.get("CEP_PRIVATE_ASSETS_ROOT")
    return Path(value).expanduser().resolve() if value else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_manifest_is_self_contained():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    files = data["comparison"]["transitive_files_sha256"]
    assert files
    assert data["comparison"]["entrypoint"] in files
    assert all(len(value) == 64 for value in files.values())
    serialized = MANIFEST.read_text(encoding="utf-8")
    assert "D:\\" not in serialized


def test_private_comparison_matches_manifest_when_explicitly_configured():
    private_root = configured_private_root()
    if private_root is None:
        pytest.skip("private comparison opt-in unavailable; set CEP_PRIVATE_ASSETS_ROOT")
    if not private_root.is_dir():
        pytest.skip("configured private comparison source unavailable")
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for relative, expected in data["comparison"]["transitive_files_sha256"].items():
        path = private_root / relative
        assert path.is_file(), relative
        assert sha256(path) == expected, relative


def test_generator_synthetic_tree_and_exclusive_output(tmp_path):
    import subprocess
    import sys
    root = tmp_path / "unitree_rl_gym"
    base = root / "resources" / "robots" / "g1_description"
    meshes = base / "meshes"
    meshes.mkdir(parents=True)
    (base / "scene.xml").write_text('<mujoco><include file="g1_12dof.xml"/></mujoco>', encoding="utf-8")
    (base / "g1_12dof.xml").write_text('<mujoco><compiler meshdir="meshes/"/><asset><mesh file="one.STL"/></asset></mujoco>', encoding="utf-8")
    (meshes / "one.STL").write_bytes(b"synthetic")
    output = tmp_path / "manifest.json"
    script = Path(__file__).parents[1] / "historical" / "assets_manifest_generator.py"
    command = [sys.executable, str(script), "--source-root", str(root), "--output", str(output)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    data = json.loads(output.read_text(encoding="utf-8"))
    assert set(data["transitive_files_sha256"]) == {
        "resources/robots/g1_description/scene.xml",
        "resources/robots/g1_description/g1_12dof.xml",
        "resources/robots/g1_description/meshes/one.STL",
    }
    rejected = subprocess.run(command, capture_output=True, text=True)
    assert rejected.returncode != 0
