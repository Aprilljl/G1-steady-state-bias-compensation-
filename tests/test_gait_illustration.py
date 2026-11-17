"""Checks for the published qualitative gait illustration."""
import hashlib
import json
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / 'historical' / 'gait_illustration'


def test_english_illustration_matches_provenance():
    provenance = json.loads((ROOT / 'provenance.json').read_text(encoding='utf-8'))
    image = ROOT / 'simulation_gait_sequence.png'
    assert hashlib.sha256(image.read_bytes()).hexdigest() == provenance['english_montage_sha256']
    with Image.open(image) as figure:
        assert list(figure.size) == provenance['source_dimensions']
