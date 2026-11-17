"""Checks for the published qualitative gait illustration."""
import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'historical' / 'gait_illustration'


def test_english_illustration_matches_provenance():
    provenance = json.loads((ROOT / 'provenance.json').read_text(encoding='utf-8'))
    image = ROOT / 'simulation_gait_sequence.png'
    assert hashlib.sha256(image.read_bytes()).hexdigest() == provenance['english_montage_sha256']
    header = image.read_bytes()[:24]
    assert header[:8] == b'\x89PNG\r\n\x1a\n' and header[12:16] == b'IHDR'
    assert list(struct.unpack('>II', header[16:24])) == provenance['source_dimensions']
