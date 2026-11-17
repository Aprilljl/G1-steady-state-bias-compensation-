"""Adapted from private source analysis tests; synthetic cycle logs only."""
import math
import numpy as np
from historical.thesis_eval.analysis import sp_e1_rule as R


def cycles(rho, e0=0.1, sat=0.0, n_pre=30, n_post=25, period=0.8):
    rows = []
    onset = next(i for i in range(n_pre + n_post) if (i + 1) * period - period >= R.T_STEP - 1e-9)
    for k in range(n_pre + n_post):
        t = (k + 1) * period
        post = k >= onset
        j = k - onset if post else 0
        e = e0 * rho ** j * (-1) ** j if post else 0.0
        c = 1.1 if post else 1.0
        rows.append([t, c - e, c, c, 0.0, float(sat > 0 and post and j % 2 == 0 and sat >= 0.5), 1.0])
    return np.array(rows)


def test_decay_ratio_recovered_including_fast_decay():
    for rho in (0.6, 0.8, 0.95):
        r = R.classify(cycles(rho, e0=0.5))
        assert not r["unstable"]
        assert abs(r["rho"] - rho) < 0.01


def test_fast_convergence_is_stable_without_estimate():
    r = R.classify(cycles(0.1, e0=0.05))
    assert not r["unstable"] and math.isnan(r["rho"])


def test_growth_and_saturation_flagged():
    assert R.classify(cycles(1.08))["unstable"]
    c = cycles(0.7)
    c[:, 5] = 1.0
    assert R.classify(c)["reason"] == "saturation share > 0.5"
    assert R.classify(cycles(0.7), survived=False)["unstable"]
    assert R.classify(cycles(0.7)[:35])["unstable"]


def test_window_starts_after_step():
    w = R.window(cycles(0.8))
    assert len(w) == R.K and w[0, 0] - 0.8 >= R.T_STEP - 1e-9


def test_onset():
    assert R.onset([(1, False), (8, False), (12, True), (14, True)]) == 12
    assert R.onset([(1, False), (8, True), (12, False), (14, True)]) == 14
    assert R.onset([(1, False), (8, False)]) is None
