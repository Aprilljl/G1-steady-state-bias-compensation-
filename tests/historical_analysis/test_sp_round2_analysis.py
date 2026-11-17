"""Adapted from private source analysis tests; synthetic inputs only."""
import math
import numpy as np
from historical.thesis_eval.analysis import sp_round2_analysis as A


def test_weibull_shape_recovers_k_and_accepts_exponential():
    rng = np.random.default_rng(0)
    for k_true in (0.6, 1.0, 1.8):
        x = 30.0 * rng.weibull(k_true, 4000)
        t = 2.0 + np.minimum(x, 100.0)
        died = x < 100.0
        k, p = A.weibull_shape(t, died)
        assert abs(k - k_true) < 0.06 * k_true + 0.02
        if k_true == 1.0:
            assert p > 0.01
        else:
            assert p < 1e-6


def test_weibull_needs_events():
    k, p = A.weibull_shape([3.0, 4.0, 120.0], [True, False, False])
    assert math.isnan(k) and math.isnan(p)


def test_kaplan_meier_known_values():
    t, s = A.kaplan_meier([1, 2, 2, 3, 4, 5], [True, True, False, True, False, True])
    assert t == [1.0, 2.0, 3.0, 5.0]
    assert np.allclose(s, [5 / 6, 5 / 6 * 4 / 5, 5 / 6 * 4 / 5 * 2 / 3, 0.0])


def test_expected_failures_constant_hazard():
    lam = A.log_interp_hazard([1.0, 2.0], [0.01, 0.01], 1e-6)
    assert abs(A.expected_failures(lam, [1.5], 58.0, 2.0, 60.0) - 0.58) < 1e-12
    lam = A.log_interp_hazard([1.0, 3.0], [0.001, 0.1], 1e-6)
    assert abs(float(lam(2.0)) - 0.01) < 1e-12


def test_poisson_interval():
    assert A.poisson_interval(4.0) == (1, 8)
    assert A.poisson_interval(0.2) == (0, 1)
    lo, hi = A.poisson_interval(2000.0)
    assert abs(lo - 1912) <= 3 and abs(hi - 2088) <= 3
    assert A.poisson_interval(0.0) == (0, 0)
