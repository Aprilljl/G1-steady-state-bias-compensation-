"""Round-2 analysis for the short paper: censored Weibull shape tests, Kaplan-Meier curves, and the cross-layer check
that the SP-A1 hazard curve lambda(u) predicts the failures of the uncapped arm in SP-B7 / SP-E2.

    python -m thesis_eval.analysis.sp_round2_analysis --root eval_results --json <out.json>

Cross-layer prediction (theory_framework.md sections 2 and 3.6): an episode of the uncapped arm is driven with the
command u(t) logged in its cycle timeseries (or, without a trajectory, with its final command u_final held after
t_on). Under the constant-hazard model, the expected number of failures in an arm is
    sum_i  integral_{t0}^{T_i} lambda(u_i(t)) dt,
with lambda(u) linearly interpolated in log space between the SP-A1 grid points (payload 0 kg; the payload effect on
the hazard is the untested assumption of this check and is reported as such). Observed failures are compared with the
Poisson interval of the prediction.
"""
import argparse
import csv
import json
import math
import os

import numpy as np

T0 = 2.0


# ------------------------------------------------------------------------------------------------ survival models
def weibull_shape(times, died, t0=T0, min_events=5):
    """Censored Weibull MLE of the shape k for S(t) = exp(-((t - t0) / s)^k), left-truncated at t0, profile scale.
    Returns (k, p of the likelihood-ratio test of k = 1) or (nan, nan) with fewer than min_events events."""
    t = np.asarray(times, dtype=np.float64)
    d = np.asarray(died, dtype=bool)
    m = t > t0
    x, d = t[m] - t0, d[m]
    n_ev = int(d.sum())
    if n_ev < min_events:
        return float("nan"), float("nan")

    def nll(k):
        s = (np.sum(x ** k) / n_ev) ** (1.0 / k)
        return -(n_ev * math.log(k / s) + (k - 1) * np.sum(np.log(x[d] / s)) - np.sum((x / s) ** k))

    ks = np.exp(np.linspace(math.log(0.2), math.log(5.0), 400))
    vals = np.array([nll(k) for k in ks])
    i = int(np.argmin(vals))
    lo, hi = ks[max(i - 1, 0)], ks[min(i + 1, len(ks) - 1)]
    for _ in range(80):  # golden-section refinement inside the bracketing grid cell
        a, b = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        if nll(a) < nll(b):
            hi = b
        else:
            lo = a
    k = 0.5 * (lo + hi)
    lr = max(2.0 * (nll(1.0) - nll(k)), 0.0)
    return float(k), float(math.erfc(math.sqrt(lr / 2.0)))  # chi2(1) survival function


def kaplan_meier(times, died):
    """Product-limit estimate; returns (event times, S just after each)."""
    t = np.asarray(times, dtype=np.float64)
    d = np.asarray(died, dtype=bool)
    order = np.argsort(t, kind="stable")
    t, d = t[order], d[order]
    s, out_t, out_s = 1.0, [], []
    n = len(t)
    i = 0
    while i < n:
        j = i
        while j < n and t[j] == t[i]:
            j += 1
        deaths = int(d[i:j].sum())
        if deaths:
            s *= 1.0 - deaths / float(n - i)
            out_t.append(float(t[i]))
            out_s.append(s)
        i = j
    return out_t, out_s


# ------------------------------------------------------------------------------------------------ cross-layer link
def log_interp_hazard(grid_u, grid_lam, floor):
    """lambda(u): log-linear interpolation; zero-event grid points are replaced by `floor` (their upper bound scale),
    below the grid -> floor, above the grid -> last value."""
    u = np.asarray(grid_u, dtype=np.float64)
    lam = np.maximum(np.asarray(grid_lam, dtype=np.float64), floor)
    ll = np.log(lam)

    def f(x):
        x = np.asarray(x, dtype=np.float64)
        return np.exp(np.interp(x, u, ll, left=math.log(floor), right=ll[-1]))
    return f


def expected_failures(lam, u_t, dt, t_start, t_end, t0=T0):
    """integral of lam(u(t)) over [max(t0, t_start), t_end] for a piecewise-constant command sampled every dt."""
    u_t = np.asarray(u_t, dtype=np.float64)
    t = t_start + dt * np.arange(len(u_t))
    w = np.clip(np.minimum(t + dt, t_end) - np.maximum(t, t0), 0.0, None)
    return float(np.sum(lam(u_t) * w))


def poisson_interval(mu, level=0.95):
    """Central interval of a Poisson(mu) count (inclusive integer bounds). Log-space pmf so that large mu does not
    underflow exp(-mu) to 0 (which would never reach the upper quantile)."""
    a = (1.0 - level) / 2.0
    lo, hi, c, k = None, None, 0.0, 0
    while hi is None:
        c += math.exp(-mu + k * math.log(mu) - math.lgamma(k + 1)) if mu > 0 else (1.0 if k == 0 else 0.0)
        if lo is None and c > a:
            lo = k
        if c >= 1.0 - a:
            hi = k
        k += 1
    return lo, hi


# ------------------------------------------------------------------------------------------------ io
def read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def tf(s):
    return str(s).strip().lower() in ("true", "1")


def hazard_grid(rows, t0=T0):
    out = []
    for c in sorted({float(r["command_vx"]) for r in rows}):
        g = [r for r in rows if float(r["command_vx"]) == c]
        t = np.array([float(r["survival_time_s"]) for r in g])
        dead = np.array([not tf(r["survived"]) for r in g])
        m = t > t0
        e, d = float((t[m] - t0).sum()), int(dead[m].sum())
        k, p = weibull_shape(t, dead, t0)
        out.append(dict(command_vx=c, n=len(g), early=int((~m).sum()), deaths=d, exposure_s=e,
                        hazard=d / e if e > 0 else float("nan"), weibull_k=k, weibull_p=p))
    return out


def cross_layer(grid, arm_rows, floor):
    lam = log_interp_hazard([g["command_vx"] for g in grid], [g["hazard"] for g in grid], floor)
    out = []
    keys = sorted({(float(r["payload_kg"]), float(r["command_vx"])) for r in arm_rows})
    for m, c in keys:
        g = [r for r in arm_rows if float(r["payload_kg"]) == m and float(r["command_vx"]) == c]
        mu = 0.0
        for r in g:
            t_end = float(r["survival_time_s"])
            t_on = float(r["t_on_s"])
            u_end = float(r["u_final"])
            # without a trajectory: command = reference before t_on, u_final after (the loop settles within ~15 s)
            mu += expected_failures(lam, [c], t_on - T0, T0, min(t_on, t_end)) if t_end > T0 else 0.0
            mu += expected_failures(lam, [u_end], t_end - t_on, t_on, t_end) if t_end > t_on else 0.0
        obs = sum(not tf(r["survived"]) for r in g)
        lo, hi = poisson_interval(mu)
        out.append(dict(payload_kg=m, command_vx=c, n=len(g), observed=obs, predicted=mu, pred_lo=lo, pred_hi=hi,
                        u_final_mean=float(np.mean([float(r["u_final"]) for r in g])),
                        consistent=bool(lo <= obs <= hi)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="eval_results")
    ap.add_argument("--json", required=True)
    args = ap.parse_args(argv)
    j = lambda *p: os.path.join(args.root, *p)
    res = {}
    for name, path in [("F_mujoco_120s", j("short_paper_mujoco", "SP-A1_speed_hazard", "episodes_all.csv")),
                       ("F_isaac_60s", j("short_paper_isaacgym", "SP-A2_speed_hazard_plane", "episodes_all.csv")),
                       ("G_mujoco_120s", j("short_paper_G", "short_paper_mujoco", "SP-A1_speed_hazard", "episodes_all.csv"))]:
        if os.path.isfile(path):
            res[name] = hazard_grid(read(path))
    b7 = read(j("short_paper_control", "SP-B7_cap", "episodes_all.csv"))
    nocap = [r for r in b7 if r["controller"] == "outer_nocap"]
    floor = 1.0 / 11800.0  # one event in the SP-A1 exposure of a zero-failure grid point
    res["cross_layer_floor_per_s"] = floor
    res["cross_layer_nocap"] = cross_layer(res["F_mujoco_120s"], nocap, floor)
    res["note"] = ("lambda(u) from 0 kg SP-A1; payload effect on the hazard assumed nil (PA-4 tested only below 2 m/s); "
                   "u(t) approximated by the reference before t_on and u_final after")
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, allow_nan=True)
    for name in ("F_mujoco_120s", "F_isaac_60s", "G_mujoco_120s"):
        for g in res.get(name, []):
            if g["deaths"] >= 5:
                print("%-14s %.1f m/s  D %3d  k %.2f  p %.3f" % (name, g["command_vx"], g["deaths"], g["weibull_k"],
                                                               g["weibull_p"]))
    for r in res["cross_layer_nocap"]:
        print("nocap %2g kg %.2f m/s  u_final %.2f  observed %d  predicted %.2f [%d, %d]  %s" % (
            r["payload_kg"], r["command_vx"], r["u_final_mean"], r["observed"], r["predicted"], r["pred_lo"],
            r["pred_hi"], "consistent" if r["consistent"] else "OUTSIDE"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
