"""Instability rule for SP-E1 (gain sweep with a mid-episode reference step), fixed on 2026-11-16 BEFORE the data exist.

    python thesis_eval/analysis/sp_e1_rule.py eval_results/short_paper_followup/SP-E1_gain_onset

Per trajectory (timeseries npz, `cycles` columns t, y, c, u, z, saturated, updated):
  e_k   = c_k - y_k for the K = 20 updated cycles after the reference step (step at t_step = 25 s; first cycle whose
          start time is >= t_step, i.e. the cycle mean no longer mixes the old reference)
  rho   = exp(slope) of a least-squares line through log|e_k| over the cycles before |e_k| first drops below E_FLOOR
          (at least 3 cycles; if the error reaches the floor within 3 cycles the episode is stable and rho is not
          estimated, reported as NaN)
  unstable  if  rho > 1  or  the share of saturated updates in the window > 0.5  or  the episode fell inside it
Per gain: unstable if at least half of the episodes are unstable. Onset = smallest swept gain multiple that is unstable
and is followed only by unstable multiples. Prediction PB-2 holds if the onset lies in [0.7, 1.4] x the model bound
(model bound = gk_upper of the identified operating-point model / (gamma * kappa_op)), and |rho_hat - rho_model| <= 0.1
at the stable multiples where rho is estimated. E_FLOOR is the cycle-mean ripple level; points below it carry no decay
information and would bias the slope towards 1.
"""
import argparse
import glob
import json
import math
import os
import re

import numpy as np

K = 20
E_FLOOR = 2e-3   # m/s, below the cycle-mean ripple seen in SP-B4 (late |c - y| 1e-4 .. 7e-3)
SAT_MAX = 0.5
T_STEP = 25.0


def window(cycles, t_step=T_STEP, k=K):
    """Rows of `cycles` (updated, finite) whose cycle starts at or after the step, first k of them."""
    c = np.asarray(cycles, dtype=np.float64).reshape(-1, 7)
    period = np.median(np.diff(c[:, 0])) if len(c) > 1 else 0.8
    sel = (c[:, 0] - period >= t_step - 1e-9) & (c[:, 6] > 0) & np.isfinite(c[:, 1])
    return c[sel][:k]


def classify(cycles, survived=True, t_step=T_STEP, k=K):
    w = window(cycles, t_step, k)
    if len(w) < k:
        return dict(unstable=True, reason="fell or fewer than %d post-step cycles" % k, rho=float("nan"),
                    sat_share=float("nan"), n=len(w))
    e = np.abs(w[:, 2] - w[:, 1])
    below = np.nonzero(e < E_FLOOR)[0]
    n_fit = int(below[0]) if len(below) else k
    rho = float(math.exp(np.polyfit(np.arange(n_fit), np.log(e[:n_fit]), 1)[0])) if n_fit >= 3 else float("nan")
    sat = float(w[:, 5].mean())
    unstable = (rho > 1.0) or (sat > SAT_MAX) or (not survived)  # NaN rho (fast convergence) compares False
    reason = "rho > 1" if rho > 1.0 else ("saturation share > 0.5" if sat > SAT_MAX else ("fell" if not survived
                                                                                         else "stable"))
    return dict(unstable=bool(unstable), reason=reason, rho=rho, sat_share=sat, n=k, n_fit=n_fit)


def onset(per_gain):
    """per_gain: sorted list of (multiple, unstable_bool). Smallest multiple followed only by unstable ones."""
    out = None
    for m, u in reversed(per_gain):
        if not u:
            break
        out = m
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_dir")
    ap.add_argument("--out", default=None, help="json summary path (default <exp_dir>/e1_rule.json)")
    args = ap.parse_args(argv)
    import csv
    with open(os.path.join(args.exp_dir, "episodes_all.csv"), encoding="utf-8-sig", newline="") as f:
        surv = dict(((r["condition_id"], r["episode"]), r["survived"] == "True") for r in csv.DictReader(f))
    rows = []
    for p in sorted(glob.glob(os.path.join(args.exp_dir, "timeseries", "*.npz"))):
        m = re.match(r"gamma_mult=([0-9.]+)__ep(\d+)\.npz", os.path.basename(p))
        if not m:
            continue
        z = np.load(p)
        cid = "gamma_mult=%s" % repr(float(m.group(1)))
        r = classify(z["cycles"], surv.get((cid, m.group(2)), True))
        r.update(multiple=float(m.group(1)), episode=int(m.group(2)))
        rows.append(r)
    mults = sorted(set(r["multiple"] for r in rows))
    per = []
    for mu in mults:
        g = [r for r in rows if r["multiple"] == mu]
        per.append(dict(multiple=mu, n=len(g), n_unstable=sum(r["unstable"] for r in g),
                        rho_median=float(np.nanmedian([r["rho"] for r in g])),
                        unstable=sum(r["unstable"] for r in g) * 2 >= len(g)))
    res = dict(rule=__doc__, per_gain=per, onset_multiple=onset([(p["multiple"], p["unstable"]) for p in per]),
               episodes=rows)
    out = args.out or os.path.join(args.exp_dir, "e1_rule.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    for p in per:
        print("gamma x %-5g unstable %d/%d  rho_median %.3f" % (p["multiple"], p["n_unstable"], p["n"], p["rho_median"]))
    print("onset multiple:", res["onset_multiple"], "-> wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
