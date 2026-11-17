"""Freeze the outer-loop design (gamma, u_max) from stage-S1/S2 data before stage S3 runs (小论文补充实验设计.md section 4).

    python thesis_eval/analysis/sp_design.py \
        --hazard eval_results/short_paper_mujoco/SP-A1_speed_hazard/episodes_all.csv \
        --static eval_results/short_paper_mujoco/SP-B1_static_map/episodes_all.csv \
        --step_dir eval_results/short_paper_control/SP-B2_step_id \
        --out eval_results/short_paper_design/sp_design.json

Rules (fixed before the data exist; theory_framework.md sections 2.4, 3.4-3.6):
  u_max   u_max_hazard = largest SP-A1 command c (payload 0) such that every grid command <= c has no early failure and
          an exact-Poisson 95 % upper hazard bound <= lambda_tol = ln(1 / 0.8) / (20 - 2) s. Per payload m (eq. B-7):
          u_max(m) = largest SP-B1 command u <= u_max_hazard such that every SP-B1 cell of payload m with
          u_lo <= u' <= u survived all episodes (u_lo = 0.5, the lowest reference compensated in SP-B3); None if the cell
          at u_lo already fails (no admissible command range, the controller then cannot raise the command).
          v2 (--cap_t0 2, fixed on 2026-11-16 after S3): failures at or before 2 s are ignored. The outer loop is
          off until t_on = 2 s, so the cap cannot influence start-up failures; v1 (default, cap_t0 = 0) counted them
          and gave 15 kg a cap of 0.75 m/s from two SP-B1 falls at 1.78 s.
  kappa   secant slopes of the survivor-mean static map g(u; m) between adjacent SP-B1 commands with both ends
          <= u_max(m); kappa_max over all payloads is the design slope (stability margin shrinks with the slope).
  T, L    first-order-plus-dead-time fit of every SP-B2 per-step forward-speed trace (grid search over T and L < 0.8 s,
          linear least squares for offset and gain). T and L trade off against each other under gait ripple, so the
          design keeps every fitted (T, L) pair as one model instead of combining the largest T with the largest L.
  gamma   gamma * kappa_max = argmin over gk of the worst-case (over all fitted models) spectral radius of the lifted
          cycle-to-cycle closed loop (policy-rate plant, delay L, cycle-mean measurement, integral update every 40
          steps), so gamma = (gamma kappa)* / kappa_max; gk_upper = smallest gk at which some model becomes unstable.
          --max_T_quantile q < 1 drops the models whose T exceeds the q-quantile (declare before use; default all).
          A WARNING is printed when the predicted 2 % settling time exceeds `--settle_budget_s` (18 s = the time between
          t_on = 2 s and the settled window start 20 s of SP-B3); then lengthen the episodes or revise the design.
No scipy: Poisson limits by bisection, FOPDT by grid search.
"""
import argparse
import csv
import glob
import hashlib
import json
import math
import os
import time

import numpy as np

T0 = 2.0
P_STAR, T_EP = 0.8, 20.0
DT, CYCLE = 0.02, 40


# ------------------------------------------------------------------------------------------------ statistics
def _poisson_cdf(k, mu):
    """P(X <= k) for X ~ Poisson(mu), log-space sum."""
    if mu <= 0:
        return 1.0
    terms = [-mu + i * math.log(mu) - math.lgamma(i + 1) for i in range(int(k) + 1)]
    m = max(terms)
    return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms))


def _bisect(f, lo, hi, it=200):
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if f(mid):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def poisson_rate_interval(d, e, level=0.95):
    """Exact (Garwood) interval for a rate with d events in exposure e."""
    a = (1.0 - level) / 2.0
    hi_mu = _bisect(lambda mu: _poisson_cdf(d, mu) <= a, 0.0, 10.0 * d + 50.0)
    lo_mu = 0.0 if d == 0 else _bisect(lambda mu: 1.0 - _poisson_cdf(d - 1, mu) >= a, 0.0, 10.0 * d + 50.0)
    return lo_mu / e, hi_mu / e


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def truthy(s):
    return str(s).strip().lower() in ("true", "1")


def hazard_table(rows, t0=T0):
    out = {}
    for r in rows:
        out.setdefault(float(r["command_vx"]), []).append((float(r["survival_time_s"]), truthy(r["survived"])))
    table = []
    for c in sorted(out):
        t = np.array([x[0] for x in out[c]])
        dead = np.array([not x[1] for x in out[c]])
        m = t > t0
        d, e = int(dead[m].sum()), float((t[m] - t0).sum())
        lo, hi = poisson_rate_interval(d, e) if e > 0 else (float("nan"), float("inf"))
        table.append(dict(command_vx=c, n=len(t), early=int((~m).sum()), deaths=d, exposure_s=e,
                          hazard=d / e if e > 0 else float("nan"), lo=lo, hi=hi))
    return table


def u_max_from_hazard(table, lam_tol):
    best = None
    for row in table:  # ascending command
        if row["early"] > 0 or not row["hi"] <= lam_tol:
            break
        best = row["command_vx"]
    return best


# ------------------------------------------------------------------------------------------------ static map
def static_map(rows, cap_t0=0.0):
    """Survivor-mean static map per (payload, command). n_cap / n_survived_cap count only episodes that lived beyond
    cap_t0 s (the hazard model's start-up offset); with cap_t0 = 0 they equal n / n_survived."""
    cells = {}
    for r in rows:
        cells.setdefault((float(r["payload_kg"]), float(r["command_vx"])), []).append(r)
    out = []
    for (m, u) in sorted(cells):
        rs = cells[(m, u)]
        surv = [float(r["mean_vx"]) for r in rs if truthy(r["survived"])]
        late = [r for r in rs if "survival_time_s" not in r or float(r["survival_time_s"]) > cap_t0]
        out.append(dict(payload_kg=m, u=u, n=len(rs), n_survived=len(surv),
                        n_cap=len(late), n_survived_cap=sum(truthy(r["survived"]) for r in late),
                        g=float(np.mean(surv)) if surv else float("nan"),
                        g_sd=float(np.std(surv)) if len(surv) > 1 else float("nan")))
    return out


def payload_caps(smap, u_max_h, u_lo):
    """payload -> largest command u <= u_max_h with every cell u_lo <= u' <= u surviving (None if u_lo fails)."""
    caps = {}
    for m in sorted(set(c["payload_kg"] for c in smap)):
        cells = sorted((c for c in smap if c["payload_kg"] == m and u_lo - 1e-9 <= c["u"] <= u_max_h + 1e-9),
                       key=lambda c: c["u"])
        cap = None
        for c in cells:
            if c.get("n_survived_cap", c["n_survived"]) < c.get("n_cap", c["n"]):
                break
            cap = c["u"]
        caps[m] = cap
    return caps


def secant_slopes(smap, u_hi):
    by_m = {}
    for c in smap:
        by_m.setdefault(c["payload_kg"], []).append(c)
    slopes = []
    for m, cs in sorted(by_m.items()):
        cs = sorted(cs, key=lambda c: c["u"])
        for a, b in zip(cs[:-1], cs[1:]):
            if b["u"] <= u_hi + 1e-9 and np.isfinite(a["g"]) and np.isfinite(b["g"]):
                slopes.append(dict(payload_kg=m, u0=a["u"], u1=b["u"], kappa=(b["g"] - a["g"]) / (b["u"] - a["u"])))
    return slopes


# ------------------------------------------------------------------------------------------------ identification
def fopdt_fit(t, v, t_step, t_before=3.0, t_after=10.0):
    """Fit v(t) = y0 + dy * h(t - t_step - L; T), h(s) = 0 for s < 0 and 1 - exp(-s / T) otherwise.
    Returns dict(y0, dy, T, L, rmse); periodic gait ripple is left in the residual."""
    sel = (t >= t_step - t_before) & (t <= t_step + t_after)
    t, v = t[sel], v[sel]
    best = None

    def solve(T, L):
        s = t - t_step - L
        h = np.where(s > 0, 1.0 - np.exp(-np.clip(s, 0, None) / T), 0.0)
        A = np.stack([np.ones_like(h), h], axis=1)
        coef, res, _, _ = np.linalg.lstsq(A, v, rcond=None)
        r = float(np.sqrt(np.mean((A.dot(coef) - v) ** 2)))
        return r, coef

    for T in np.exp(np.linspace(np.log(0.03), np.log(4.0), 120)):
        for L in np.arange(0.0, 0.781, 0.02):
            r, coef = solve(T, L)
            if best is None or r < best[0]:
                best = (r, coef, T, L)
    r, coef, T0_, L0 = best
    for T in T0_ * np.exp(np.linspace(-0.1, 0.1, 21)):  # local refinement
        for L in np.clip(L0 + np.arange(-0.02, 0.0201, 0.002), 0.0, None):
            r2, c2 = solve(T, L)
            if r2 < best[0]:
                best = (r2, c2, T, L)
    r, coef, T, L = best
    return dict(y0=float(coef[0]), dy=float(coef[1]), T=float(T), L=float(L), rmse=r)


# ------------------------------------------------------------------------------------------------ loop design
def cycle_matrix(gk, T, L, dt=DT, n=CYCLE):
    """Cycle-to-cycle matrix of the deviation dynamics: per-step plant x+ = a x + (1 - a) u_delayed (unit gain),
    delay d = round(L / dt) < n steps, y = mean of x after each of the n steps, u+ = u - gk * y at the boundary.
    With d < n every delayed input at a boundary equals the previous cycle's u, so the state is (x, u_prev, u)."""
    a = math.exp(-dt / T)
    d = int(round(L / dt))
    if d >= n:
        raise ValueError("dead time %.3f s is not shorter than one gait cycle" % L)

    def one_cycle(s):
        x, u_prev, u = s
        acc = 0.0
        for k in range(n):
            x = a * x + (1.0 - a) * (u_prev if k < d else u)
            acc += x
        return np.array([x, u, u - gk * acc / n])

    return np.stack([one_cycle(e) for e in np.eye(3)], axis=1)


def spectral_radius(gk, T, L):
    return float(np.max(np.abs(np.linalg.eigvals(cycle_matrix(gk, T, L)))))


def design_gain(T, L, gk_max=12.0, n=1200):
    return design_gain_robust([(T, L)], gk_max, n)


def design_gain_robust(models, gk_max=12.0, n=1200):
    """Min-max gain over identified (T, L) models: rho_max(gk) = max_i rho(gk; T_i, L_i); gk* = argmin rho_max,
    gk_upper = smallest grid gk with rho_max >= 1."""
    gks = np.linspace(1e-3, gk_max, n)
    rho = np.array([max(spectral_radius(g, T, L) for T, L in models) for g in gks])
    unstable = np.nonzero(rho >= 1.0)[0]
    upper = float(gks[unstable[0]]) if len(unstable) else float("inf")
    i = int(np.argmin(rho))
    return dict(gk_star=float(gks[i]), rho_star=float(rho[i]), gk_upper=upper)


def settle_time(rho, frac=0.02, period=DT * CYCLE):
    return float(math.log(frac) / math.log(rho) * period) if 0 < rho < 1 else float("inf")


# ------------------------------------------------------------------------------------------------ main
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hazard", required=True)
    ap.add_argument("--static", required=True)
    ap.add_argument("--step_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lam_tol", type=float, default=math.log(1.0 / P_STAR) / (T_EP - T0))
    ap.add_argument("--u_lo", type=float, default=0.5)
    ap.add_argument("--max_T_quantile", type=float, default=1.0)
    ap.add_argument("--settle_budget_s", type=float, default=18.0)
    ap.add_argument("--cap_t0", type=float, default=0.0,
                    help="v2 rule: ignore SP-B1 failures at or before this time (s) when deriving u_max(m); the hazard "
                         "model excludes the same start-up segment (T0 = 2 s). Default 0 = frozen v1 rule.")
    args = ap.parse_args(argv)

    haz = hazard_table(read_csv(args.hazard))
    u_max_h = u_max_from_hazard(haz, args.lam_tol)
    if u_max_h is None:
        raise SystemExit("no SP-A1 command satisfies the hazard bound; inspect the data before designing")
    smap = static_map(read_csv(args.static), args.cap_t0)
    caps = payload_caps(smap, u_max_h, args.u_lo)
    if caps.get(0.0) is None:
        raise SystemExit("SP-B1 payload 0 fails at u_lo = %g; inspect the data" % args.u_lo)
    slopes = [s_ for m, c in caps.items() if c is not None
              for s_ in secant_slopes([x for x in smap if x["payload_kg"] == m], c)]
    if not slopes or min(s_["kappa"] for s_ in slopes) <= 0:
        raise SystemExit("static map not increasing below u_max (min slope %s); the design assumption fails" %
                         (min(s_["kappa"] for s_ in slopes) if slopes else None))
    k_max, k_min = max(s_["kappa"] for s_ in slopes), min(s_["kappa"] for s_ in slopes)
    fails = sorted(set((c["payload_kg"], c["u"]) for c in smap if c["n_survived"] < c["n"]))

    fits = []
    for p in sorted(glob.glob(os.path.join(args.step_dir, "timeseries", "*.npz"))):
        z = np.load(p, allow_pickle=False)
        cond = json.loads(str(z["condition"]))
        t_step, u0, u1 = float(cond["command_step"][0]), float(cond["command_vx"]), float(cond["command_step"][1])
        if len(z["t"]) == 0 or z["t"][-1] < t_step + 10.0:
            continue  # fell before the fit window ended
        f = fopdt_fit(z["t"], z["vx"], t_step)
        f.update(file=os.path.basename(p), payload_kg=float(cond["payload_kg"]), u0=u0, u1=u1,
                 kappa_step=f["dy"] / (u1 - u0))
        fits.append(f)
    if not fits:
        raise SystemExit("no usable SP-B2 traces in %s" % args.step_dir)
    T_cut = float(np.quantile([f["T"] for f in fits], args.max_T_quantile))
    models = sorted(set((round(f["T"], 3), round(f["L"], 2)) for f in fits if f["T"] <= T_cut + 1e-12))
    g = design_gain_robust(models, n=600)
    gamma = g["gk_star"] / k_max
    rho_kmin = max(spectral_radius(gamma * k_min, T, L) for T, L in models)
    T_d, L_d = max(T for T, _ in models), max(L for _, L in models)
    design = dict(
        gamma=gamma, u_min=0.0, u_max=caps[0.0], u_max_by_payload=dict(("%g" % m, c) for m, c in caps.items()),
        u_lo=args.u_lo, cap_t0_s=args.cap_t0, kappa_max=k_max, kappa_min=k_min, T_s=T_d, L_s=L_d, models_T_L=models,
        gk_star=g["gk_star"], gk_upper=g["gk_upper"], gain_margin=g["gk_upper"] / g["gk_star"],
        rho_at_kappa_max=g["rho_star"], rho_at_kappa_min=rho_kmin,
        settle_2pct_s_at_kappa_max=settle_time(g["rho_star"]), settle_2pct_s_at_kappa_min=settle_time(rho_kmin),
        lambda_tol=args.lam_tol, u_max_from_hazard=u_max_h, sp_b1_failure_cells=fails,
        max_T_quantile=args.max_T_quantile, T_cut_s=T_cut, settle_budget_s=args.settle_budget_s,
        T_median_s=float(np.median([f["T"] for f in fits])), L_median_s=float(np.median([f["L"] for f in fits])),
        kappa_step_median=float(np.median([f["kappa_step"] for f in fits])),
        hazard_table=haz, static_map=smap, slopes=slopes, step_fits=fits,
        inputs=dict(hazard=os.path.abspath(args.hazard), hazard_sha256=sha(args.hazard),
                    static=os.path.abspath(args.static), static_sha256=sha(args.static),
                    step_dir=os.path.abspath(args.step_dir), n_step_traces=len(fits)),
        rules=__doc__.split("Rules")[1].strip(), created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(design, f, indent=2)
    print("u_max_hazard %.3f (lambda_tol %.4f 1/s), u_max by payload %s, kappa [%.3f, %.3f], %d (T, L) models, T <= %.3f s, L <= %.3f s" % (
        u_max_h, args.lam_tol, design["u_max_by_payload"], k_min, k_max, len(models), T_d, L_d))
    print("gamma %.4f  (gamma kappa)* %.3f, stability bound %.3f, margin %.1f, rho %.3f / %.3f, settle %.1f / %.1f s" % (
        gamma, g["gk_star"], g["gk_upper"], design["gain_margin"], g["rho_star"], rho_kmin,
        design["settle_2pct_s_at_kappa_max"], design["settle_2pct_s_at_kappa_min"]))
    if design["settle_2pct_s_at_kappa_min"] > args.settle_budget_s:
        print("WARNING: predicted settling %.1f s exceeds the %.1f s budget; review before stage S3" % (
            design["settle_2pct_s_at_kappa_min"], args.settle_budget_s))
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
