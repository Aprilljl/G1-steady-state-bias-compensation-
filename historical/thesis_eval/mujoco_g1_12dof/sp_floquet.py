"""SP-A3: orbital stability of the frozen 12-DoF policy in MuJoCo from the gait-cycle stroboscopic return map
(小论文补充实验设计.md; theory_framework.md section 2.6, prediction PA-3).

    python thesis_eval/mujoco_g1_12dof/sp_floquet.py --registry eval_results/registry.csv \
        --out_root eval_results/short_paper_floquet --workers 16
    python thesis_eval/mujoco_g1_12dof/sp_floquet.py --local_jit PATH/policy.pt --speeds 1.0 --payloads 0 --episodes 1  # smoke

Per (payload, command, episode): reset as ch5_mujoco (initial_state(seed, 0, episode)), walk `settle_cycles` gait cycles,
sample the reduced state z at every phase-clock boundary (env.n_steps % 40 == 0, phase 0). The orbit period p (in phase
cycles, 1 ... max_period) is the smallest lag whose scaled change max_k |z_k - z_{k-p}| / scale over the last 6 samples is
below `tol` (a gait may repeat every 2 phase cycles: the TorchScript stand-in does); converged = such a p exists.
z* = last sample (yaw, x, y removed by symmetry). Then
  - P(z) = reduced state after p * 40 policy steps started from the snapshot with z overwritten,
  - J = central differences of P at z* with step eps_rel * scale_j (eigenvalues are invariant to the diagonal scaling),
  - rho_J = max |eig(J)| of the p-cycle map for each eps_rel, rho_per_cycle = rho_J ** (1 / p),
  - rho_decay = per-p-cycle decay factor of a perturbation along the dominant eigenvector (independent check).
J and rho are computed only for orbits classified as periodic. For every condition (periodic or not):
  - ftle = finite-time maximal Lyapunov exponent per phase cycle (Benettin renormalisation of `ftle_dirs` random
    perturbations of scaled size 1e-3 over `ftle_cycles` cycles along the nominal trajectory); on a periodic orbit
    ftle ~ ln(rho_per_cycle), ftle > 0 indicates irregular (non-periodic) motion.
Reduced state (45): base height, roll, pitch, heading-frame base linear velocity (3), body angular velocity (3), joint
positions (12), joint velocities (12), previous action (12). Only stateless policies are supported: the script refuses a
policy whose action on a repeated identical observation changes (recurrent nets, e.g. the LSTM TorchScript stand-in
deploy/pre_train/g1/motion.pt), because a hidden state outside z invalidates snapshot/restore and the return map.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for _p in (str(HERE), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mujoco  # noqa: E402
import ch5_mujoco as C5  # noqa: E402
from policies12 import load_policy  # noqa: E402

CYCLE_STEPS = 40
NZ = 45
SCALE = np.concatenate([[0.01], [0.01, 0.01], [0.05] * 3, [0.1] * 3, [0.01] * 12, [0.1] * 12, [0.04] * 12])
COLUMNS = ["policy_id", "checkpoint_sha256", "payload_kg", "command_vx", "episode", "eps_rel", "converged",
           "fell_during_settle", "settle_cycles", "period_cycles", "change_lag1", "change_lag2", "fixed_point_residual",
           "rho_J", "rho_per_cycle", "mu_abs_1", "mu_abs_2", "mu_abs_3", "mu_abs_4", "mu_abs_5", "mu_abs_6", "n_fd_falls",
           "rho_decay", "ftle", "ftle_sd", "ftle_falls", "mean_vx_cycle"]


def rz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_from_euler_xyz(roll, pitch, yaw):
    """Inverse of ch5_mujoco.euler_xyz (R = Rz(yaw) Ry(pitch) Rx(roll)), MuJoCo (w, x, y, z)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def read_z(env):
    """Reduced state and the removed coordinates (yaw, x, y). Free-joint qvel[0:3] is world-frame, qvel[3:6] body-frame."""
    d = env.data
    roll, pitch, yaw = C5.euler_xyz(d.qpos[3:7])
    z = np.concatenate([[d.qpos[2], roll, pitch], rz(yaw).T.dot(d.qvel[0:3]), d.qvel[3:6], d.qpos[env.qadr],
                        d.qvel[env.dadr], env.last_action])
    return z, (yaw, d.qpos[0], d.qpos[1])


def write_z(env, z, keep):
    yaw, x, y = keep
    d = env.data
    d.qpos[0:3] = [x, y, z[0]]
    d.qpos[3:7] = quat_from_euler_xyz(z[1], z[2], yaw)
    d.qvel[0:3] = rz(yaw).dot(z[3:6])
    d.qvel[3:6] = z[6:9]
    d.qpos[env.qadr] = z[9:21]
    d.qvel[env.dadr] = z[21:33]
    env.last_action = np.array(z[33:45], dtype=np.float64)
    mujoco.mj_forward(env.model, d)


def snapshot(env):
    return dict(data=env.data.__copy__(), last_action=env.last_action.copy(), n_steps=env.n_steps, cmd=env.cmd.copy())


def restore(env, s):
    env.data = s["data"].__copy__()
    env.last_action, env.n_steps, env.cmd = s["last_action"].copy(), s["n_steps"], s["cmd"].copy()


def fallen(sim):
    st = sim.read_state()
    return st["pelvis_force"] > C5.TERM_FORCE_N or abs(st["rpy"][1]) > C5.TERM_PITCH or abs(st["rpy"][0]) > C5.TERM_ROLL


def roll_steps(sim, policy, n):
    """n policy steps from the current state (observation recomputed from the state). Returns (ok, mean vx)."""
    env = sim.env
    obs = np.clip(env.get_obs(), -C5.CLIP_OBS, C5.CLIP_OBS)
    vx = 0.0
    for _ in range(n):
        a = np.clip(policy.act(obs)[0], -C5.CLIP_ACTIONS, C5.CLIP_ACTIONS)
        obs = np.clip(env.step(a), -C5.CLIP_OBS, C5.CLIP_OBS)
        vx += float(sim.read_state()["lin_b"][0])
        if fallen(sim):
            return False, vx / n
    return True, vx / n


def is_stateless(policy, n_obs=47):
    o = np.random.RandomState(0).randn(n_obs).astype(np.float32)
    policy.reset()
    a = [np.asarray(policy.act(o)[0]) for _ in range(3)]
    policy.reset()
    return all(np.array_equal(a[0], x) for x in a[1:])


def jacobian(fmap, z0, eps):
    """Central differences of fmap at z0 with per-coordinate steps eps. Returns (J, number of failed evaluations)."""
    n = len(z0)
    J = np.zeros((n, n))
    n_fail = 0
    for j in range(n):
        e = np.zeros(n)
        e[j] = eps[j]
        fp, okp = fmap(z0 + e)
        fm, okm = fmap(z0 - e)
        n_fail += (not okp) + (not okm)
        J[:, j] = (fp - fm) / (2.0 * eps[j])
    return J, n_fail


def ftle(sim, policy, snap, n_cycles, n_dirs, rng, size=1e-3):
    """Mean log growth per phase cycle of a renormalised small perturbation along the nominal trajectory."""
    env = sim.env
    restore(env, snap)
    nominal = [snapshot(env)]
    for _ in range(n_cycles):
        if not roll_steps(sim, policy, CYCLE_STEPS)[0]:
            break
        nominal.append(snapshot(env))
    rates, falls = [], 0
    for _ in range(n_dirs):
        d = rng.randn(NZ)
        d *= size / np.linalg.norm(d)
        logs = []
        for k in range(len(nominal) - 1):
            restore(env, nominal[k])
            z, keep = read_z(env)
            write_z(env, z + d * SCALE, keep)
            ok, _ = roll_steps(sim, policy, CYCLE_STEPS)
            z_pert = read_z(env)[0]
            restore(env, nominal[k + 1])
            d1 = (z_pert - read_z(env)[0]) / SCALE
            if not ok:
                falls += 1
                break
            g = np.linalg.norm(d1) / size
            logs.append(math.log(max(g, 1e-300)))
            d = d1 * (size / max(np.linalg.norm(d1), 1e-300))
        if logs:
            rates.append(float(np.mean(logs)))
    restore(env, snap)
    return (float(np.mean(rates)) if rates else float("nan"), float(np.std(rates)) if len(rates) > 1 else float("nan"),
            falls)


def analyse_condition(sim, policy, payload, cmd, episode, seed, settle_cycles, tol, eps_list, decay_cycles,
                      max_period=2, ftle_cycles=20, ftle_dirs=3):
    env = sim.env
    sim.apply_condition(float(payload), 1.0)
    policy.reset()
    env.reset(cmd=np.array([cmd, 0.0, 0.0]), perturb=C5.initial_state(seed, 0, episode, env.default_pos))
    env.step(np.zeros(12))
    base = dict(payload_kg=payload, command_vx=cmd, episode=episode, settle_cycles=settle_cycles)
    ok, _ = roll_steps(sim, policy, CYCLE_STEPS - env.n_steps % CYCLE_STEPS)
    zs, vx_cycle = [], float("nan")
    for _ in range(settle_cycles):
        if not ok:
            break
        zs.append(read_z(env)[0])
        ok, vx_cycle = roll_steps(sim, policy, CYCLE_STEPS)
    if not ok:
        return [dict(base, converged=False, fell_during_settle=True)], None
    zs.append(read_z(env)[0])
    Z = np.array(zs) / SCALE
    change = {}
    for p in range(1, max_period + 1):
        change[p] = float(np.max(np.abs(Z[-6:] - Z[-6 - p:-p]))) if len(Z) >= 6 + p else float("inf")
    good = [p for p in sorted(change) if change[p] < tol]
    converged = bool(good)
    period = good[0] if good else min(change, key=change.get)
    n_steps = period * CYCLE_STEPS
    z_star, keep = read_z(env)
    snap = snapshot(env)

    def fmap(z):
        restore(env, snap)
        write_z(env, z, keep)
        good_, _ = roll_steps(sim, policy, n_steps)
        return read_z(env)[0], good_

    fl, fl_sd, fl_falls = ftle(sim, policy, snap, ftle_cycles, ftle_dirs, np.random.RandomState([seed, 2, episode]))
    common = dict(base, converged=converged, fell_during_settle=False, period_cycles=period, change_lag1=change.get(1),
                  change_lag2=change.get(2), mean_vx_cycle=vx_cycle, ftle=fl, ftle_sd=fl_sd, ftle_falls=fl_falls)
    mats = {}
    if not converged:
        mats.update(z_star=z_star, settle_samples=np.array(zs), period_cycles=period)
        return [dict(common, eps_rel=float("nan"))], mats
    p0, _ = fmap(z_star)
    resid = float(np.max(np.abs((p0 - z_star) / SCALE)))
    rows = []
    for er in eps_list:
        J, n_fail = jacobian(fmap, z_star, er * SCALE)
        mu = np.sort(np.abs(np.linalg.eigvals(J)))[::-1]
        mats["J_eps%g" % er] = J
        rows.append(dict(common, eps_rel=er, fixed_point_residual=resid, rho_J=float(mu[0]),
                         rho_per_cycle=float(mu[0]) ** (1.0 / period), n_fd_falls=n_fail,
                         **dict(("mu_abs_%d" % (i + 1), float(mu[i])) for i in range(6))))

    # independent check: decay of a perturbation along the dominant eigenvector (middle eps), per p-cycle step
    J = mats["J_eps%g" % eps_list[len(eps_list) // 2]]
    w, V = np.linalg.eig(np.diag(1.0 / SCALE).dot(J).dot(np.diag(SCALE)))
    v = np.real(V[:, np.argmax(np.abs(w))])
    v = 0.05 * v / np.max(np.abs(v)) * SCALE
    restore(env, snap)
    nom = []
    for _ in range(decay_cycles):
        roll_steps(sim, policy, n_steps)
        nom.append(read_z(env)[0])
    restore(env, snap)
    write_z(env, z_star + v, keep)
    dist = []
    for k in range(decay_cycles):
        good_, _ = roll_steps(sim, policy, n_steps)
        if not good_:
            break
        dist.append(float(np.linalg.norm((read_z(env)[0] - nom[k]) / SCALE)))
    dist = np.array(dist)
    sel = np.arange(len(dist))[(np.arange(len(dist)) >= 1) & (dist > 1e-8)][:10]
    rho_decay = float(np.exp(np.polyfit(sel, np.log(dist[sel]), 1)[0])) if len(sel) >= 3 else float("nan")
    for r in rows:
        r["rho_decay"] = rho_decay
    mats.update(z_star=z_star, settle_samples=np.array(zs), decay_dist=dist, period_cycles=period)
    return rows, mats


_W = {}


def run_job(job):
    try:
        if "sim" not in _W:
            _W["sim"] = C5.Ch5Sim(0.005, 4)
        if "policy" not in _W:
            policy = load_policy(job["policy"])
            if not is_stateless(policy):
                raise RuntimeError("stateful policy (repeated observation gives different actions): its hidden state "
                                   "is not part of the reduced state, the return map is undefined")
            _W["policy"] = policy
        rows, mats = analyse_condition(_W["sim"], _W["policy"], job["payload"], job["cmd"], job["episode"],
                                       job["seed"], job["settle_cycles"], job["tol"], job["eps"], job["decay_cycles"],
                                       job["max_period"], job["ftle_cycles"], job["ftle_dirs"])
        if mats is not None:
            os.makedirs(job["mat_dir"], exist_ok=True)
            np.savez(os.path.join(job["mat_dir"], "m%g_c%g_ep%d.npz" % (job["payload"], job["cmd"], job["episode"])),
                     **mats)
        return dict(rows=rows)
    except Exception:
        return dict(error="payload %s cmd %s ep %s\n%s" % (job["payload"], job["cmd"], job["episode"],
                                                          traceback.format_exc()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=None)
    ap.add_argument("--policy", default="Mar11_13-47-46_")
    ap.add_argument("--local_jit", default=None)
    ap.add_argument("--out_root", default=str(REPO_ROOT / "eval_results" / "short_paper_floquet"))
    ap.add_argument("--speeds", type=float, nargs="*", default=[0.5, 1.0, 1.5, 1.8, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5])
    ap.add_argument("--payloads", type=float, nargs="*", default=[0.0, 10.0])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--settle_cycles", type=int, default=40)
    ap.add_argument("--tol", type=float, default=0.2, help="scaled change threshold of the period test")
    ap.add_argument("--max_period", type=int, default=2)
    ap.add_argument("--ftle_cycles", type=int, default=20)
    ap.add_argument("--ftle_dirs", type=int, default=3)
    ap.add_argument("--eps_rel", type=float, nargs="*", default=[0.01, 0.1, 1.0])
    ap.add_argument("--decay_cycles", type=int, default=12)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    standin = args.local_jit is not None
    out_root = os.path.join(os.path.abspath(args.out_root), "_jit_standin") if standin else os.path.abspath(args.out_root)
    out_csv = os.path.join(out_root, "floquet.csv")
    if os.path.exists(out_csv) and not args.force:
        print("skip: %s exists (--force to re-run)" % out_csv)
        return 0
    if standin:
        ckpt, sha_reg = os.path.abspath(args.local_jit), ""
    else:
        if not args.registry:
            raise SystemExit("--registry is required (or --local_jit for a smoke test)")
        from thesis_eval.registry import load_registry
        reg = load_registry(args.registry)
        ckpt, sha_reg = reg[args.policy]["final_checkpoint"], reg[args.policy].get("final_sha256", "")
    if not os.path.isfile(ckpt):
        raise SystemExit("checkpoint not found: %s" % ckpt)
    sha = C5.file_sha256(ckpt)
    if sha_reg and sha_reg != sha:
        raise SystemExit("checkpoint sha256 mismatch for %s" % ckpt)
    pol = dict(kind="jit" if standin else "ckpt", path=ckpt, activation="elu")
    jobs = [dict(payload=m, cmd=c, episode=ep, seed=args.seed, settle_cycles=args.settle_cycles, tol=args.tol,
                 eps=list(args.eps_rel), decay_cycles=args.decay_cycles, max_period=args.max_period,
                 ftle_cycles=args.ftle_cycles, ftle_dirs=args.ftle_dirs, policy=pol,
                 mat_dir=os.path.join(out_root, "matrices"))
            for m in args.payloads for c in args.speeds for ep in range(args.episodes)]
    print("SP-A3: %d conditions x episodes%s" % (len(jobs), "  [JIT STAND-IN]" if standin else ""), flush=True)
    t0 = time.time()
    pool = Pool(args.workers, initializer=C5._worker_init) if args.workers > 1 else None
    results = pool.map(run_job, jobs, chunksize=1) if pool else [run_job(j) for j in jobs]
    if pool:
        pool.close()
        pool.join()
    rows, errors = [], []
    pid = args.policy + (C5.STANDIN_SUFFIX if standin else "")
    for r in results:
        if "error" in r:
            errors.append(r["error"])
            print("ERROR " + r["error"], flush=True)
        else:
            rows.extend(dict(r_, policy_id=pid, checkpoint_sha256=sha) for r_ in r["rows"])
    os.makedirs(out_root, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([C5.fmt(r.get(k)) for k in COLUMNS])
    meta = dict(args=vars(args), policy=pol, checkpoint_sha256=sha, jit_standin=standin, scale=SCALE.tolist(),
                reduced_state="height, roll, pitch, heading-frame lin vel (3), body ang vel (3), q (12), qd (12), "
                              "last action (12)", cycle_steps=CYCLE_STEPS, mujoco_version=mujoco.__version__,
                wall_s=time.time() - t0, n_errors=len(errors), end_time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(os.path.join(out_root, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    if errors:
        with open(os.path.join(out_root, "errors.log"), "w", encoding="utf-8") as f:
            f.write("\n\n".join(errors))
    print("done: %d rows, %d errors, %.0f s" % (len(rows), len(errors), time.time() - t0), flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
