"""Short-paper control experiments in MuJoCo (SP-B2 ... SP-B7, 小论文补充实验设计.md): the frozen 12-DoF policy with a
gait-cycle-synchronous outer integral loop on the forward-velocity command, reference steps, payload steps and
velocity-estimate bias / noise. Theory: harness/short_paper/robustness_analysis/docs/theory_framework.md, section 3.

    python thesis_eval/mujoco_g1_12dof/sp_control.py --config thesis_eval/mujoco_g1_12dof/configs/short_paper_control.yaml \
        --registry eval_results/registry.csv --out_root eval_results/short_paper_control --workers 16 --only SP-B2
    # experiments that use `design` values additionally need --design eval_results/short_paper_design/sp_design.json
    # local smoke test with a TorchScript stand-in (never real results; outputs go to <out_root>/_jit_standin)
    python ... --local_jit PATH/policy.pt --only SP0

Episode protocol = ch5_mujoco.run_episode (same reset noise, zero-action first step, termination and EpisodeRecorder
feed); with controller `none` and no schedules the recorder output is identical (tests/test_sp_control.py). Differences:
  - initial state RandomState([seed, 0, episode]) for EVERY condition, so conditions are paired episode by episode;
  - the recorder always receives the reference c(t); only the policy observation carries u = c + z;
  - a second recorder with warmup = settle_s gives the settled-window metrics (columns settled_*);
  - schedules (reference step, payload step) act before policy step round(t / 0.02), like the push in ch5_mujoco.
Outer loop (theory eq. B-5): at every gait-cycle boundary (env.n_steps % 40 == 0, i.e. phase 0) with t >= t_on_s,
  y = mean measured forward speed over the last 40 policy steps (an incomplete first cycle is discarded),
  z = clip(z + gamma (c - y), u_min - c, u_max - c),  u = c + z.
Measured speed = body-frame pelvis-COM forward velocity (the recorder's base_lin_vel[0]) + est_bias + N(0, est_noise_std^2)
per policy step: simulation ground truth, a real robot needs a state estimator. Python 3.8 compatible.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for _p in (str(HERE), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mujoco  # noqa: E402
import ch5_mujoco as C5  # noqa: E402
from policies12 import load_policy  # noqa: E402
from thesis_eval.metrics import EpisodeRecorder, RecorderParams  # noqa: E402
from thesis_eval.analysis.sp_design import read_csv, static_map  # noqa: E402
from collections import deque

CYCLE_STEPS = 40  # gait period 0.8 s / policy period 0.02 s (g1_env.py: period = 0.8)
CONDITION_KEYS = ("payload_kg", "friction", "command_vx", "command_step", "payload_step", "controller", "gamma_mult",
                  "est_bias", "est_noise_std")
CONDITION_DEFAULTS = dict(payload_kg=0.0, friction=1.0, command_vx=1.0, command_step=None, payload_step=None,
                          controller="none", gamma_mult=1.0, est_bias=0.0, est_noise_std=0.0)
CONTROLLER_DEFAULTS = dict(type="none", gamma=0.0, u_min=0.0, u_max=3.5, t_on_s=2.0)
SETTLED_KEYS = ["window_s", "velocity_rmse", "lateral_velocity_rmse", "yaw_rmse", "mean_vx", "mean_vy",
                "orientation_rms_rad", "foot_slip_mps", "cot", "torque_saturation_rate"]
ID_COLUMNS = ["experiment", "policy_id", "checkpoint", "checkpoint_sha256", "seed", "condition_id", "episode",
              "payload_kg", "friction", "command_vx", "command_step_t_s", "command_step_vx", "payload_step_t_s",
              "payload_step_kg", "controller", "ctrl_type", "gamma", "u_min", "u_max", "t_on_s", "est_bias",
              "est_noise_std"]
CTRL_COLUMNS = ["u_final", "z_final", "u_mean_settled", "sat_frac_settled", "n_updates", "t_settle_s", "n_unstable"]
EXTRA_COLUMNS = ["simulator", "physics_dt", "decimation"]
SETTLE_BAND = 0.05  # m/s, |y_k - c| band used for t_settle_s


# ------------------------------------------------------------------------------------------------ outer loop
class OuterLoop(object):
    """Cycle-synchronous integral command compensation with clamped integrator state (anti-windup).
    type 'none' never changes u but still logs the cycle means y_k (used by the step-identification experiment)."""

    def __init__(self, params, rng=None, cycle_steps=CYCLE_STEPS):
        p = dict(CONTROLLER_DEFAULTS)
        p.update(params or {})
        self.type, self.gamma = p["type"], float(p["gamma"])
        self.u_min, self.u_max, self.t_on = float(p["u_min"]), float(p["u_max"]), float(p["t_on_s"])
        self.bias, self.noise = float(p.get("est_bias", 0.0)), float(p.get("est_noise_std", 0.0))
        if self.type not in ("none", "outer"):
            raise ValueError("unknown controller type %r" % self.type)
        self.rng = rng if rng is not None else np.random.RandomState(0)
        self.cycle_steps = int(cycle_steps)

    def reset(self, c):
        self.c, self.z, self.buf, self.log = float(c), 0.0, [], []
        return self.u

    @property
    def u(self):
        return self.c + self.z

    def _clamp(self, z):
        lo, hi = self.u_min - self.c, self.u_max - self.c
        return min(max(z, lo), hi), (z < lo or z > hi)

    def set_reference(self, c):
        self.c = float(c)
        if self.type == "outer":
            self.z = self._clamp(self.z)[0]
        return self.u

    def observe(self, v_true, n_steps, t):
        """Feed the true forward speed after a policy step; returns the new u at an update, else None."""
        v = v_true + self.bias + (self.noise * self.rng.randn() if self.noise > 0 else 0.0)
        self.buf.append(v)
        if n_steps % self.cycle_steps != 0:
            return None
        full = len(self.buf) == self.cycle_steps
        y = float(np.mean(self.buf)) if full else float("nan")
        self.buf = []
        update = full and self.type == "outer" and t >= self.t_on
        sat = False
        if update:
            self.z, sat = self._clamp(self.z + self.gamma * (self.c - y))
        self.log.append((t, y, self.c, self.u, self.z, float(sat), float(update)))
        return self.u if update else None


# ------------------------------------------------------------------------------------------------ baselines
class StaticInverse(object):
    """Invert survivor means without extrapolation or monotonicity repair."""

    def __init__(self, path, expected_payloads=None, expected_commands=None):
        self.maps = {}
        cells = static_map(read_csv(path))
        if (expected_payloads is None) != (expected_commands is None):
            raise ValueError("both expected payload and command grids are required")
        if expected_payloads is not None:
            expected = {(float(m), float(u)) for m in expected_payloads for u in expected_commands}
            actual = {(cell["payload_kg"], cell["u"]) for cell in cells}
            if not expected or actual != expected:
                raise ValueError("static calibration grid mismatch: missing %s; unexpected %s" %
                                 (sorted(expected - actual), sorted(actual - expected)))
        for cell in cells:
            self.maps.setdefault(cell["payload_kg"], []).append((cell["u"], cell["g"]))
        if not self.maps:
            raise ValueError("empty static calibration")
        grids = []
        for m, cells in self.maps.items():
            cells.sort()
            u, g = np.asarray(cells, dtype=float).T
            if len(u) < 2 or not np.all(np.isfinite(g)) or not np.all(np.diff(g) > 0):
                raise ValueError("missing survivors or non-monotonic static map at payload %g" % m)
            grids.append(u)
        if any(not np.array_equal(grids[0], grid) for grid in grids[1:]):
            raise ValueError("incompatible command grids in static calibration")
        self.grid = grids[0]

    def inverse(self, payload, reference, u_min, u_max):
        masses = sorted(self.maps)
        m = float(payload)
        if not np.isfinite(m) or m < masses[0] or m > masses[-1]:
            raise ValueError("payload outside static calibration: %g" % m)
        i = int(np.searchsorted(masses, m))
        if i < len(masses) and masses[i] == m:
            g = np.asarray(self.maps[m])[:, 1]
        else:
            lo, hi = masses[i - 1], masses[i]
            a = (m - lo) / (hi - lo)
            g = (1 - a) * np.asarray(self.maps[lo])[:, 1] + a * np.asarray(self.maps[hi])[:, 1]
        lo, hi = max(float(u_min), self.grid[0]), min(float(u_max), self.grid[-1])
        if not np.isfinite(reference) or not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
            raise ValueError("invalid reference or empty command intersection")
        u = np.concatenate(([lo], self.grid[(self.grid > lo) & (self.grid < hi)], [hi]))
        return float(np.interp(reference, np.interp(u, self.grid, g), u))


class BaselineLoop(OuterLoop):
    def __init__(self, params, rng=None, cycle_steps=CYCLE_STEPS, policy_dt=0.02):
        p = dict(params)
        self.baseline_type = p["type"]
        p["type"] = "none"
        super(BaselineLoop, self).__init__(p, rng, cycle_steps)
        self.type = self.baseline_type
        self.policy_dt = float(policy_dt)
        self.inverse_map = StaticInverse(params["static_csv"], params.get("expected_payloads"),
                                         params.get("expected_commands")) if self.type == "feedforward" else None

    def reset(self, c, payload=0.0):
        super(BaselineLoop, self).reset(c)
        self.payload, self.active = float(payload), False
        self.window, self.updates, self.pending_update = deque(maxlen=self.cycle_steps), [], False
        self.z = self._clamp(self.z)[0]
        return self.u

    def _refresh(self):
        if self.active and self.type == "feedforward":
            self.z = self.inverse_map.inverse(self.payload, self.c, self.u_min, self.u_max) - self.c
        else:
            self.z = self._clamp(self.z)[0]
        return self.u

    def set_reference(self, c):
        self.c = float(c)
        self._refresh()
        self.pending_update = True
        return self.u

    def set_payload(self, payload):
        self.payload = float(payload)
        self._refresh()
        self.pending_update = True
        return self.u

    def boundary_occupied(self):
        lo, hi = self.u_min, self.u_max
        if self.active and self.type == "feedforward":
            lo, hi = max(lo, self.inverse_map.grid[0]), min(hi, self.inverse_map.grid[-1])
        return float(np.isclose(self.u, lo, rtol=0, atol=1e-12) or
                     np.isclose(self.u, hi, rtol=0, atol=1e-12))

    def observe(self, v_true, n_steps, t):
        v = v_true + self.bias + (self.noise * self.rng.randn() if self.noise > 0 else 0.0)
        self.window.append(v)
        self.buf.append(v)
        was_active = self.active
        self.active = t >= self.t_on
        sat, update = False, False
        if self.active and self.type == "feedforward" and not was_active:
            self._refresh()
            update = True
        elif self.active and self.type == "integral" and len(self.window) == self.cycle_steps:
            y = float(np.mean(self.window))
            self.z, sat = self._clamp(self.z + self.gamma / self.cycle_steps * (self.c - y))
            update = True
        if update:
            self.pending_update = True
            self.updates.append((t, float(np.mean(self.window)), self.c, self.u, self.z, float(sat), 1.0))
        if n_steps % self.cycle_steps == 0:
            y_cycle = float(np.mean(self.buf)) if len(self.buf) == self.cycle_steps else float("nan")
            self.buf = []
            self.log.append((t, y_cycle, self.c, self.u, self.z, float(sat), float(update)))
        return self.u if update else None


# ------------------------------------------------------------------------------------------------ episode
def _sched(pair, policy_dt):
    """[t_s, value] -> (policy step index at which it applies, value) or (-1, None)."""
    if not pair:
        return -1, None
    return int(round(float(pair[0]) / policy_dt)), float(pair[1])


def run_control_episode(sim, policy, cond, ctrl, seed, episode, episode_length_s, warmup_s, settle_s,
                        want_timeseries=False, cond_idx=0):
    """One episode. `cond`: resolved condition dict, `ctrl`: resolved controller dict (type, gamma, u_min, u_max,
    t_on_s, est_bias, est_noise_std). Returns (recorder result, settled result, extras, timeseries or None)."""
    t0 = time.time()
    env = sim.env
    pdt = env.policy_dt
    friction = float(cond["friction"])
    total_mass, base_mass = sim.apply_condition(float(cond["payload_kg"]), friction)
    mass = torch.tensor([total_mass], dtype=torch.float)
    effort = torch.from_numpy(env.effort).float()
    rp = lambda w: RecorderParams(warmup_s=w, base_height_target=C5.BASE_HEIGHT_TARGET)
    rec = EpisodeRecorder(1, pdt, env.dt, rp(warmup_s), mass, effort)
    rec_s = EpisodeRecorder(1, pdt, env.dt, rp(settle_s), mass, effort)

    rng = np.random.RandomState([int(seed), 1, int(episode)])
    baseline = ctrl["type"] in ("integral", "feedforward")
    loop = BaselineLoop(ctrl, rng=rng, policy_dt=pdt) if baseline else OuterLoop(ctrl, rng=rng)
    c = float(cond["command_vx"])
    u = loop.reset(c, cond["payload_kg"]) if baseline else loop.reset(c)
    cs_step, cs_val = _sched(cond.get("command_step"), pdt)
    ps_step, ps_val = _sched(cond.get("payload_step"), pdt)

    pert = C5.initial_state(seed, cond_idx, episode, env.default_pos)
    policy.reset()
    env.reset(cmd=np.array([u, 0.0, 0.0]), perturb=pert)
    obs = np.clip(env.step(np.zeros(12)), -C5.CLIP_OBS, C5.CLIP_OBS)
    d = env.data
    warn0 = d.warning[int(mujoco.mjtWarning.mjWARN_BADQACC)].number

    def hook(tau, qd):
        tt, qq = C5._t(tau), C5._t(qd)
        rec.on_substep(tt, qq)
        rec_s.on_substep(tt, qq)

    def set_u(new_u):
        env.set_command(new_u, 0.0, 0.0)
        return np.clip(env.get_obs(), -C5.CLIP_OBS, C5.CLIP_OBS)  # same state, new command entry

    max_ep = int(math.ceil(round(episode_length_s / pdt, 6)))
    false1 = torch.zeros(1, dtype=torch.bool)
    last, n_unstable = None, 0
    ts = dict(t=[], c=[], u=[], vx=[], payload=[]) if want_timeseries else None
    baseline_steps = []  # actual applied command: t, u, update since preceding step, boundary occupancy
    payload_now, u_settled = float(cond["payload_kg"]), []
    for _ in range(max_ep + 5):
        if cs_step >= 0 and rec.k + 1 == cs_step:
            c = cs_val
            u = loop.set_reference(c)
            obs = set_u(u)
        if ps_step >= 0 and rec.k + 1 == ps_step:
            sim.apply_condition(ps_val, friction)  # recorder total mass (CoT) keeps the initial value
            payload_now = ps_val
            if baseline:
                u = loop.set_payload(payload_now)
                obs = set_u(u)
        action = np.clip(policy.act(obs)[0], -C5.CLIP_ACTIONS, C5.CLIP_ACTIONS)
        obs = np.clip(env.step(action, on_substep=hook), -C5.CLIP_OBS, C5.CLIP_OBS)
        if sim.unstable(warn0) and last is not None:
            st, unstable = last, True
            n_unstable += 1
        else:
            st, unstable = sim.read_state(), False
            last = st
        time_out = rec.k + 2 > max_ep
        fall = unstable or st["pelvis_force"] > C5.TERM_FORCE_N or abs(st["rpy"][1]) > C5.TERM_PITCH or \
            abs(st["rpy"][0]) > C5.TERM_ROLL
        args = (C5._t(st["pos"]), C5._t(st["lin_b"]), C5._t(st["ang_b"]), C5._t(st["rpy"]),
                C5._t(st["pos"][2:3])[:, 0], C5._t(np.array([c, 0.0, 0.0])), C5._t(action), C5._t(st["feet_vel"]),
                C5._t(st["feet_force"]), torch.tensor([bool(st["collision"])]), torch.tensor([bool(fall or time_out)]),
                torch.tensor([bool(time_out)]), false1, false1)
        alive = bool(rec.active[0])
        rec.on_step(*args)
        rec_s.on_step(*args)
        t_now = rec.k * pdt
        if alive and t_now >= settle_s:
            u_settled.append(u)
        if baseline and alive:
            baseline_steps.append((t_now, u, float(loop.pending_update), loop.boundary_occupied()))
            loop.pending_update = False
        if ts is not None and alive:
            for k, v in (("t", t_now), ("c", c), ("u", u), ("vx", float(st["lin_b"][0])), ("payload", payload_now)):
                ts[k].append(v)
        new_u = loop.observe(float(st["lin_b"][0]), env.n_steps, t_now) if alive else None
        if new_u is not None:
            u = new_u
            obs = set_u(u)
        if rec.all_done():
            break
    res, res_s = rec.finalize(), rec_s.finalize()

    cyc = np.array(loop.log, dtype=np.float64).reshape(-1, 7)  # t, y, c, u, z, saturated, updated
    updates = np.array(loop.updates, dtype=np.float64).reshape(-1, 7) if baseline else cyc
    steps = np.asarray(baseline_steps, dtype=np.float64).reshape(-1, 4)
    late = steps[steps[:, 0] >= settle_s] if baseline else cyc[(cyc[:, 0] >= settle_s) & (cyc[:, 6] > 0)]
    t_settle = float("nan")
    if bool(res["survived"][0]) and len(cyc):
        ok = np.abs(cyc[:, 1] - cyc[:, 2]) < SETTLE_BAND
        bad = np.nonzero(~ok)[0]
        if len(bad) == 0:
            t_settle = float(cyc[0, 0])
        elif bad[-1] + 1 < len(cyc):
            t_settle = float(cyc[bad[-1] + 1, 0])
    extras = dict(total_mass=total_mass, base_mass=base_mass, n_unstable=n_unstable, wall_s=time.time() - t0,
                  u_final=loop.u, z_final=loop.z, u_mean_settled=float(np.mean(u_settled)) if u_settled else float("nan"),
                  sat_frac_settled=float(late[:, 3 if baseline else 5].mean()) if len(late) else float("nan"),
                  n_updates=int(steps[:, 2].sum()) if baseline else int(updates[:, 6].sum()) if len(updates) else 0,
                  t_settle_s=t_settle)
    if ts is not None:
        ts = dict((k, np.asarray(v, dtype=np.float64)) for k, v in ts.items())
        ts["cycles"] = cyc
        if baseline:
            ts["updates"] = updates
            ts["baseline_steps"] = steps  # columns: t, applied u, updated, boundary occupied
    return res, res_s, extras, ts


# ------------------------------------------------------------------------------------------------ config
def _cid_value(v):
    if isinstance(v, (list, tuple)):
        return ":".join("%g" % float(x) for x in v)
    if isinstance(v, (int, float)):
        return repr(float(v))
    return str(v)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k in ("defaults", "controllers", "experiments"):
        if k not in cfg:
            raise ValueError("config %s must contain '%s'" % (path, k))
    return cfg


def design_u_max(design, cond):
    """Per-payload cap u_max(m) of the design (eq. B-7) for the heaviest payload of the condition; a payload without an
    admissible command range (None) gets u_max = reference, so the loop can only lower the command."""
    m = max(float(cond["payload_kg"]), float((cond.get("payload_step") or [0.0, 0.0])[1]))
    table = design.get("u_max_by_payload")
    if table is None:
        return float(design["u_max"])
    if "%g" % m not in table:
        raise ValueError("payload %g kg not in design u_max_by_payload %s" % (m, sorted(table)))
    cap = table["%g" % m]
    return float(cond["command_vx"]) if cap is None else float(cap)


def resolve_controller(cfg, cond, design):
    """Preset name + condition overrides -> concrete controller dict. Values 'design' are read from the design JSON."""
    name = cond["controller"]
    if name not in cfg["controllers"]:
        raise ValueError("unknown controller preset %r" % name)
    p = dict(CONTROLLER_DEFAULTS)
    p.update(cfg["controllers"][name] or {})
    for k in ("gamma", "u_max", "u_min"):
        if p[k] == "design":
            if design is None:
                raise ValueError("controller %r uses design values; pass --design" % name)
            p[k] = design_u_max(design, cond) if k == "u_max" else float(design[k])
        p[k] = float(p[k])
    p["gamma"] *= float(cond["gamma_mult"])
    p["est_bias"], p["est_noise_std"] = float(cond["est_bias"]), float(cond["est_noise_std"])
    return p


def expand(cfg, only=None, design=None):
    """-> list of experiment dicts with resolved conditions (cid, cond, ctrl)."""
    names = list(cfg["experiments"])
    if only:
        sel = []
        for o in only:
            hits = [n for n in names if n == o or n.startswith(o)]
            if not hits:
                raise ValueError("--only %r matches no experiment; available: %s" % (o, names))
            sel.extend(n for n in hits if n not in sel)
        names = sel
    out = []
    for name in names:
        exp = cfg["experiments"][name]
        if not exp.get("enabled", True):
            continue
        opts = dict(cfg["defaults"])
        opts.update({k: v for k, v in exp.items() if k not in ("grid", "conditions", "condition", "enabled")})
        base = dict(CONDITION_DEFAULTS)
        base.update(cfg["defaults"].get("condition") or {})
        base.update(exp.get("condition") or {})
        grid = exp.get("grid") or {}
        items = exp.get("conditions") or [{}]  # explicit list, crossed with the grid product
        for k in list(base) + list(grid) + [k for it in items for k in it]:
            if k not in CONDITION_KEYS:
                raise ValueError("unknown condition key %r in %s" % (k, name))
        conds = []
        for combo in itertools.product(*[[(k, v) for v in vals] for k, vals in grid.items()]):
            for it in items:
                cond = dict(base)
                cond.update(dict(combo))
                cond.update(it)
                parts = list(combo) + sorted(it.items())
                cid = "|".join("%s=%s" % (k, _cid_value(v)) for k, v in parts) or "nominal"
                conds.append((cid, cond, resolve_controller(cfg, cond, design)))
        ids = [c[0] for c in conds]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate condition ids in %s" % name)
        if not conds:
            conds.append(("nominal", dict(base), resolve_controller(cfg, base, design)))
        out.append(dict(name=name, opts=opts, conds=conds))
    return out


def needs_design(cfg, only=None):
    try:
        expand(cfg, only, design=None)
        return False
    except ValueError as e:
        if "pass --design" in str(e):
            return True
        raise


# ------------------------------------------------------------------------------------------------ rows
def csv_header():
    keys = C5.recorder_keys()
    return ID_COLUMNS + keys + ["settled_" + k for k in SETTLED_KEYS] + CTRL_COLUMNS + EXTRA_COLUMNS


def make_row(job, res, res_s, extras):
    cond, ctrl = job["cond"], job["ctrl"]
    cs, ps = cond.get("command_step") or [None, None], cond.get("payload_step") or [None, None]
    ids = [job["experiment"], job["policy_id"], job["policy"]["path"], job["sha"], job["seed"], job["cid"],
           job["episode"], cond["payload_kg"], cond["friction"], cond["command_vx"], cs[0], cs[1], ps[0], ps[1],
           cond["controller"], ctrl["type"], ctrl["gamma"], ctrl["u_min"], ctrl["u_max"], ctrl["t_on_s"],
           ctrl["est_bias"], ctrl["est_noise_std"]]
    row = [C5.fmt(v) for v in ids] + [C5.fmt(res[k][0]) for k in C5.recorder_keys()]
    row += [C5.fmt(res_s[k][0]) for k in SETTLED_KEYS]
    row += [C5.fmt(extras[k]) for k in CTRL_COLUMNS]
    return row + ["mujoco", C5.fmt(job["opts"]["dt"]), C5.fmt(job["opts"]["decimation"])]


# ------------------------------------------------------------------------------------------------ worker
_W = {}


def run_job(job):
    try:
        o = job["opts"]
        key = (o["dt"], o["decimation"])
        if _W.get("sim_key") != key:
            _W["sim"] = C5.Ch5Sim(o["dt"], o["decimation"])
            _W["sim_key"] = key
        cache = _W.setdefault("policies", {})
        pk = (job["policy"]["kind"], job["policy"]["path"])
        if pk not in cache:
            cache[pk] = load_policy(job["policy"])
        res, res_s, extras, ts = run_control_episode(
            _W["sim"], cache[pk], job["cond"], job["ctrl"], job["seed"], job["episode"], job["episode_length_s"],
            job["warmup_s"], job["settle_s"], want_timeseries=job["timeseries"])
        if ts is not None:
            os.makedirs(job["ts_dir"], exist_ok=True)
            fn = "%s__ep%d.npz" % (C5_safe(job["cid"]), job["episode"])
            np.savez(os.path.join(job["ts_dir"], fn), condition=np.array(json.dumps(job["cond"])),
                     controller=np.array(json.dumps(job["ctrl"])), **ts)
        return dict(exp=job["experiment"], row=make_row(job, res, res_s, extras), extras=extras)
    except Exception:
        return dict(exp=job["experiment"], error="%s ep %s\n%s" % (job["cid"], job["episode"], traceback.format_exc()))


def C5_safe(s):
    return "".join(ch if (ch.isalnum() or ch in "_.-=") else "_" for ch in s)


def _git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


# ------------------------------------------------------------------------------------------------ driver
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "configs" / "short_paper_control.yaml"))
    ap.add_argument("--registry", default=None)
    ap.add_argument("--design", default=None, help="sp_design.json (thesis_eval/analysis/sp_design.py)")
    ap.add_argument("--out_root", default=str(REPO_ROOT / "eval_results" / "short_paper_control"))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--episodes", type=int, default=None, help="override episodes of every selected experiment")
    ap.add_argument("--local_jit", default=None, help="TorchScript policy standing in for the policy (smoke tests)")
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--decimation", type=int, default=4)
    ap.add_argument("--policy", default=None, help="registry run name overriding defaults.policy (second checkpoint)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args(argv)
    if abs(args.dt * args.decimation - 0.02) > 1e-9:
        raise SystemExit("dt * decimation must be 0.02 s")

    cfg = load_config(args.config)
    design, design_sha = None, None
    if args.design:
        with open(args.design, "r", encoding="utf-8") as f:
            design = json.load(f)
        with open(args.design, "rb") as f:  # CRLF -> LF so a Windows checkout hashes like the Linux server
            design_sha = hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()
    elif needs_design(cfg, args.only):
        raise SystemExit("selected experiments use design values: pass --design <sp_design.json>")
    exps = expand(cfg, args.only, design)

    standin = args.local_jit is not None
    out_root = os.path.abspath(args.out_root)
    # Never write into the archived repository results tree, and never merge into
    # an existing output unless the caller explicitly opts in. This guard applies
    # to direct historical launches as well as the top-level guarded launcher.
    archive_root = os.path.abspath(os.path.join(REPO_ROOT, "eval_results"))
    try:
        inside_archive = os.path.commonpath([out_root, archive_root]) == archive_root
    except ValueError:
        inside_archive = False
    if inside_archive:
        raise SystemExit("refusing output inside archived eval_results; choose a new external directory")
    if os.path.exists(out_root) and not args.force and not args.dry_run:
        raise SystemExit("output directory exists; choose a new directory or pass --force after review")
    if not args.dry_run and not os.path.exists(out_root):
        os.makedirs(out_root)
    if not standin:
        print("WARNING: historical checkpoint loading may execute code; use only trusted, hash-verified weights.",
              file=sys.stderr)
    if standin:
        out_root = os.path.join(out_root, "_jit_standin")
    policy_id = args.policy or cfg["defaults"]["policy"]
    if standin:
        if not os.path.isfile(args.local_jit):
            raise SystemExit("--local_jit file not found: %s" % args.local_jit)
        ckpt, sha_reg = os.path.abspath(args.local_jit), ""
    else:
        if not args.registry:
            raise SystemExit("--registry is required (or --local_jit for a smoke test)")
        from thesis_eval.registry import load_registry
        reg = load_registry(args.registry)
        if policy_id not in reg:
            raise SystemExit("policy %s not in registry" % policy_id)
        ckpt, sha_reg = reg[policy_id]["final_checkpoint"], reg[policy_id].get("final_sha256", "")

    n_total = 0
    for e in exps:
        n_ep = int(args.episodes if args.episodes is not None else e["opts"]["episodes"])
        e["n_ep"] = n_ep
        n_total += n_ep * len(e["conds"])
    print("expanded %d experiments, %d episodes%s" % (len(exps), n_total, "  [JIT STAND-IN]" if standin else ""))
    for e in exps:
        print("  %-28s conditions=%3d episodes=%4d length=%5.1fs" % (
            e["name"], len(e["conds"]), e["n_ep"] * len(e["conds"]), float(e["opts"]["episode_length_s"])))
    if args.dry_run:
        return 0

    if not os.path.isfile(ckpt):
        raise SystemExit("checkpoint not found: %s" % ckpt)
    sha = C5.file_sha256(ckpt)
    if sha_reg and sha_reg != sha:
        raise SystemExit("checkpoint sha256 mismatch for %s" % ckpt)
    pol = dict(kind="jit" if standin else "ckpt", path=ckpt, activation=cfg["defaults"].get("activation", "elu"))
    pid_out = policy_id + (C5.STANDIN_SUFFIX if standin else "")
    opts = dict(dt=args.dt, decimation=args.decimation)

    jobs, todo = [], []
    for e in exps:
        exp_dir = os.path.join(out_root, e["name"])
        if os.path.exists(os.path.join(exp_dir, "episodes_all.csv")) and not args.force:
            print("skip %s (episodes_all.csv exists; --force to re-run)" % e["name"])
            continue
        todo.append(e)
        o = e["opts"]
        for ci, (cid, cond, ctrl) in enumerate(e["conds"]):
            for ep in range(e["n_ep"]):
                jobs.append(dict(experiment=e["name"], cid=cid, cond=cond, ctrl=ctrl, episode=ep, seed=int(o["seed"]),
                                 episode_length_s=float(o["episode_length_s"]), warmup_s=float(o["warmup_s"]),
                                 settle_s=float(o["settle_s"]), timeseries=ep < int(o.get("timeseries_episodes", 0)),
                                 ts_dir=os.path.join(exp_dir, "timeseries"), policy=pol, policy_id=pid_out, sha=sha,
                                 opts=opts))

    t0 = time.time()
    rows, extras, errors = dict((e["name"], []) for e in todo), dict((e["name"], []) for e in todo), []
    pool = Pool(args.workers, initializer=C5._worker_init) if args.workers > 1 and jobs else None
    results = pool.imap_unordered(run_job, jobs, chunksize=2) if pool else (run_job(j) for j in jobs)
    for i, r in enumerate(results, 1):
        if "error" in r:
            errors.append(r["error"])
            print("ERROR %s: %s" % (r["exp"], r["error"]), flush=True)
        else:
            rows[r["exp"]].append(r["row"])
            extras[r["exp"]].append(r["extras"])
        if i % 200 == 0 or i == len(jobs):
            print("[%6.0fs] %d/%d episodes" % (time.time() - t0, i, len(jobs)), flush=True)
    if pool:
        pool.close()
        pool.join()

    ci, ei = ID_COLUMNS.index("condition_id"), ID_COLUMNS.index("episode")
    for e in todo:
        exp_dir = os.path.join(out_root, e["name"])
        os.makedirs(exp_dir, exist_ok=True)
        rs = sorted(rows[e["name"]], key=lambda r: ([c for c, _, _ in e["conds"]].index(r[ci]), int(r[ei])))
        expected = e["n_ep"] * len(e["conds"])
        name = "episodes_all.csv" if len(rs) == expected else "episodes_partial.csv"
        with open(os.path.join(exp_dir, name), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(csv_header())
            w.writerows(rs)
        ex = extras[e["name"]]
        meta = dict(experiment=e["name"], config=os.path.abspath(args.config), config_sha256=C5.file_sha256(args.config),
                    design=os.path.abspath(args.design) if args.design else None, design_sha256=design_sha,
                    design_values=design, policy_id=pid_out, policy=pol, checkpoint_sha256=sha, jit_standin=standin,
                    options={k: v for k, v in e["opts"].items()}, episodes_expected=expected, episodes_written=len(rs),
                    conditions=[dict(condition_id=c, condition=cd, controller=ct) for c, cd, ct in e["conds"]],
                    physics_dt=args.dt, decimation=args.decimation, cycle_steps=CYCLE_STEPS,
                    initial_state="ch5_mujoco.initial_state(seed, 0, episode): identical for every condition (paired)",
                    recorder_reference="EpisodeRecorder receives the reference c(t); the policy observes u = c + z",
                    measured_speed="body-frame pelvis COM forward velocity (ground truth) + est_bias + white noise",
                    payload_step_note="total_mass_kg / CoT keep the initial payload",
                    friction_rule=C5.SIM_RULES["friction_rule"]["equal"],
                    payload_inertia_rule=C5.SIM_RULES["payload_inertia"]["scale"],
                    velocity_point=C5.SIM_RULES["vel_point"]["com"],
                    n_unstable_episodes=int(sum(1 for x in ex if x["n_unstable"] > 0)),
                    mean_episode_wall_s=float(np.mean([x["wall_s"] for x in ex])) if ex else None,
                    workers=args.workers, git_commit=_git_commit(), mujoco_version=mujoco.__version__,
                    torch_version=torch.__version__, numpy_version=np.__version__,
                    end_time=time.strftime("%Y-%m-%dT%H:%M:%S"))
        if design and "round3" in design:
            with open(args.config, "rb") as config_file:
                meta["config_sha256_lf_normalized"] = hashlib.sha256(
                    config_file.read().replace(b"\r\n", b"\n")).hexdigest()
        with open(os.path.join(exp_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
        print("%s: wrote %s (%d/%d rows)" % (e["name"], name, len(rs), expected), flush=True)
    if errors:
        with open(os.path.join(out_root, "errors.log"), "a", encoding="utf-8") as f:
            f.write("\n\n".join(errors) + "\n")
    print("done: %d episodes in %.1f s wall, errors %d" % (len(jobs), time.time() - t0, len(errors)), flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
