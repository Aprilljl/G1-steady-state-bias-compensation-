"""Chapter-5 cross-simulator evaluation (experiment family E5-11): the Isaac Gym conditions of
`thesis_eval/configs/chapter5.yaml` re-run in MuJoCo with the SAME metrics (`thesis_eval/metrics.py::EpisodeRecorder`)
and the SAME CSV columns as `legged_gym/scripts/eval_batch.py` (+ simulator, physics_dt, decimation).

    python thesis_eval/mujoco_g1_12dof/ch5_mujoco.py --config thesis_eval/mujoco_g1_12dof/configs/chapter5_mujoco.yaml \
        --registry eval_results/registry.csv --out_root eval_results/chapter5_mujoco --workers 16 [--only E5-11a E5-11d]
    # local smoke test with a TorchScript stand-in (never real results; outputs go to <out_root>/_jit_standin)
    python thesis_eval/mujoco_g1_12dof/ch5_mujoco.py --config ... --local_jit PATH/policy_1.pt --only E5-11_smoke

Episode protocol (mirrors eval_batch.py / legged_gym, see README_ch5.md for the differences that remain)
  reset  : joints = default * U(0.5, 1.5) (_reset_dofs), root lin/ang vel = U(-0.5, 0.5) (_reset_root_states), pelvis z = 0.85,
           one unrecorded zero-action policy step (BaseTask.reset() calls step(0)), then the recorded loop.
  loop   : push (heading-relative velocity increment on the base) is applied before policy step `push_step` (as in
           eval_batch: when rec.k + 1 == push_step); obs -> policy -> env.step with on_substep per physics substep ->
           termination -> EpisodeRecorder.on_step with the post-physics pre-reset state.
  end    : pelvis-floor contact force > 1 N, |pitch| > 1.0, |roll| > 0.8, or episode_length_buf > max_episode_length.
Python 3.8 compatible (server: mujoco 3.2.3, no MjSpec).
"""
import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):  # one BLAS thread per worker process
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for _p in (str(HERE), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mujoco  # noqa: E402
from g1_12_env import ACTION_SCALE, DEFAULT_XML, G1Mujoco12Env  # noqa: E402
from policies12 import load_policy  # noqa: E402
from thesis_eval.metrics import EpisodeRecorder, RecorderParams  # noqa: E402

# Copy of legged_gym/scripts/eval_batch.py::CSV_ID_COLUMNS (that module imports isaacgym at import time and cannot be
# imported here). tests/test_ch5_mujoco.py asserts that both lists are identical.
CSV_ID_COLUMNS = ["launch_id", "experiment", "policy_id", "checkpoint", "checkpoint_sha256", "seed", "terrain",
                  "mesh_type", "terrain_difficulty", "ground_friction", "env_index", "condition_id", "episode",
                  "payload_kg", "friction", "command_vx", "command_vy", "command_yaw", "push_dir_deg",
                  "push_mag_mps", "push_time_s", "push_impulse_ns", "base_mass_kg", "terrain_col", "terrain_row"]
EXTRA_COLUMNS = ["simulator", "physics_dt", "decimation"]
STANDIN_SUFFIX = "@jit_standin"

INIT_HEIGHT = 0.85          # g1_config.init_state.pos[2]; the lowest foot sphere is lifted above the floor if needed
BASE_HEIGHT_TARGET = 0.78   # g1_config.rewards.base_height_target (RecorderParams.base_height_target in eval_batch)
CLIP_ACTIONS = 100.0        # normalization.clip_actions
CLIP_OBS = 100.0            # normalization.clip_observations
TERM_PITCH, TERM_ROLL, TERM_FORCE_N = 1.0, 0.8, 1.0   # LeggedRobot.check_termination
SIM_RULES = {
    "friction_rule": {
        "equal": "robot geoms = floor = mu; MuJoCo contact friction is the MAX of the two geoms -> effective mu",
        "physx_avg": "robot geoms = floor = (mu + ground_friction) / 2 -> effective (mu + ground_friction) / 2, "
                     "the PhysX default 'average' combine rule (assumed for Isaac Gym)"},
    "payload_inertia": {
        "scale": "pelvis mass += payload, pelvis inertia scaled by (m + payload) / m, COM unchanged, mj_setConst",
        "none": "pelvis mass += payload, pelvis inertia unchanged, mj_setConst"},
    "vel_point": {
        "com": "base/foot velocities are the velocity of the body COM (PhysX / Isaac Gym rigid-body state convention)",
        "origin": "base/foot velocities are the velocity of the body frame origin"},
}


def fmt(v):
    """Same formatting as eval_batch.fmt (kept identical so merged CSVs from both simulators parse alike)."""
    if v is None:
        return "nan"
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)):
        return "nan" if v != v else "%.9g" % v
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return str(v)


def optf(env_spec, key):
    v = env_spec.get(key)
    return None if v is None else float(v)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ conditions
def set_friction(model, floor_gid, mu):
    """Sliding friction coefficient of every geom (robot shapes AND the floor) := mu.
    MuJoCo uses max(geom1, geom2) per friction component, so with both set the effective coefficient is mu."""
    model.geom_friction[:, 0] = mu
    model.geom_friction[floor_gid, 0] = mu


def apply_payload(model, base, pelvis_id, payload_kg, inertia_rule="scale"):
    """Add `payload_kg` to the pelvis body (restoring the unmodified model first). Returns (total_mass, base_mass).
    Isaac Gym: props[0].mass += payload followed by set_actor_rigid_body_properties(recomputeInertia=True), i.e. the
    inertia is recomputed for the new mass; here it is scaled by m_new / m_old (same mass distribution, COM fixed)."""
    model.body_mass[:] = base["mass"]
    model.body_inertia[:] = base["inertia"]
    m0 = float(base["mass"][pelvis_id])
    if payload_kg:
        model.body_mass[pelvis_id] = m0 + payload_kg
        if inertia_rule == "scale":
            model.body_inertia[pelvis_id] = base["inertia"][pelvis_id] * (m0 + payload_kg) / m0
    return float(model.body_mass.sum()), m0


def push_delta_v(yaw, dir_deg, mag):
    """World-frame (dvx, dvy) of a heading-relative push: 0 deg = robot forward, 90 deg = robot left."""
    theta = yaw + math.radians(dir_deg)
    return mag * math.cos(theta), mag * math.sin(theta)


def initial_state(seed, cond_idx, episode, default_pos):
    """Reset noise as legged_gym: joints = default * U(0.5, 1.5) (_reset_dofs), root lin/ang vel U(-0.5, 0.5)
    (_reset_root_states). Depends only on (seed, condition index, episode), hence identical for every policy."""
    rng = np.random.RandomState([int(seed), int(cond_idx), int(episode)])
    scale = rng.uniform(0.5, 1.5, 12)
    vel = rng.uniform(-0.5, 0.5, 6)
    return dict(dq=np.asarray(default_pos) * (scale - 1.0), v=vel[:3].copy(), w=vel[3:].copy())


def euler_xyz(q):
    """legged_gym.utils.isaacgym_utils.get_euler_xyz for a MuJoCo (w, x, y, z) quaternion -> (roll, pitch, yaw)."""
    w, x, y, z = q
    roll = math.atan2(2.0 * (w * x + y * z), w * w - x * x - y * y + z * z)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    yaw = math.atan2(2.0 * (w * z + x * y), w * w + x * x - y * y - z * z)
    return np.array([roll, pitch, yaw])


def quat_rotate_inverse(q, v):
    w, x, y, z = q
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return r.T.dot(v)


# ------------------------------------------------------------------------------------------------ simulator wrapper
class Ch5Sim(object):
    """One MuJoCo model + data. `apply_condition` / `run_episode` are called once per episode on the same model."""

    def __init__(self, dt=0.005, decimation=4, xml_path=None, vel_point="com", friction_rule="equal",
                 payload_inertia="scale"):
        self.env = G1Mujoco12Env(xml_path=xml_path, dt=dt, decimation=decimation)
        self.env.init_height = INIT_HEIGHT
        self.vel_point, self.friction_rule, self.payload_inertia = vel_point, friction_rule, payload_inertia
        m = self.env.model
        self.base = dict(mass=m.body_mass.copy(), inertia=m.body_inertia.copy(), friction=m.geom_friction.copy())
        self.pelvis_id = self.env.root_id
        # contact categories of floor contacts: 0/1 left/right foot, 2 pelvis (termination), 3 hip/knee (collision)
        self.geom_cat = np.full(m.ngeom, -1, dtype=np.int64)
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if b == int(self.env.foot_body_ids[0]):
                self.geom_cat[g] = 0
            elif b == int(self.env.foot_body_ids[1]):
                self.geom_cat[g] = 1
            elif b == self.pelvis_id:
                self.geom_cat[g] = 2
            elif "hip" in name or "knee" in name:
                self.geom_cat[g] = 3
        self._f6 = np.zeros(6)
        self._v6 = np.zeros(6)

    # -- condition --------------------------------------------------------------------------------
    def apply_condition(self, payload_kg, friction, ground_friction=1.0):
        m = self.env.model
        total, base_mass = apply_payload(m, self.base, self.pelvis_id, payload_kg, self.payload_inertia)
        mu = friction if self.friction_rule == "equal" else 0.5 * (friction + ground_friction)
        set_friction(m, self.env.floor_gid, mu)
        mujoco.mj_setConst(m, self.env.data)  # body_subtreemass, dof/body_invweight0 depend on mass and inertia
        return total, base_mass

    # -- readout ----------------------------------------------------------------------------------
    def body_lin_vel(self, bid):
        """World-frame linear velocity of body `bid` at its COM (mj_objectVelocity) or frame origin."""
        d = self.env.data
        mujoco.mj_objectVelocity(self.env.model, d, mujoco.mjtObj.mjOBJ_BODY, int(bid), self._v6, 0)
        v = self._v6[3:6].copy()
        if self.vel_point == "origin":
            v = v + np.cross(self._v6[0:3], d.xpos[bid] - d.xipos[bid])
        return v

    def contact_forces(self):
        """Floor-contact force vectors in the world frame, acting on the robot: feet (2, 3), plus the norms of the
        summed pelvis and hip/knee floor-contact forces."""
        m, d = self.env.model, self.env.data
        feet = np.zeros((2, 3))
        pelvis = np.zeros(3)
        hipknee = np.zeros(3)
        floor = self.env.floor_gid
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 == floor:
                other = g2
            elif g2 == floor:
                other = g1
            else:
                continue
            cat = self.geom_cat[other]
            if cat < 0:
                continue
            mujoco.mj_contactForce(m, d, i, self._f6)
            fr = c.frame  # rows: normal, tangent1, tangent2 (world axes)
            fw = self._f6[0] * fr[0:3] + self._f6[1] * fr[3:6] + self._f6[2] * fr[6:9]
            if fr[2] < 0.0:  # orient so that the normal force on the robot points up (floor normal is +z)
                fw = -fw
            if cat < 2:
                feet[cat] += fw
            elif cat == 2:
                pelvis += fw
            else:
                hipknee += fw
        return feet, float(np.linalg.norm(pelvis)), float(np.linalg.norm(hipknee))

    def read_state(self):
        env, d = self.env, self.env.data
        quat = d.qpos[3:7].copy()
        feet_vel = np.stack([self.body_lin_vel(b) for b in env.foot_body_ids])
        feet_force, pelvis_f, hipknee_f = self.contact_forces()
        return dict(pos=d.qpos[0:3].copy(), rpy=euler_xyz(quat),
                    lin_b=quat_rotate_inverse(quat, self.body_lin_vel(self.pelvis_id)),
                    ang_b=d.qvel[3:6].copy(), feet_vel=feet_vel, feet_force=feet_force,
                    pelvis_force=pelvis_f, collision=hipknee_f > TERM_FORCE_N)

    def unstable(self, warn0=0):
        """Non-finite state or a new MuJoCo bad-qacc warning (MuJoCo then silently resets the data)."""
        d = self.env.data
        return (not np.isfinite(d.qpos).all()) or (not np.isfinite(d.qvel).all()) or \
            d.warning[int(mujoco.mjtWarning.mjWARN_BADQACC)].number > warn0


def _t(a, dtype=torch.float):
    return torch.from_numpy(np.ascontiguousarray(a)).to(dtype).unsqueeze(0)


def run_episode(sim, policy, e, cond_idx, seed, episode_length_s, warmup_s, recorder_params=None,
                ground_friction=1.0):
    """Run one episode of condition `e` (an `envs` entry of a scenarios.expand spec).
    Returns (recorder result dict of 1-element arrays, extras dict)."""
    t0 = time.time()
    env = sim.env
    policy_dt = env.policy_dt
    total_mass, base_mass = sim.apply_condition(float(e["payload_kg"]), float(e["friction"]), ground_friction)

    mag = float(e.get("push_mag_mps") or 0.0)
    pt, pdir = optf(e, "push_time_s"), optf(e, "push_dir_deg")
    has_push = pt is not None and pdir is not None and mag > 0
    push_step = int(round(pt / policy_dt)) if has_push else -1

    kwargs = {"warmup_s": warmup_s, "base_height_target": BASE_HEIGHT_TARGET}
    kwargs.update(recorder_params or {})
    params = RecorderParams(**kwargs)
    rec = EpisodeRecorder(1, policy_dt, env.dt, params, torch.tensor([total_mass], dtype=torch.float),
                          torch.from_numpy(env.effort).float(), push_step=torch.tensor([push_step], dtype=torch.long))

    cmd = np.array([e["command_vx"], e["command_vy"], e["command_yaw"]], dtype=np.float64)
    pert = initial_state(seed, cond_idx, e["episode"], env.default_pos)
    policy.reset()
    env.reset(cmd=cmd, perturb=pert)
    obs = env.step(np.zeros(12))  # BaseTask.reset() runs one zero-action step; it is not recorded
    obs = np.clip(obs, -CLIP_OBS, CLIP_OBS)
    d = env.data
    warn0 = d.warning[int(mujoco.mjtWarning.mjWARN_BADQACC)].number

    def hook(tau, qd):
        rec.on_substep(_t(tau), _t(qd))

    max_ep = int(math.ceil(round(episode_length_s / policy_dt, 6)))  # LeggedRobot.max_episode_length
    false1 = torch.zeros(1, dtype=torch.bool)
    cmd_t = _t(cmd)
    last, n_unstable = None, 0
    for _ in range(max_ep + 5):
        if has_push and rec.k + 1 == push_step:
            dvx, dvy = push_delta_v(euler_xyz(d.qpos[3:7])[2], pdir, mag)
            d.qvel[0] += dvx
            d.qvel[1] += dvy
        action = np.clip(policy.act(obs)[0], -CLIP_ACTIONS, CLIP_ACTIONS)
        obs = np.clip(env.step(action, on_substep=hook), -CLIP_OBS, CLIP_OBS)
        if sim.unstable(warn0) and last is not None:
            st, unstable = last, True  # MuJoCo auto-reset on bad qacc: keep the last finite state and fail the episode
            n_unstable += 1
        else:
            st, unstable = sim.read_state(), False
            last = st
        time_out = rec.k + 2 > max_ep  # episode_length_buf = k + 2 after this step (the reset step counts as 1)
        fall = unstable or st["pelvis_force"] > TERM_FORCE_N or abs(st["rpy"][1]) > TERM_PITCH or \
            abs(st["rpy"][0]) > TERM_ROLL
        rec.on_step(_t(st["pos"]), _t(st["lin_b"]), _t(st["ang_b"]), _t(st["rpy"]), _t(st["pos"][2:3])[:, 0], cmd_t,
                    _t(action), _t(st["feet_vel"]), _t(st["feet_force"]),
                    torch.tensor([bool(st["collision"])]), torch.tensor([bool(fall or time_out)]),
                    torch.tensor([bool(time_out)]), false1, false1)
        if rec.all_done():
            break
    res = rec.finalize()
    return res, dict(total_mass=total_mass, base_mass=base_mass, n_unstable=n_unstable, wall_s=time.time() - t0,
                     steps=rec.k)


def recorder_keys():
    """Column names produced by EpisodeRecorder.finalize (independent of the data)."""
    rec = EpisodeRecorder(1, 0.02, 0.005, RecorderParams(), torch.tensor([1.0]), torch.ones(12))
    return list(rec.finalize().keys())


def csv_header():
    return CSV_ID_COLUMNS + recorder_keys() + EXTRA_COLUMNS


def make_row(spec, e, res, extras, policy_id, ckpt_path, sha, physics_dt, decimation):
    """One CSV row (list of strings) in the column order of csv_header()."""
    terr = spec["terrain"]
    mag = float(e.get("push_mag_mps") or 0.0)
    row = [spec["launch_id"], spec["experiment"], policy_id, ckpt_path, sha, spec["seed"], terr["name"],
           terr["mesh_type"], terr.get("difficulty"), terr["ground_friction"], e["env_index"], e["condition_id"],
           e["episode"], e["payload_kg"], e["friction"], e["command_vx"], e["command_vy"], e["command_yaw"],
           optf(e, "push_dir_deg"), mag, optf(e, "push_time_s"), extras["total_mass"] * mag, extras["base_mass"],
           -1, -1]
    out = [fmt(v) for v in row] + [fmt(res[k][0]) for k in recorder_keys()]
    return out + ["mujoco", fmt(physics_dt), fmt(decimation)]


# ------------------------------------------------------------------------------------------------ worker side
_W = {}


def _worker_init():
    torch.set_num_threads(1)


def _get_sim(opts):
    key = (opts["dt"], opts["decimation"], opts["vel_point"], opts["friction_rule"], opts["payload_inertia"],
           opts["xml_path"])
    if _W.get("sim_key") != key:
        _W["sim"] = Ch5Sim(opts["dt"], opts["decimation"], opts["xml_path"], opts["vel_point"],
                           opts["friction_rule"], opts["payload_inertia"])
        _W["sim_key"] = key
    return _W["sim"]


def _get_policy(pol):
    key = (pol["kind"], pol["path"], pol.get("activation", "elu"))
    cache = _W.setdefault("policies", {})
    if key not in cache:
        cache[key] = load_policy(pol)
    return cache[key]


def run_job(job):
    try:
        sim = _get_sim(job["opts"])
        policy = _get_policy(job["policy"])
        res, extras = run_episode(sim, policy, job["env"], job["cond_idx"], job["seed"], job["episode_length_s"],
                                  job["warmup_s"], job["recorder_params"], job["ground_friction"])
        row = make_row(job["spec_ctx"], job["env"], res, extras, job["policy_id_out"], job["policy"]["path"],
                       job["sha"], job["opts"]["dt"], job["opts"]["decimation"])
        return dict(launch=job["launch"], env_index=job["env"]["env_index"], row=row, extras=extras,
                    survived=bool(res["survived"][0]))
    except Exception:
        return dict(launch=job["launch"], env_index=job["env"]["env_index"], error=traceback.format_exc())


# ------------------------------------------------------------------------------------------------ driver
def _spec_ctx(spec):
    return dict(launch_id=spec["launch_id"], experiment=spec["experiment"], seed=spec["seed"], terrain=spec["terrain"])


def _write_launch(spec, rows, extras_list, args, opts, pol, sha, wall_s, standin):
    import csv
    os.makedirs(os.path.dirname(spec["out_csv"]), exist_ok=True)
    rows = sorted(rows, key=lambda r: int(r[CSV_ID_COLUMNS.index("env_index")]))
    with open(spec["out_csv"], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(csv_header())
        w.writerows(rows)
    m_xml = str(Path(opts["xml_path"]).resolve()) if opts["xml_path"] else str(DEFAULT_XML.resolve())
    meta = {k: v for k, v in spec.items() if k not in ("envs", "out_timeseries_dir")}
    meta.update({
        "simulator": "mujoco", "mujoco_version": mujoco.__version__, "n_envs": len(rows),
        "policy_id_in_csv": spec["policy_id"] + (STANDIN_SUFFIX if standin else ""),
        "jit_standin": bool(standin), "policy": pol, "checkpoint_sha256": sha,
        "physics_dt": opts["dt"], "decimation": opts["decimation"], "policy_dt": opts["dt"] * opts["decimation"],
        "mujoco_friction_rule": SIM_RULES["friction_rule"][opts["friction_rule"]],
        "payload_inertia_rule": SIM_RULES["payload_inertia"][opts["payload_inertia"]],
        "velocity_point": SIM_RULES["vel_point"][opts["vel_point"]],
        "friction_rule_key": opts["friction_rule"], "payload_inertia_key": opts["payload_inertia"],
        "vel_point_key": opts["vel_point"],
        "model_xml": m_xml, "model_xml_sha256": file_sha256(m_xml) if os.path.isfile(m_xml) else None,
        "pd_mode": "explicit, tau = kp (q_t - q) - kd qdot, clipped to actuator force range",
        "action_scale": ACTION_SCALE,
        "initial_state": "joints default*U(0.5,1.5), root lin/ang vel U(-0.5,0.5), pelvis z %.2f, "
                         "RandomState([seed, condition index, episode]); one unrecorded zero-action step" % INIT_HEIGHT,
        "termination": "pelvis floor-contact force > 1 N, |pitch| > 1.0, |roll| > 0.8, time-out at "
                       "episode_length_buf > max_episode_length",
        "recorder_params_effective": RecorderParams(**dict(
            {"warmup_s": spec["warmup_s"], "base_height_target": BASE_HEIGHT_TARGET},
            **(spec.get("recorder_params") or {}))).as_dict(),
        "n_unstable_episodes": int(sum(1 for x in extras_list if x["n_unstable"] > 0)),
        "mean_episode_wall_s": float(np.mean([x["wall_s"] for x in extras_list])) if extras_list else None,
        "launch_wall_s": wall_s, "workers": args.workers,
        "versions": dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                         mujoco=mujoco.__version__),
        "end_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    os.makedirs(os.path.dirname(spec["out_meta"]), exist_ok=True)
    with open(spec["out_meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "configs" / "chapter5_mujoco.yaml"))
    ap.add_argument("--registry", default=None, help="eval_results/registry.csv (thesis_eval/registry.py)")
    ap.add_argument("--out_root", default=str(REPO_ROOT / "eval_results" / "chapter5_mujoco"))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--episodes", type=int, default=None, help="override episodes of every selected experiment")
    ap.add_argument("--local_jit", default=None, help="TorchScript policy standing in for EVERY policy id (smoke tests)")
    ap.add_argument("--dt", type=float, default=0.005, help="physics step; default 0.005 = legged_gym sim.dt")
    ap.add_argument("--decimation", type=int, default=4, help="default 4 = legged_gym control.decimation")
    ap.add_argument("--friction_rule", choices=sorted(SIM_RULES["friction_rule"]), default="equal")
    ap.add_argument("--payload_inertia", choices=sorted(SIM_RULES["payload_inertia"]), default="scale")
    ap.add_argument("--vel_point", choices=sorted(SIM_RULES["vel_point"]), default="com")
    ap.add_argument("--xml", default=None, help="MJCF scene (default resources/robots/g1_description/scene.xml)")
    ap.add_argument("--force", action="store_true", help="re-run launches whose CSV already exists")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args(argv)

    from thesis_eval import scenarios
    from thesis_eval.registry import load_registry
    from thesis_eval.run_eval import _write_launches_csv, merge_experiment

    if abs(args.dt * args.decimation - 0.02) > 1e-9:
        raise SystemExit("dt * decimation must be 0.02 s (policy period), got %g" % (args.dt * args.decimation))
    xml_path = Path(args.xml).expanduser().resolve() if args.xml else DEFAULT_XML.resolve()
    if not xml_path.is_file():
        raise SystemExit("MJCF scene not found: %s" % xml_path)
    args.xml = str(xml_path)
    out_root = os.path.abspath(args.out_root)
    standin = args.local_jit is not None
    if standin:
        out_root = os.path.join(out_root, "_jit_standin")  # never mixed with real experiment folders
    config = scenarios.load_config(args.config)
    if args.episodes is not None:
        config.setdefault("defaults", {})["episodes"] = int(args.episodes)
        for exp in config["experiments"].values():
            exp.pop("episodes", None)
    if standin:
        if not os.path.isfile(args.local_jit):
            raise SystemExit("--local_jit file not found: %s" % args.local_jit)
        ids = set(p for exp in config["experiments"].values() for p in exp["policies"])
        registry = dict((p, {"final_checkpoint": os.path.abspath(args.local_jit), "final_sha256": ""}) for p in ids)
    else:
        if not args.registry:
            raise SystemExit("--registry is required (or --local_jit for a smoke test)")
        registry = load_registry(args.registry)
    specs = scenarios.expand(config, registry, out_root, only=args.only)
    spec_paths = scenarios.write_specs(specs, out_root)
    n_total = sum(len(s["envs"]) for s in specs)
    print("expanded %d launches, %d episodes in total (dt %g x %d, %d workers)%s" % (
        len(specs), n_total, args.dt, args.decimation, args.workers, "  [JIT STAND-IN]" if standin else ""), flush=True)
    if args.dry_run:
        for s, p in zip(specs, spec_paths):
            print("  %s  envs=%d  -> %s" % (s["launch_id"], len(s["envs"]), p))
        return 0

    opts = dict(dt=args.dt, decimation=args.decimation, vel_point=args.vel_point, friction_rule=args.friction_rule,
                payload_inertia=args.payload_inertia, xml_path=args.xml)
    shas, jobs, run_idx, launch_info = {}, [], [], {}
    status = {}
    for li, spec in enumerate(specs):
        if os.path.exists(spec["out_csv"]) and not args.force:
            status[li] = "skipped"
            continue
        path = spec["checkpoint"]
        if not os.path.isfile(path):
            raise SystemExit("checkpoint not found: %s (launch %s)" % (path, spec["launch_id"]))
        if path not in shas:
            shas[path] = file_sha256(path)
        if spec.get("checkpoint_sha256") and spec["checkpoint_sha256"] != shas[path]:
            raise SystemExit("checkpoint sha256 mismatch for %s" % path)
        pol = dict(kind="jit" if standin else "ckpt", path=path,
                   activation=(spec.get("policy_overrides") or {}).get("activation", "elu"))
        pid_out = spec["policy_id"] + (STANDIN_SUFFIX if standin else "")
        cond_ids = []
        for e in spec["envs"]:
            if e["condition_id"] not in cond_ids:
                cond_ids.append(e["condition_id"])
        ctx = _spec_ctx(spec)
        ground_friction = float(spec["terrain"]["ground_friction"])
        for e in spec["envs"]:
            jobs.append(dict(launch=li, env=e, cond_idx=cond_ids.index(e["condition_id"]), seed=spec["seed"],
                             episode_length_s=spec["episode_length_s"], warmup_s=spec["warmup_s"],
                             recorder_params=spec.get("recorder_params"), ground_friction=ground_friction,
                             policy=pol, policy_id_out=pid_out, sha=shas[path], opts=opts, spec_ctx=ctx))
        launch_info[li] = dict(rows=[], extras=[], n=len(spec["envs"]), pol=pol, sha=shas[path], failed=None)

    t0 = time.time()
    done = 0
    pool = Pool(args.workers, initializer=_worker_init) if args.workers > 1 and jobs else None
    results = pool.imap_unordered(run_job, jobs, chunksize=4) if pool else (run_job(j) for j in jobs)
    for r in results:
        li = r["launch"]
        info = launch_info[li]
        done += 1
        if "error" in r:
            info["failed"] = r["error"]
            print("ERROR in %s env %d:\n%s" % (specs[li]["launch_id"], r["env_index"], r["error"]), flush=True)
        else:
            info["rows"].append(r["row"])
            info["extras"].append(r["extras"])
        if len(info["rows"]) == info["n"]:
            _write_launch(specs[li], info["rows"], info["extras"], args, opts, info["pol"], info["sha"],
                          time.time() - t0, standin)
            status[li] = "ok"
            print("[%5.0fs] %d/%d episodes | wrote %s" % (time.time() - t0, done, len(jobs), specs[li]["out_csv"]),
                  flush=True)
    if pool:
        pool.close()
        pool.join()
    wall = time.time() - t0

    rows_by_exp = {}
    for li, spec in enumerate(specs):
        st = status.get(li, "failed")
        chunk = int(spec["launch_id"].rsplit("__c", 1)[1])
        ex = launch_info.get(li, {}).get("extras") or []
        secs = round(float(sum(x["wall_s"] for x in ex)), 1)  # summed episode time (CPU seconds)
        rows_by_exp.setdefault(spec["experiment"], []).append(
            {"launch_id": spec["launch_id"], "policy": spec["policy_id"], "terrain": spec["terrain"]["name"],
             "chunk": chunk, "n_envs": len(spec["envs"]), "status": st, "seconds": secs})
    for exp, rows in rows_by_exp.items():
        exp_dir = os.path.join(out_root, exp)
        os.makedirs(exp_dir, exist_ok=True)
        _write_launches_csv(exp_dir, rows)
        print("%s: episodes_all.csv rows=%s" % (exp, merge_experiment(exp_dir)), flush=True)
    n_fail = sum(1 for li in range(len(specs)) if status.get(li, "failed") == "failed")
    allx = [x for i in launch_info.values() for x in i["extras"]]
    print("done: %d episodes in %.1f s wall (%.2f s/episode CPU on average), unstable episodes %d, failed launches %d" % (
        len(allx), wall, np.mean([x["wall_s"] for x in allx]) if allx else float("nan"),
        sum(1 for x in allx if x["n_unstable"] > 0), n_fail), flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
