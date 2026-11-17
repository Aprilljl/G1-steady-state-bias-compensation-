"""Batched offline evaluation of one G1 checkpoint: one episode per env, per-env conditions.

Usage (remote server):
    EVAL_SPEC=/path/spec.json python legged_gym/scripts/eval_batch.py --task g1 --headless
The spec JSON is produced by thesis_eval/scenarios.py; metrics come from thesis_eval/metrics.py.
"""
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime

# Repo root on sys.path so that `legged_gym` and `thesis_eval` import without installation.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import isaacgym  # noqa: F401  (must be imported before torch)
from isaacgym import gymtorch
from legged_gym.envs import *  # noqa: F401,F403  (registers tasks)
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import set_seed
from legged_gym.utils.terrain import Terrain

import numpy as np
import torch

from thesis_eval.metrics import EpisodeRecorder, RecorderParams
from thesis_eval.policy_arch import apply_arch_to_train_cfg, file_sha256, infer_arch, load_checkpoint

# Per-env values read by the patched creation callbacks (filled in main() before the env is built).
_STATE = {"friction": None, "payload": None, "total_mass": None, "base_mass": None}

# (name, width) of the time-series blocks stored per policy step.
TS_FIELDS = [("commands", 3), ("base_lin_vel", 3), ("base_ang_vel", 3), ("rpy", 3), ("rel_height", 1),
             ("base_pos", 3), ("dof_pos", 12), ("dof_vel", 12), ("actions", 12), ("torques", 12),
             ("feet_pos_z", 2), ("feet_vel", 6), ("feet_force_z", 2), ("feet_contact", 2)]

CSV_ID_COLUMNS = ["launch_id", "experiment", "policy_id", "checkpoint", "checkpoint_sha256", "seed", "terrain",
                  "mesh_type", "terrain_difficulty", "ground_friction", "env_index", "condition_id", "episode",
                  "payload_kg", "friction", "command_vx", "command_vy", "command_yaw", "push_dir_deg",
                  "push_mag_mps", "push_time_s", "push_impulse_ns", "base_mass_kg", "terrain_col", "terrain_row"]


# ---------------------------------------------------------------- monkeypatches (before env creation)
def patch_terrain(difficulty):
    def curiculum(self):  # same loops as Terrain.curiculum, but constant difficulty on every row
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                terrain = self.make_terrain(j / self.cfg.num_cols + 0.001, difficulty)
                self.add_terrain_to_map(terrain, i, j)
    Terrain.curiculum = curiculum


def patch_robot_props():
    def process_rigid_shape_props(self, props, env_id):
        for s in range(len(props)):
            props[s].friction = float(_STATE["friction"][env_id])
        return props

    def process_rigid_body_props(self, props, env_id):
        _STATE["base_mass"][env_id] = props[0].mass
        props[0].mass += float(_STATE["payload"][env_id])  # payload rigidly attached to the pelvis
        _STATE["total_mass"][env_id] = sum(p.mass for p in props)
        return props

    LeggedRobot._process_rigid_shape_props = process_rigid_shape_props
    LeggedRobot._process_rigid_body_props = process_rigid_body_props


# ---------------------------------------------------------------- helpers
def fmt(v):
    if v is None:
        return "nan"
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)):
        return "nan" if v != v else "%.9g" % v
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return str(v)


def json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def git_commit(path):
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def optf(env_spec, key):
    v = env_spec.get(key)
    return None if v is None else float(v)


def write_timeseries(spec, envs, ts_ids, ts_rows, ts_active, dt):
    out_dir = spec["out_timeseries_dir"]
    os.makedirs(out_dir, exist_ok=True)
    data = np.stack(ts_rows)      # (T, S, F)
    active = np.stack(ts_active)  # (T, S)
    for s, env_i in enumerate(ts_ids):
        n = int(active[:, s].sum())  # active rows form a prefix of the episode
        arr = data[:n, s, :]
        out, col = {"time_s": (np.arange(n) + 1) * dt}, 0
        for name, width in TS_FIELDS:
            block = arr[:, col:col + width]
            out[name] = block.reshape(n, 2, 3) if name == "feet_vel" else (block[:, 0] if width == 1 else block)
            col += width
        e = envs[env_i]
        cid = re.sub(r"[^A-Za-z0-9_.\-]+", "_", str(e["condition_id"]))
        np.savez(os.path.join(out_dir, "%s__ep%s.npz" % (cid, e["episode"])),
                 condition_json=np.array(json.dumps(e, default=json_default)), **out)


# ---------------------------------------------------------------- main
def main():
    t_start, start_iso = time.time(), datetime.now().isoformat()
    spec_path = os.environ.get("EVAL_SPEC")
    if not spec_path or not os.path.isfile(spec_path):
        raise RuntimeError("EVAL_SPEC must point to an existing launch spec JSON (got %r)" % spec_path)
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    ckpt_path = spec["checkpoint"]
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError("checkpoint not found: %s" % ckpt_path)
    sha = file_sha256(ckpt_path)
    if spec.get("checkpoint_sha256") and spec["checkpoint_sha256"] != sha:
        raise RuntimeError("checkpoint sha256 mismatch: spec %s vs file %s" % (spec["checkpoint_sha256"], sha))

    envs = sorted(spec["envs"], key=lambda e: e["env_index"])
    n_envs = len(envs)
    if [e["env_index"] for e in envs] != list(range(n_envs)):
        raise ValueError("env_index must be contiguous from 0")
    terr = spec["terrain"]
    is_mesh = terr["mesh_type"] == "trimesh"

    args = get_args()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = n_envs
    env_cfg.env.episode_length_s = spec["episode_length_s"]
    env_cfg.env.test = False  # test=True would make step() sleep to real time
    env_cfg.seed = spec["seed"]
    train_cfg.seed = spec["seed"]
    env_cfg.noise.add_noise = bool(spec["add_noise"])
    env_cfg.domain_rand.push_robots = False  # pushes are applied deterministically below
    env_cfg.commands.heading_command = False
    env_cfg.commands.curriculum = False
    env_cfg.commands.resampling_time = 1e6
    env_cfg.terrain.mesh_type = terr["mesh_type"]
    env_cfg.terrain.static_friction = float(terr["ground_friction"])
    env_cfg.terrain.dynamic_friction = float(terr["ground_friction"])
    if is_mesh:
        if terr.get("difficulty") is None:
            raise ValueError("trimesh terrain needs a numeric 'difficulty'")
        env_cfg.terrain.curriculum = True  # Terrain builds via the (patched) curiculum()
        if terr.get("proportions"):
            if abs(sum(terr["proportions"]) - 1.0) > 1e-6:
                raise ValueError("terrain proportions must sum to 1: %s" % terr["proportions"])
            env_cfg.terrain.terrain_proportions = [float(p) for p in terr["proportions"]]
        patch_terrain(float(terr["difficulty"]))

    _STATE["friction"] = np.array([float(e["friction"]) for e in envs])
    _STATE["payload"] = np.array([float(e["payload_kg"]) for e in envs])
    _STATE["total_mass"] = np.zeros(n_envs)
    _STATE["base_mass"] = np.zeros(n_envs)
    patch_robot_props()

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    dev = env.device
    body_names = list(env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0]))
    if body_names[0] != "pelvis":
        raise RuntimeError("body 0 is %r, expected 'pelvis' (payload is added to props[0])" % body_names[0])

    # ---- policy
    ckpt, sd = load_checkpoint(ckpt_path)
    arch = infer_arch(sd)
    if arch["num_actor_obs"] != env.num_obs or arch["num_actions"] != env.num_actions:
        raise RuntimeError("checkpoint obs/actions (%d/%d) != env (%d/%d)" % (
            arch["num_actor_obs"], arch["num_actions"], env.num_obs, env.num_actions))
    if env.num_privileged_obs is not None and arch["num_critic_obs"] != env.num_privileged_obs:
        raise RuntimeError("checkpoint critic obs %d != env privileged obs %d" % (
            arch["num_critic_obs"], env.num_privileged_obs))
    activation = spec.get("policy_overrides", {}).get("activation", "elu")
    apply_arch_to_train_cfg(train_cfg, arch, activation=activation)
    train_cfg.runner.resume = False
    runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None)
    actor_critic = runner.alg.actor_critic
    actor_critic.load_state_dict(sd)  # strict
    policy = runner.get_inference_policy(device=dev)
    recurrent = arch["policy_class_name"] == "ActorCriticRecurrent"

    # ---- placement
    if is_mesh:
        num_cols = env_cfg.terrain.num_cols
        env.terrain_levels[:] = 0  # row 0: the whole terrain length lies ahead along +x
        env.terrain_types[:] = torch.arange(n_envs, device=dev) % num_cols
        env.env_origins[:] = env.terrain_origins[env.terrain_levels, env.terrain_types]
        env.cfg.terrain.curriculum = False
        terrain_col = env.terrain_types.cpu().numpy()
        terrain_row = env.terrain_levels.cpu().numpy()
    else:
        terrain_col = terrain_row = np.full(n_envs, -1)

    set_seed(spec["seed"])  # reproducible reset noise independent of network construction
    env.reset()
    # Recurrent hidden states start as None and are zero-initialised by the first forward pass;
    # rsl_rl 1.0.x Memory.reset() iterates hidden_states, so it must not be called before that.
    cmd = torch.tensor([[e["command_vx"], e["command_vy"], e["command_yaw"]] for e in envs],
                       dtype=torch.float, device=dev)
    env.commands[:, :3] = cmd
    env.compute_observations()
    obs = env.get_observations()

    # ---- recorder and pushes
    dt, sim_dt = env.dt, env.sim_params.dt
    push_time = [optf(e, "push_time_s") for e in envs]
    push_dir = [optf(e, "push_dir_deg") for e in envs]
    push_mag = np.array([float(e.get("push_mag_mps") or 0.0) for e in envs])
    has_push = np.array([t is not None and d is not None and m > 0 for t, d, m in zip(push_time, push_dir, push_mag)])
    push_step = torch.tensor([int(round(t / dt)) if h else -1 for t, h in zip(push_time, has_push)],
                             dtype=torch.long, device=dev)
    push_dir_rad = torch.tensor([math.radians(d) if h else 0.0 for d, h in zip(push_dir, has_push)],
                                dtype=torch.float, device=dev)
    push_mag_t = torch.tensor(push_mag, dtype=torch.float, device=dev)

    kwargs = {"warmup_s": spec["warmup_s"], "base_height_target": env_cfg.rewards.base_height_target}
    kwargs.update(spec.get("recorder_params") or {})
    params = RecorderParams(**kwargs)
    total_mass = torch.tensor(_STATE["total_mass"], dtype=torch.float, device=dev)
    rec = EpisodeRecorder(n_envs, dt, sim_dt, params, total_mass, env.torque_limits, push_step=push_step, device=dev)

    all_ids = torch.arange(n_envs, device=dev)
    ts_ids = [i for i, e in enumerate(envs) if e.get("save_timeseries")]
    ts_idx = torch.tensor(ts_ids, dtype=torch.long, device=dev)
    ts_rows, ts_active = [], []
    border_x = env_cfg.terrain.num_rows * env_cfg.terrain.terrain_length
    border_y = env_cfg.terrain.num_cols * env_cfg.terrain.terrain_width

    # ---- hooks (instance attributes shadow the class methods used inside env.step)
    orig_torques, orig_reward = env._compute_torques, env.compute_reward

    def torques_hook(actions):
        torques = orig_torques(actions)
        rec.on_substep(torques, env.dof_vel)  # dof_vel here is the pre-substep velocity
        return torques

    def reward_hook():
        orig_reward()
        # check_termination() and compute_reward() run before reset_idx(): all states are pre-reset.
        pos = env.root_states[:, 0:3]
        if is_mesh:
            rel_height = pos[:, 2] - env._get_terrain_height_at_xy(all_ids)
            abnormal = (rel_height > 1.5) | (rel_height < -0.15) | (pos[:, 2] < -1.0) | (pos[:, 2] > 5.0)
            out_of_terrain = ((pos[:, 0] < 0) | (pos[:, 0] > border_x) | (pos[:, 1] < 0) | (pos[:, 1] > border_y))
        else:
            rel_height = pos[:, 2]
            abnormal = out_of_terrain = torch.zeros(n_envs, dtype=torch.bool, device=dev)
        feet_force = env.contact_forces[:, env.feet_indices, :]
        collision = (env.contact_forces[:, env.penalised_contact_indices, :].norm(dim=-1) > 1.0).any(dim=1)
        if ts_ids:
            fz = feet_force[:, :, 2]
            parts = [env.commands[:, :3], env.base_lin_vel, env.base_ang_vel, env.rpy, rel_height.unsqueeze(1), pos,
                     env.dof_pos, env.dof_vel, env.actions, env.torques, env.feet_pos[:, :, 2],
                     env.feet_vel.reshape(n_envs, 6), fz, (fz > params.contact_force_n).float()]
            ts_rows.append(torch.cat([p[ts_idx] for p in parts], dim=1).float().cpu().numpy())
            ts_active.append(rec.active[ts_idx].cpu().numpy())
        rec.on_step(pos, env.base_lin_vel, env.base_ang_vel, env.rpy, rel_height, env.commands[:, :3], env.actions,
                    env.feet_vel, feet_force, collision, env.reset_buf.bool(), env.time_out_buf.bool(),
                    abnormal, out_of_terrain)

    env._compute_torques = torques_hook
    env.compute_reward = reward_hook

    # ---- main loop
    max_iter = int(env.max_episode_length) + 5
    loop_steps, t_loop = 0, time.time()
    with torch.no_grad():
        for it in range(max_iter):
            ids = ((push_step == rec.k + 1) & rec.active).nonzero(as_tuple=False).flatten()
            if len(ids) > 0:
                # push direction is relative to the robot heading: 0 deg = forward, 90 deg = robot's left
                theta = env.rpy[ids, 2] + push_dir_rad[ids]
                env.root_states[ids, 7] += push_mag_t[ids] * torch.cos(theta)
                env.root_states[ids, 8] += push_mag_t[ids] * torch.sin(theta)
                env.gym.set_actor_root_state_tensor_indexed(
                    env.sim, gymtorch.unwrap_tensor(env.root_states), gymtorch.unwrap_tensor(ids.int()), len(ids))
            env.commands[:, :3] = cmd  # reset_idx resamples commands of finished envs; keep them fixed
            actions = policy(obs.detach())
            obs, _, _, dones, _ = env.step(actions.detach())
            if recurrent:
                actor_critic.reset(dones)
            loop_steps += 1
            if loop_steps % 100 == 0:
                print("[eval] step %d, active %d/%d, %.1fs" % (
                    loop_steps, int(rec.active.sum()), n_envs, time.time() - t_loop), flush=True)
            if rec.all_done():
                break

    # ---- outputs
    res = rec.finalize()
    sha_out = sha
    os.makedirs(os.path.dirname(os.path.abspath(spec["out_csv"])), exist_ok=True)
    tm = _STATE["total_mass"]
    with open(spec["out_csv"], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_ID_COLUMNS + list(res.keys()))
        for i, e in enumerate(envs):
            row = [spec["launch_id"], spec["experiment"], spec["policy_id"], ckpt_path, sha_out, spec["seed"],
                   terr["name"], terr["mesh_type"], terr.get("difficulty"), terr["ground_friction"], e["env_index"],
                   e["condition_id"], e["episode"], e["payload_kg"], e["friction"], e["command_vx"], e["command_vy"],
                   e["command_yaw"], optf(e, "push_dir_deg"), push_mag[i], optf(e, "push_time_s"),
                   tm[i] * push_mag[i], _STATE["base_mass"][i], terrain_col[i], terrain_row[i]]
            w.writerow([fmt(v) for v in row] + [fmt(res[k][i]) for k in res])
    if ts_ids:
        write_timeseries(spec, envs, ts_ids, ts_rows, ts_active, dt)

    ft = env_cfg.terrain
    meta = {k: v for k, v in spec.items() if k != "envs"}
    meta.update({
        "n_envs": n_envs, "arch": arch, "activation": activation, "checkpoint_sha256": sha_out,
        "recorder_params": params.as_dict(), "dt": dt, "sim_dt": sim_dt, "decimation": env_cfg.control.decimation,
        "max_episode_length": int(env.max_episode_length),
        "env_cfg_terrain": {k: getattr(ft, k) for k in (
            "mesh_type", "horizontal_scale", "vertical_scale", "border_size", "num_rows", "num_cols",
            "terrain_length", "terrain_width", "terrain_proportions", "slope_treshold", "static_friction",
            "dynamic_friction")},
        "p_gains": env.p_gains.cpu().tolist(), "d_gains": env.d_gains.cpu().tolist(),
        "torque_limits": env.torque_limits.cpu().tolist(), "dof_names": list(env.dof_names),
        "feet_indices": env.feet_indices.cpu().tolist(),
        "feet_body_names": [body_names[i] for i in env.feet_indices.cpu().tolist()], "body_names": body_names,
        "base_mass_note": "base_mass_kg = pelvis mass before adding payload; total_mass_kg includes payload",
        "git_commit": git_commit(ROOT_DIR), "python": sys.version.split()[0], "torch": torch.__version__,
        "numpy": np.__version__, "wall_time_s": time.time() - t_start, "start_time": start_iso,
        "end_time": datetime.now().isoformat(), "loop_steps": loop_steps,
    })
    os.makedirs(os.path.dirname(os.path.abspath(spec["out_meta"])), exist_ok=True)
    with open(spec["out_meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=json_default)
    print("[eval] done: %d envs, %d steps, %.1fs -> %s" % (n_envs, loop_steps, time.time() - t_start, spec["out_csv"]),
          flush=True)


if __name__ == "__main__":
    main()
    # Outputs are closed above; skip interpreter teardown because Isaac Gym can hang on exit.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
