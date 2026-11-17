"""MuJoCo twin of the legged_gym (Isaac Gym) Unitree G1 12-DoF PPO environment: Chapter-3 style policies without motion
prior / mirror constraint, evaluated with the same scenarios and metrics as the 29-DoF AMP policies.

Interface replicated from (read-only sources in this repo)
  legged_gym/envs/g1/g1_env.py::compute_observations   47-dim obs
  legged_gym/envs/g1/g1_config.py                      gains, default pose, action scale 0.25, PPO nets
  deploy/deploy_mujoco/deploy_mujoco.py + configs/g1.yaml   MuJoCo stepping (sim dt 0.002, decimation 10), obs scales
obs (47) = [ang_vel*0.25 (3) | projected gravity (3) | cmd*[2,2,0.25] (3) | (q - q_default)*1.0 (12) | qdot*0.05 (12) |
            last action (12) | sin(2 pi phase), cos(2 pi phase) (2)],  phase = (episode time) % 0.8 / 0.8
Joint order (policy / action order, deploy config): per leg hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll;
left leg first. Torques: explicit PD, tau = kp (q_t - q) - kd qdot, clipped to the MJCF actuator force range
(88 / 139 / 50 Nm = URDF effort limits). The MJCF keeps its own joint damping 0.001, armature 0.01, frictionloss 0.1 (as in
the deploy script). Python 3.8.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mujoco_g1_29dof"))
from g1_mj_env import FootMixin, quat_to_mat, _id  # noqa: E402

REPO_ROOT = HERE.parents[1]
DEFAULT_XML = Path(os.environ.get("UNITREE_RL_GYM_ROOT", REPO_ROOT.parent.parent / "unitree_rl_gym")) / "resources" / "robots" / "g1_description" / "scene.xml"

_TYPES = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]
JOINT_NAMES_12 = ["%s_%s_joint" % (side, t) for side in ("left", "right") for t in _TYPES]
KP = np.array([100, 100, 100, 150, 40, 40] * 2, dtype=np.float64)
KD = np.array([2, 2, 2, 4, 2, 2] * 2, dtype=np.float64)
DEFAULT_POS = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2, dtype=np.float64)
ACTION_SCALE = 0.25
OBS_SCALES = dict(ang_vel=0.25, dof_pos=1.0, dof_vel=0.05, cmd=np.array([2.0, 2.0, 0.25]))
PHASE_PERIOD = 0.8


def mirror_joints_12(q):
    """Left-right swap of a 12-vector (position / offset / action): swap legs, flip roll and yaw."""
    q = np.asarray(q)
    out = np.empty_like(q)
    out[..., 0:6], out[..., 6:12] = q[..., 6:12], q[..., 0:6]
    for j in (1, 2, 7, 8):  # hip_roll, hip_yaw
        out[..., j] = -out[..., j]
    for j in (5, 11):  # ankle_roll
        out[..., j] = -out[..., j]
    return out


def projected_gravity(q):
    """deploy_mujoco.get_gravity_orientation: R^T [0, 0, -1]."""
    w, x, y, z = q
    return np.array([2 * (-z * x + w * y), -2 * (z * y + w * x), 1 - 2 * (w * w + z * z)])


class G1Mujoco12Env(FootMixin):
    def __init__(self, xml_path=None, dt=0.002, decimation=10, friction=1.0):
        self.xml_path = Path(xml_path) if xml_path else DEFAULT_XML
        m = mujoco.MjModel.from_xml_path(str(self.xml_path))
        m.opt.timestep = dt
        if friction != 1.0:
            m.geom_friction[:, 0] = friction
        self.model = m
        self.data = mujoco.MjData(m)
        self.dt, self.decimation, self.policy_dt = dt, decimation, dt * decimation
        jid = [_id(m, "joint", n) for n in JOINT_NAMES_12]
        self.qadr = np.array([m.jnt_qposadr[j] for j in jid])
        self.dadr = np.array([m.jnt_dofadr[j] for j in jid])
        j2a = dict((int(m.actuator_trnid[a, 0]), a) for a in range(m.nu))
        self.act_idx = np.array([j2a[j] for j in jid])
        assert np.allclose(m.actuator_gear[:, 0], 1.0)
        self.effort = np.array([m.jnt_actfrcrange[j, 1] for j in jid], dtype=np.float64)
        self.kp, self.kd, self.default_pos = KP.copy(), KD.copy(), DEFAULT_POS.copy()
        self.root_id = _id(m, "body", "pelvis")
        self.floor_gid = _id(m, "geom", "floor")
        self.foot_geoms = {}
        for side in ("left", "right"):
            bid = _id(m, "body", "%s_ankle_roll_link" % side)
            self.foot_geoms[side] = set(int(g) for g in range(m.ngeom) if m.geom_bodyid[g] == bid)
        self.geom_side = dict((g, si) for si, side in enumerate(("left", "right")) for g in self.foot_geoms[side])
        self.foot_body_ids = np.array([_id(m, "body", "left_ankle_roll_link"), _id(m, "body", "right_ankle_roll_link")])
        self.init_height = 0.8
        self.cmd = np.zeros(3)
        self.last_action = np.zeros(12)
        self.n_steps = 0
        self.step_tau = self.step_abs_power = self.step_sat = np.zeros(12)

    # ---- state ----------------------------------------------------------------------------------------
    def joint_pos(self):
        return self.data.qpos[self.qadr].copy()

    def joint_vel(self):
        return self.data.qvel[self.dadr].copy()

    def root_quat(self):
        return self.data.qpos[3:7].copy()

    def root_pos(self):
        return self.data.qpos[0:3].copy()

    def base_ang_vel_b(self):
        return self.data.qvel[3:6].copy()  # free joint: body-frame angular velocity

    def base_lin_vel_b(self):
        return quat_to_mat(self.root_quat()).T @ self.data.qvel[0:3]

    def set_command(self, vx, vy, wz):
        self.cmd = np.array([vx, vy, wz], dtype=np.float64)

    def get_obs(self):
        phase = (self.n_steps * self.policy_dt) % PHASE_PERIOD / PHASE_PERIOD
        obs = np.concatenate([
            self.base_ang_vel_b() * OBS_SCALES["ang_vel"], projected_gravity(self.root_quat()),
            self.cmd * OBS_SCALES["cmd"], (self.joint_pos() - self.default_pos) * OBS_SCALES["dof_pos"],
            self.joint_vel() * OBS_SCALES["dof_vel"], self.last_action,
            [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]])
        return obs.astype(np.float32)

    def _foot_lowest_z(self):
        zs = [self.data.geom_xpos[g][2] - self.model.geom_size[g][0] for g in self.geom_side
              if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
        return min(zs)

    def reset(self, cmd=(0.0, 0.0, 0.0), perturb=None):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[0:3] = [0.0, 0.0, self.init_height]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[self.qadr] = self.default_pos + (0.0 if perturb is None else np.asarray(perturb["dq"]))
        mujoco.mj_forward(m, d)
        low = self._foot_lowest_z()
        if low < 0.0:
            d.qpos[2] += -low + 1e-3
            mujoco.mj_forward(m, d)
        if perturb is not None:
            d.qvel[0:3] = perturb["v"]
            d.qvel[3:6] = perturb["w"]
            mujoco.mj_forward(m, d)
        self.set_command(*cmd)
        self.last_action = np.zeros(12)
        self.n_steps = 0
        return self.get_obs()

    def step(self, action, on_substep=None):
        """on_substep(tau, qd): optional per-physics-substep hook called with the applied torque and the pre-substep
        joint velocity (the legged_gym `_compute_torques` hook point used by the Chapter-5 evaluation)."""
        action = np.asarray(action, dtype=np.float64)
        q_target = self.default_pos + ACTION_SCALE * action
        d, m = self.data, self.model
        tau_sum, pw_sum, sat_sum = np.zeros(12), np.zeros(12), np.zeros(12)
        for _ in range(self.decimation):
            q = d.qpos[self.qadr]
            qd = d.qvel[self.dadr]
            tau = np.clip(self.kp * (q_target - q) - self.kd * qd, -self.effort, self.effort)
            d.ctrl[self.act_idx] = tau
            if on_substep is not None:
                on_substep(tau, qd)
            tau_sum += tau
            pw_sum += np.abs(tau * qd)
            sat_sum += np.abs(tau) >= 0.99 * self.effort
            mujoco.mj_step(m, d)
        n = float(self.decimation)
        self.step_tau, self.step_abs_power, self.step_sat = tau_sum / n, pw_sum / n, sat_sum / n
        self.last_action = action.copy()
        self.n_steps += 1
        return self.get_obs()

    def snapshot(self):
        touch, fn = self.foot_info()
        fpos, fvel = self.foot_state()
        return dict(base_pos=self.root_pos(), base_quat=self.root_quat(), lin_vel_b=self.base_lin_vel_b(),
                    ang_vel_b=self.base_ang_vel_b(), q=self.joint_pos(), qd=self.joint_vel(), tau=self.step_tau.copy(),
                    abs_power=self.step_abs_power.copy(), sat=self.step_sat.copy(), foot_pos=fpos, foot_vel=fvel,
                    contact=touch, foot_fn=fn, key_body_b=None)
