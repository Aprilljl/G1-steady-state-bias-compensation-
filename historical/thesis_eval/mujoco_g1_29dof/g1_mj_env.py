"""Closed-loop MuJoCo environment that reproduces the Isaac Lab (legged_lab, g1_amp) observation/action
interface of the Unitree G1 29-DoF AMP policies.

Reference sources (read-only, see README.md):
  legged_lab/tasks/locomotion/amp/mdp/observations.py       root_local_rot_tan_norm
  legged_lab/tasks/locomotion/deepmimic/mdp/observations.py key_body_pos_b
  legged_lab/tasks/locomotion/amp/amp_env_cfg.py            policy observation group, history 5
  legged_lab/assets/unitree.py                              actuator gains / limits / default pose
  <run>/params/env.yaml                                     ground truth of what was actually trained

Python 3.8 compatible.
"""
import re
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_XML = REPO_ROOT / "resources" / "robots" / "g1_description" / "g1_29dof_rev_1_0.xml"

# Isaac Lab articulation joint order (policy obs / actions use this order). Source: comment block in
# legged_lab/tasks/locomotion/amp/mdp/symmetry/g1.py. Verified against a BFS of the MJCF tree in tests.
LAB_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]
# g1_amp_env_cfg.py KEY_BODY_NAMES (preserve_order=True), also in env.yaml policy.key_body_pos_b
KEY_BODY_NAMES = [
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
    "left_shoulder_roll_link", "right_shoulder_roll_link",
]
ROOT_BODY = "pelvis"
NUM_JOINTS = 29
HISTORY_LEN = 5
ACTION_SCALE = 0.25
# per-frame term layout of the policy group (order in env.yaml observations.policy)
TERM_DIMS = [("base_ang_vel", 3), ("root_local_rot_tan_norm", 6), ("velocity_commands", 3),
             ("joint_pos", 29), ("joint_vel", 29), ("last_action", 29), ("key_body_pos_b", 18)]
FRAME_DIM = sum(d for _, d in TERM_DIMS)  # 117
OBS_DIM = FRAME_DIM * HISTORY_LEN  # 585


# --------------------------------------------------------------------------------------------------
# env.yaml parsing (Isaac Lab dumps python tags; ignore them)
# --------------------------------------------------------------------------------------------------
_OBJ = {"joint": mujoco.mjtObj.mjOBJ_JOINT, "body": mujoco.mjtObj.mjOBJ_BODY, "geom": mujoco.mjtObj.mjOBJ_GEOM}


def _id(model, kind, name):
    i = mujoco.mj_name2id(model, _OBJ[kind], name)
    if i < 0:
        raise KeyError("%s '%s' not found in MJCF" % (kind, name))
    return i


class _Loader(yaml.SafeLoader):
    pass


def _py_tag(loader, suffix, node):
    if isinstance(node, yaml.SequenceNode):
        return tuple(loader.construct_sequence(node, deep=True))
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return loader.construct_scalar(node)


_Loader.add_multi_constructor("tag:yaml.org,2002:python/", _py_tag)


def load_env_yaml(path):
    with open(str(path), "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=_Loader)


def _resolve(names, spec, what):
    """Isaac Lab resolve_matching_names_values semantics: scalar -> all, dict -> regex fullmatch.
    Raises if a joint matches several patterns (ambiguity) or none."""
    if not isinstance(spec, dict):
        return {n: float(spec) for n in names}
    out = {}
    for n in names:
        hits = [p for p in spec if re.fullmatch(p, n)]
        if len(hits) != 1:
            raise ValueError("%s: joint %s matched patterns %s" % (what, n, hits))
        out[n] = float(spec[hits[0]])
    return out


def robot_params_from_env_yaml(env_cfg):
    """Return dict of arrays in Lab joint order: default_pos, kp, kd, effort, vel_limit, armature."""
    robot = env_cfg["scene"]["robot"]
    names = LAB_JOINT_NAMES
    jp = robot["init_state"]["joint_pos"]
    default = {}
    for n in names:
        hits = [p for p in jp if re.fullmatch(p, n)]
        if len(hits) > 1:
            raise ValueError("default_pos ambiguity for %s: %s" % (n, hits))
        default[n] = float(jp[hits[0]]) if hits else 0.0
    kp, kd, eff, vel, arm, owner = {}, {}, {}, {}, {}, {}
    for gname, act in robot["actuators"].items():
        gj = [n for n in names if any(re.fullmatch(p, n) for p in act["joint_names_expr"])]
        for n in gj:
            if n in owner:
                raise ValueError("joint %s in two actuator groups (%s, %s)" % (n, owner[n], gname))
            owner[n] = gname
        kp.update(_resolve(gj, act["stiffness"], gname + ".stiffness"))
        kd.update(_resolve(gj, act["damping"], gname + ".damping"))
        eff.update(_resolve(gj, act["effort_limit_sim"], gname + ".effort"))
        vel.update(_resolve(gj, act["velocity_limit_sim"], gname + ".vel"))
        arm.update(_resolve(gj, act["armature"], gname + ".armature"))
    missing = [n for n in names if n not in owner]
    if missing:
        raise ValueError("joints without actuator group: %s" % missing)

    def arr(d):
        return np.array([d[n] for n in names], dtype=np.float64)
    return dict(default_pos=arr(default), kp=arr(kp), kd=arr(kd), effort=arr(eff), vel_limit=arr(vel),
                armature=arr(arm), actuator_group=[owner[n] for n in names])


# --------------------------------------------------------------------------------------------------
# math helpers (numpy, quaternions are (w, x, y, z) as in Isaac Lab and MuJoCo qpos)
# --------------------------------------------------------------------------------------------------
def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def root_local_rot_tan_norm(q):
    """Isaac Lab: q_local = conj(yaw_quat(q)) * q ; R = matrix_from_quat(q_local) ; [R[:,0], R[:,2]].
    conj(yaw)*q as a rotation matrix is Rz(yaw)^T R."""
    R = quat_to_mat(q)
    w, x, y, z = q
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    Rl = Rz.T @ R
    return np.concatenate([Rl[:, 0], Rl[:, 2]])


def roll_pitch_yaw(q):
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


# --------------------------------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------------------------------
class FootMixin(object):
    """Foot contact / state readout shared by the 29-DoF and 12-DoF environments. Needs: model, data, floor_gid,
    geom_side (foot geom id -> 0 left / 1 right), foot_body_ids (left, right ankle_roll_link)."""

    def foot_info(self):
        """Return (touch (2,) bool, fn (2,) float) = any floor contact and summed floor normal force [N] of the
        left/right ankle_roll_link geoms (mj_contactForce, force along the contact normal)."""
        d, m = self.data, self.model
        touch = np.zeros(2, dtype=bool)
        fn = np.zeros(2)
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = int(c.geom1), int(c.geom2)
            if a != self.floor_gid and b != self.floor_gid:
                continue
            side = self.geom_side.get(b if a == self.floor_gid else a)
            if side is None:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            touch[side] = True
            fn[side] += f6[0]
        return touch, fn

    def foot_state(self):
        """World position (2,3) of the ankle_roll_link frame origins and their world linear velocity (2,3)."""
        d, m = self.data, self.model
        pos = d.xpos[self.foot_body_ids].copy()
        vel = np.zeros((2, 3))
        res = np.zeros(6)
        for k, bid in enumerate(self.foot_body_ids):
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, int(bid), res, 0)  # world axes, at body COM
            w, v_com = res[0:3], res[3:6]
            vel[k] = v_com + np.cross(w, pos[k] - d.xipos[bid])  # shift to the frame origin
        return pos, vel


class G1MujocoEnv(FootMixin):
    """MuJoCo twin of the Isaac Lab G1 AMP play environment (flat ground, no noise, no push).

    pd_mode='explicit': tau = kp (q_target - q) - kd qdot computed in python before every MuJoCo step,
                        clipped to +-effort, written to the <motor> actuators (default, as specified).
    pd_mode='implicit': same torque law evaluated inside MuJoCo by position-type actuators
                        (gain kp, bias -kp q - kd qdot) with the implicitfast integrator.
    """

    def __init__(self, env_yaml, xml_path=None, dt=0.001, decimation=20, friction=1.0, pd_mode="explicit",
                 cone="pyramidal"):
        self.env_cfg = load_env_yaml(env_yaml)
        sim = self.env_cfg["sim"]
        self.isaac_dt = sim["dt"]
        self.isaac_decimation = self.env_cfg["decimation"]
        self.rp = robot_params_from_env_yaml(self.env_cfg)
        self.dt = dt
        self.decimation = decimation
        self.policy_dt = dt * decimation
        self.pd_mode = pd_mode
        self.friction = friction
        self.xml_path = Path(xml_path) if xml_path else DEFAULT_XML

        # Plain MjModel edits only (no MjSpec): identical behaviour on mujoco 3.2.x and 4.x.
        m = mujoco.MjModel.from_xml_path(str(self.xml_path))
        m.opt.timestep = dt
        m.opt.cone = {"pyramidal": int(mujoco.mjtCone.mjCONE_PYRAMIDAL),
                      "elliptic": int(mujoco.mjtCone.mjCONE_ELLIPTIC)}[cone]
        if pd_mode == "implicit":
            m.opt.integrator = int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
        self.mjcf_original = {}
        lab_idx = {n: i for i, n in enumerate(LAB_JOINT_NAMES)}
        for n, i in lab_idx.items():
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
            da = int(m.jnt_dofadr[jid])
            self.mjcf_original[n] = dict(armature=float(m.dof_armature[da]), damping=float(m.dof_damping[da]),
                                         frictionloss=float(m.dof_frictionloss[da]),
                                         actfrcrange=tuple(float(v) for v in m.jnt_actfrcrange[jid]))
            m.dof_armature[da] = float(self.rp["armature"][i])
            e = float(self.rp["effort"][i])
            m.jnt_actfrclimited[jid] = 1
            m.jnt_actfrcrange[jid] = [-e, e]  # replace MJCF limits (ankle 50 / wrist 5.. differ from training)
        if pd_mode == "implicit":
            for a in range(m.nu):
                jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a, 0]))
                i = lab_idx[jname]
                kp, kd, e = self.rp["kp"][i], self.rp["kd"][i], float(self.rp["effort"][i])
                m.actuator_gaintype[a] = int(mujoco.mjtGain.mjGAIN_FIXED)
                m.actuator_biastype[a] = int(mujoco.mjtBias.mjBIAS_AFFINE)
                m.actuator_gainprm[a, 0] = kp
                m.actuator_biasprm[a, 0:3] = [0.0, -kp, -kd]
                m.actuator_forcelimited[a] = 1
                m.actuator_forcerange[a] = [-e, e]
        if friction != 1.0:  # robot geoms and floor share the same sliding friction (max-combine => friction)
            m.geom_friction[:, 0] = friction
        # armature/limits changed after compile: refresh the derived reference-pose constants (dof_invweight0,
        # body_invweight0, acc0 ...) that the contact solver uses, so the model equals a compile with these values
        mujoco.mj_setConst(m, mujoco.MjData(m))
        self.model = m
        self.data = mujoco.MjData(self.model)
        m = self.model

        # index maps (by NAME): Lab order -> MuJoCo address
        self.qadr = np.array([m.jnt_qposadr[_id(m, 'joint', n)] for n in LAB_JOINT_NAMES])
        self.dadr = np.array([m.jnt_dofadr[_id(m, 'joint', n)] for n in LAB_JOINT_NAMES])
        jid_to_act = {int(m.actuator_trnid[a, 0]): a for a in range(m.nu)}
        self.act_idx = np.array([jid_to_act[_id(m, 'joint', n)] for n in LAB_JOINT_NAMES])
        assert np.allclose(m.actuator_gear[:, 0], 1.0), "motor gear must be 1 for torque control"
        self.root_id = _id(m, 'body', ROOT_BODY)
        self.key_ids = np.array([_id(m, 'body', n) for n in KEY_BODY_NAMES])
        self.floor_gid = _id(m, "geom", "floor")
        self.foot_geoms = {}
        for side in ("left", "right"):
            bid = _id(m, "body", "%s_ankle_roll_link" % side)
            self.foot_geoms[side] = set(int(g) for g in range(m.ngeom) if m.geom_bodyid[g] == bid)
        all_foot = self.foot_geoms["left"] | self.foot_geoms["right"]
        self.nonfoot_geoms = set(range(m.ngeom)) - all_foot - {self.floor_gid}

        self.default_pos = self.rp["default_pos"]
        self.kp, self.kd, self.effort = self.rp["kp"], self.rp["kd"], self.rp["effort"]
        self.cmd = np.zeros(3)
        self.last_action = np.zeros(NUM_JOINTS)
        self.hist = {k: deque(maxlen=HISTORY_LEN) for k, _ in TERM_DIMS}
        self.init_height = 0.8
        self.geom_side = {}  # foot geom id -> 0 (left) / 1 (right), used by foot_info()
        for si, side in enumerate(("left", "right")):
            for g in self.foot_geoms[side]:
                self.geom_side[g] = si
        self.foot_body_ids = np.array([_id(m, "body", "left_ankle_roll_link"), _id(m, "body", "right_ankle_roll_link")])
        self.step_tau = np.zeros(NUM_JOINTS)  # per policy step: mean applied torque over the substeps
        self.step_abs_power = np.zeros(NUM_JOINTS)  # mean |tau * qdot| over the substeps (W)
        self.step_sat = np.zeros(NUM_JOINTS)  # fraction of substeps with |tau| >= 0.99 * effort
        self.sat_hits = 0  # number of (substep, joint) torque-limit clips, diagnostic only
        self.sat_total = 0

    # ---- state accessors ------------------------------------------------------------------------
    def joint_pos(self):
        return self.data.qpos[self.qadr].copy()

    def joint_vel(self):
        return self.data.qvel[self.dadr].copy()

    def root_quat(self):
        return self.data.qpos[3:7].copy()

    def root_pos(self):
        return self.data.qpos[0:3].copy()

    def base_ang_vel_b(self):
        # MuJoCo free joint: qvel[3:6] is the angular velocity expressed in the body (local) frame
        return self.data.qvel[3:6].copy()

    def base_lin_vel_b(self):
        # qvel[0:3] is the world-frame linear velocity of the pelvis frame origin
        return quat_to_mat(self.root_quat()).T @ self.data.qvel[0:3]

    def key_body_pos_b(self):
        R = quat_to_mat(self.root_quat())
        rel = self.data.xpos[self.key_ids] - self.data.xpos[self.root_id]  # (6,3) world frame
        return (rel @ R).reshape(-1)  # R^T (p_body - p_root) per row == quat_apply_inverse

    # ---- observation ----------------------------------------------------------------------------
    def frame_terms(self):
        return {
            "base_ang_vel": self.base_ang_vel_b(),
            "root_local_rot_tan_norm": root_local_rot_tan_norm(self.root_quat()),
            "velocity_commands": self.cmd.copy(),
            "joint_pos": self.joint_pos(),  # absolute (isaaclab mdp.joint_pos), not relative to default
            "joint_vel": self.joint_vel(),
            "last_action": self.last_action.copy(),
            "key_body_pos_b": self.key_body_pos_b(),
        }

    def _push_history(self, first):
        terms = self.frame_terms()
        for k, _ in TERM_DIMS:
            if first:  # Isaac Lab CircularBuffer: first append after reset fills all slots
                self.hist[k].clear()
                for _ in range(HISTORY_LEN):
                    self.hist[k].append(terms[k])
            else:
                self.hist[k].append(terms[k])

    def get_obs(self):
        """Term-major layout [term0(t-4..t), term1(t-4..t), ...], oldest -> newest, 585 dims."""
        return np.concatenate([np.concatenate(list(self.hist[k])) for k, _ in TERM_DIMS]).astype(np.float32)

    # ---- dynamics -------------------------------------------------------------------------------
    def set_command(self, vx, vy, wz):
        """Set the velocity command. If the history is already filled the newest frame is patched, i.e. the
        policy sees the new command immediately (as after a command resample in Isaac Lab)."""
        self.cmd = np.array([vx, vy, wz], dtype=np.float64)
        h = self.hist["velocity_commands"]
        if len(h) > 0:
            h[-1] = self.cmd.copy()

    def _foot_lowest_z(self):
        zs = []
        for g in self.foot_geoms["left"] | self.foot_geoms["right"]:
            if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
                zs.append(self.data.geom_xpos[g][2] - self.model.geom_size[g][0])
        return min(zs)

    def reset(self, cmd=(0.0, 0.0, 0.0), perturb=None):
        """perturb: optional dict(dq=(29,) joint offsets in Lab order [rad], v=(3,) root lin vel [m/s] (world,
        identity heading), w=(3,) root ang vel [rad/s] (body))."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[0:3] = [0.0, 0.0, self.init_height]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[self.qadr] = self.default_pos + (0.0 if perturb is None else np.asarray(perturb["dq"]))
        mujoco.mj_forward(m, d)
        low = self._foot_lowest_z()
        self.init_foot_clearance = low  # <0: penetration, >0: free drop height
        if low < 0.0:
            d.qpos[2] += -low + 1e-3
            mujoco.mj_forward(m, d)
        if perturb is not None:
            d.qvel[0:3] = perturb["v"]
            d.qvel[3:6] = perturb["w"]
            mujoco.mj_forward(m, d)
        self.init_pelvis_z = float(d.qpos[2])
        self.sat_hits = self.sat_total = 0
        for h in self.hist.values():
            h.clear()
        self.set_command(*cmd)
        self.last_action = np.zeros(NUM_JOINTS)
        self._push_history(first=True)
        return self.get_obs()

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        q_target = self.default_pos + ACTION_SCALE * action  # JointPositionAction, use_default_offset
        d, m = self.data, self.model
        implicit = self.pd_mode == "implicit"
        if implicit:
            d.ctrl[self.act_idx] = q_target
        tau_sum = np.zeros(NUM_JOINTS)
        pw_sum = np.zeros(NUM_JOINTS)
        sat_sum = np.zeros(NUM_JOINTS)
        for _ in range(self.decimation):
            q = d.qpos[self.qadr]
            qd = d.qvel[self.dadr]
            raw = self.kp * (q_target - q) - self.kd * qd
            tau = np.clip(raw, -self.effort, self.effort)
            self.sat_hits += int(np.sum(np.abs(raw) > self.effort))
            self.sat_total += NUM_JOINTS
            tau_sum += tau
            pw_sum += np.abs(tau * qd)
            sat_sum += np.abs(tau) >= 0.99 * self.effort
            if not implicit:
                d.ctrl[self.act_idx] = tau
            mujoco.mj_step(m, d)
        n = float(self.decimation)
        self.step_tau, self.step_abs_power, self.step_sat = tau_sum / n, pw_sum / n, sat_sum / n
        self.last_action = action.copy()
        self._push_history(first=False)
        return self.get_obs()

    # ---- diagnostics ----------------------------------------------------------------------------
    def contacts(self):
        """Return (left_foot_on_floor, right_foot_on_floor, nonfoot_on_floor)."""
        d = self.data
        L = R = N = False
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = int(c.geom1), int(c.geom2)
            if a != self.floor_gid and b != self.floor_gid:
                continue
            o = b if a == self.floor_gid else a
            if o in self.foot_geoms["left"]:
                L = True
            elif o in self.foot_geoms["right"]:
                R = True
            elif o in self.nonfoot_geoms:
                N = True
        return L, R, N

    def snapshot(self):
        """State after a policy step, shared by all recorders (arrays are copies)."""
        touch, fn = self.foot_info()
        fpos, fvel = self.foot_state()
        return dict(base_pos=self.root_pos(), base_quat=self.root_quat(), lin_vel_b=self.base_lin_vel_b(),
                    ang_vel_b=self.base_ang_vel_b(), q=self.joint_pos(), qd=self.joint_vel(), tau=self.step_tau.copy(),
                    abs_power=self.step_abs_power.copy(), sat=self.step_sat.copy(), foot_pos=fpos, foot_vel=fvel,
                    contact=touch, foot_fn=fn, key_body_b=self.key_body_pos_b())
