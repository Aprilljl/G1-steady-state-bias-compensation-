"""Streaming per-episode metrics for batched legged-robot evaluation.

Pure torch; no isaacgym import, so it can be unit-tested anywhere.
Every env runs exactly one episode. Metrics accumulate over the steady-state
window t >= warmup_s while the env is still alive, and freeze at the first
termination or time-out.
"""
import math

import torch


FAILURE_TYPES = [
    "none",
    "fall_forward",
    "fall_backward",
    "fall_lateral",
    "foot_slip",
    "obstacle_collision",
    "base_height_violation",
    "out_of_terrain",
    "timeout_other",
]


class RecorderParams:
    """Thresholds frozen before evaluation (stored in the run metadata)."""

    def __init__(self, warmup_s=2.0, contact_force_n=1.0, base_height_target=0.78,
                 rec_eps_v=0.25, rec_eps_tilt=0.1, rec_hold_s=0.5, rec_ema_tau_s=0.2,
                 slip_ema_tau_s=0.25, slip_fail_mps=0.3, collision_window_s=0.5,
                 saturation_ratio=0.99, cot_min_distance_m=1.0, heading_min_cmd=0.1):
        self.warmup_s = warmup_s
        self.contact_force_n = contact_force_n
        self.base_height_target = base_height_target
        self.rec_eps_v = rec_eps_v
        self.rec_eps_tilt = rec_eps_tilt
        self.rec_hold_s = rec_hold_s
        self.rec_ema_tau_s = rec_ema_tau_s
        self.slip_ema_tau_s = slip_ema_tau_s
        self.slip_fail_mps = slip_fail_mps
        self.collision_window_s = collision_window_s
        self.saturation_ratio = saturation_ratio
        self.cot_min_distance_m = cot_min_distance_m
        self.heading_min_cmd = heading_min_cmd

    def as_dict(self):
        return dict(self.__dict__)


def classify_failure(pitch, roll, slip_ema, steps_since_collision, abnormal, out_of_terrain,
                     params, dt):
    """Rule-based failure type for terminated envs; first matching rule wins.

    pitch > 0 is taken as forward lean (nose down) following the Isaac Gym
    get_euler_xyz convention; the raw angles are stored too, so the label can be
    re-derived if the sign convention turns out different.
    Returns a long tensor of indices into FAILURE_TYPES.
    """
    n = pitch.shape[0]
    out = torch.full((n,), FAILURE_TYPES.index("timeout_other"), dtype=torch.long, device=pitch.device)
    undecided = torch.ones(n, dtype=torch.bool, device=pitch.device)

    def assign(mask, name):
        nonlocal undecided
        m = mask & undecided
        out[m] = FAILURE_TYPES.index(name)
        undecided = undecided & ~m

    assign(out_of_terrain, "out_of_terrain")
    assign(abnormal, "base_height_violation")
    assign(slip_ema > params.slip_fail_mps, "foot_slip")
    assign(steps_since_collision.float() * dt <= params.collision_window_s, "obstacle_collision")
    pitch_dominant = pitch.abs() >= roll.abs()
    assign(pitch_dominant & (pitch > 0), "fall_forward")
    assign(pitch_dominant & (pitch <= 0), "fall_backward")
    assign(~pitch_dominant, "fall_lateral")
    return out


class EpisodeRecorder:
    """Accumulates one episode per env.

    Call order per policy step: `on_substep` once per physics substep (inside
    env.step), then `on_step` once after termination checks and before reset.
    """

    def __init__(self, num_envs, dt, sim_dt, params, total_mass, torque_limits,
                 push_step=None, device="cpu"):
        self.n = num_envs
        self.dt = dt
        self.sim_dt = sim_dt
        self.p = params
        self.device = device
        self.warmup_steps = int(round(params.warmup_s / dt))
        self.total_mass = total_mass.to(device).float()
        self.torque_limits = torque_limits.to(device).float()
        self.push_step = (push_step if push_step is not None
                          else torch.full((num_envs,), -1, dtype=torch.long)).to(device)
        self.k = 0  # policy steps since reset

        z = lambda: torch.zeros(num_envs, dtype=torch.float, device=device)
        self.active = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.survived = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.end_step = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.failure = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.n_win = z()
        self.sq_v, self.sq_vy, self.sq_yaw, self.sum_vx, self.sum_vy = z(), z(), z(), z(), z()
        self.sum_heading, self.n_heading = z(), z()
        self.sq_tilt, self.max_tilt, self.sq_h, self.sq_vz = z(), z(), z(), z()
        self.slip_num, self.slip_den = z(), z()
        self.contact_l, self.contact_r = z(), z()
        self.impact, self.peak_fz = z(), z()
        self.e_abs, self.e_pos = z(), z()
        self.sat_count, self.sub_count = z(), z()
        self.act_rate, self.n_act = z(), z()
        self.path = z()
        self.term_pitch, self.term_roll, self.term_height = z(), z(), z()
        self.slip_ema = z()
        self.steps_since_collision = torch.full((num_envs,), 10 ** 6, dtype=torch.long, device=device)
        self.err_ema = z()
        self.ok_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.rec_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.post_push_peak_tilt = z()
        self.post_push_peak_err = z()

        self.start_xy = None
        self.end_xy = None
        self.prev_xy = None
        self.prev_action = None
        self.prev_force = None

    # -- physics substep -------------------------------------------------
    def on_substep(self, torques, dof_vel):
        # The substep belongs to the policy step that is about to finish (k + 1).
        in_win = self.active & (self.k + 1 >= self.warmup_steps)
        w = in_win.float()
        power = torques * dof_vel
        self.e_abs += w * power.abs().sum(dim=1) * self.sim_dt
        self.e_pos += w * power.clamp(min=0.0).sum(dim=1) * self.sim_dt
        sat = (torques.abs() >= self.p.saturation_ratio * self.torque_limits).float().sum(dim=1)
        self.sat_count += w * sat
        self.sub_count += w * torques.shape[1]

    # -- policy step ------------------------------------------------------
    def on_step(self, base_pos, base_lin_vel, base_ang_vel, rpy, rel_height, commands, actions,
                feet_vel, feet_force, collision, reset, time_out, abnormal, out_of_terrain):
        """All inputs are (N, ...) tensors of the post-physics, pre-reset state."""
        self.k += 1
        k = self.k
        if self.start_xy is None:
            self.start_xy = base_pos[:, :2].clone()
            self.end_xy = base_pos[:, :2].clone()
            self.prev_xy = base_pos[:, :2].clone()
            self.prev_action = actions.clone()
            self.prev_force = feet_force.clone()

        act = self.active
        in_win = act & (k >= self.warmup_steps)
        w = in_win.float()

        v_err = base_lin_vel[:, :2] - commands[:, :2]
        v_err_norm = v_err.norm(dim=1)
        self.n_win += w
        self.sq_v += w * v_err_norm ** 2
        self.sq_vy += w * v_err[:, 1] ** 2
        self.sq_yaw += w * (base_ang_vel[:, 2] - commands[:, 2]) ** 2
        self.sum_vx += w * base_lin_vel[:, 0]
        self.sum_vy += w * base_lin_vel[:, 1]

        cmd_norm = commands[:, :2].norm(dim=1)
        vel_norm = base_lin_vel[:, :2].norm(dim=1)
        hmask = w * (cmd_norm > self.p.heading_min_cmd).float() * (vel_norm > 1e-3).float()
        cross = commands[:, 0] * base_lin_vel[:, 1] - commands[:, 1] * base_lin_vel[:, 0]
        dot = (commands[:, :2] * base_lin_vel[:, :2]).sum(dim=1)
        self.sum_heading += hmask * torch.atan2(cross, dot).abs()
        self.n_heading += hmask

        tilt = torch.sqrt(rpy[:, 0] ** 2 + rpy[:, 1] ** 2)
        self.sq_tilt += w * tilt ** 2
        self.max_tilt = torch.where(in_win, torch.maximum(self.max_tilt, tilt), self.max_tilt)
        self.sq_h += w * (rel_height - self.p.base_height_target) ** 2
        self.sq_vz += w * base_lin_vel[:, 2] ** 2

        # feet: index 0 = left, 1 = right (order of feet_indices in the asset)
        contact = feet_force[:, :, 2] > self.p.contact_force_n
        cf = contact.float()
        foot_speed = feet_vel[:, :, :2].norm(dim=2)
        self.slip_num += w * (cf * foot_speed).sum(dim=1)
        self.slip_den += w * cf.sum(dim=1)
        self.contact_l += w * cf[:, 0]
        self.contact_r += w * cf[:, 1]
        df = (feet_force - self.prev_force).norm(dim=2) ** 2
        self.impact += w * df.sum(dim=1)
        self.peak_fz = torch.where(in_win, torch.maximum(self.peak_fz, feet_force[:, :, 2].max(dim=1).values),
                                   self.peak_fz)

        step_act = w * float(k > self.warmup_steps)
        self.act_rate += step_act * ((actions - self.prev_action) ** 2).sum(dim=1)
        self.n_act += step_act
        self.path += w * (base_pos[:, :2] - self.prev_xy).norm(dim=1)

        # slip EMA over stance feet only (used by the failure rule)
        stance_slip = (cf * foot_speed).sum(dim=1) / cf.sum(dim=1).clamp(min=1.0)
        a_slip = self.dt / (self.p.slip_ema_tau_s + self.dt)
        self.slip_ema = torch.where(act, self.slip_ema + a_slip * (stance_slip - self.slip_ema), self.slip_ema)
        self.steps_since_collision = torch.where(act & collision, torch.zeros_like(self.steps_since_collision),
                                                 self.steps_since_collision + 1)

        # push recovery: EMA of the speed error filters the gait oscillation
        a_err = self.dt / (self.p.rec_ema_tau_s + self.dt)
        self.err_ema = self.err_ema + a_err * (v_err_norm - self.err_ema)
        after_push = act & (self.push_step >= 0) & (k > self.push_step)
        self.post_push_peak_tilt = torch.where(after_push, torch.maximum(self.post_push_peak_tilt, tilt),
                                               self.post_push_peak_tilt)
        self.post_push_peak_err = torch.where(after_push, torch.maximum(self.post_push_peak_err, self.err_ema),
                                              self.post_push_peak_err)
        ok = (self.err_ema < self.p.rec_eps_v) & (tilt < self.p.rec_eps_tilt)
        pending = after_push & (self.rec_step < 0)
        self.ok_count = torch.where(pending & ok, self.ok_count + 1, torch.zeros_like(self.ok_count))
        hold = max(1, int(round(self.p.rec_hold_s / self.dt)))
        just_rec = pending & (self.ok_count >= hold)
        self.rec_step = torch.where(just_rec, torch.full_like(self.rec_step, k) - hold + 1, self.rec_step)

        # termination bookkeeping (state is pre-reset)
        done = act & reset
        if torch.any(done):
            fail = done & ~time_out
            self.survived |= done & time_out
            self.end_step = torch.where(done, torch.full_like(self.end_step, k), self.end_step)
            self.term_pitch = torch.where(done, rpy[:, 1], self.term_pitch)
            self.term_roll = torch.where(done, rpy[:, 0], self.term_roll)
            self.term_height = torch.where(done, rel_height, self.term_height)
            ftype = classify_failure(rpy[:, 1], rpy[:, 0], self.slip_ema, self.steps_since_collision,
                                     abnormal, out_of_terrain, self.p, self.dt)
            self.failure = torch.where(fail, ftype, self.failure)
            self.active = act & ~done

        self.prev_xy = base_pos[:, :2].clone()
        self.prev_action = actions.clone()
        self.prev_force = feet_force.clone()
        self.end_xy = torch.where(act.unsqueeze(1), base_pos[:, :2], self.end_xy)

    def all_done(self):
        return not bool(torch.any(self.active))

    def finalize(self):
        """Returns a dict of 1-D numpy arrays, one entry per env."""
        if self.start_xy is None:  # no step was recorded
            self.start_xy = torch.zeros(self.n, 2, device=self.device)
            self.end_xy = self.start_xy.clone()
        # envs still active when the loop ends ran the full horizon
        still = self.active.clone()
        self.survived |= still
        self.end_step = torch.where(still, torch.full_like(self.end_step, self.k), self.end_step)

        nan = float("nan")
        nw = self.n_win.clamp(min=1.0)
        valid = self.n_win > 0

        def masked(x):
            return torch.where(valid, x, torch.full_like(x, nan))

        t_win = self.n_win * self.dt
        g = 9.81
        cot_ok = self.path >= self.p.cot_min_distance_m
        denom = self.total_mass * g * self.path.clamp(min=1e-6)
        dl = self.contact_l / nw
        dr = self.contact_r / nw
        rec_time = torch.where(self.rec_step >= 0, (self.rec_step - self.push_step).float() * self.dt,
                               torch.full_like(self.n_win, nan))
        has_push = self.push_step >= 0
        survival_time = self.end_step.float() * self.dt
        out = {
            "survived": self.survived,
            "survival_time_s": survival_time,
            "failure_time_s": torch.where(self.survived, torch.full_like(survival_time, nan), survival_time),
            "failure_type_id": torch.where(self.survived, torch.zeros_like(self.failure), self.failure),
            "window_s": t_win,
            "distance_m": (self.end_xy - self.start_xy).norm(dim=1),
            "path_window_m": self.path,
            "velocity_rmse": masked(torch.sqrt(self.sq_v / nw)),
            "lateral_velocity_rmse": masked(torch.sqrt(self.sq_vy / nw)),
            "yaw_rmse": masked(torch.sqrt(self.sq_yaw / nw)),
            "mean_vx": masked(self.sum_vx / nw),
            "mean_vy": masked(self.sum_vy / nw),
            "heading_error_rad": torch.where(self.n_heading > 0, self.sum_heading / self.n_heading.clamp(min=1.0),
                                             torch.full_like(self.n_win, nan)),
            "orientation_rms_rad": masked(torch.sqrt(self.sq_tilt / nw)),
            "orientation_peak_rad": masked(self.max_tilt),
            "height_rmse_m": masked(torch.sqrt(self.sq_h / nw)),
            "vertical_velocity_rms": masked(torch.sqrt(self.sq_vz / nw)),
            "foot_slip_mps": torch.where(self.slip_den > 0, self.slip_num / self.slip_den.clamp(min=1.0),
                                         torch.full_like(self.n_win, nan)),
            "duty_factor_left": masked(dl),
            "duty_factor_right": masked(dr),
            "duty_factor_error": masked((dl - dr).abs()),
            "si_duty_pct": torch.where(valid & (dl + dr > 0), 200.0 * (dl - dr).abs() / (dl + dr).clamp(min=1e-6),
                                       torch.full_like(self.n_win, nan)),
            "ground_impact": masked(self.impact / nw),
            "peak_grf_bw": masked(self.peak_fz / (self.total_mass * g)),
            "mechanical_energy_j": masked(self.e_abs),
            "positive_energy_j": masked(self.e_pos),
            "mean_abs_power_w": torch.where(t_win > 0, self.e_abs / t_win.clamp(min=1e-6),
                                            torch.full_like(self.n_win, nan)),
            "cot": torch.where(valid & cot_ok, self.e_abs / denom, torch.full_like(self.n_win, nan)),
            "cot_positive": torch.where(valid & cot_ok, self.e_pos / denom, torch.full_like(self.n_win, nan)),
            "action_rate": torch.where(self.n_act > 0, self.act_rate / self.n_act.clamp(min=1.0),
                                       torch.full_like(self.n_win, nan)),
            "torque_saturation_rate": torch.where(self.sub_count > 0, self.sat_count / self.sub_count.clamp(min=1.0),
                                                  torch.full_like(self.n_win, nan)),
            "recovery_time_s": rec_time,
            "post_push_peak_tilt_rad": torch.where(has_push, self.post_push_peak_tilt, torch.full_like(self.n_win, nan)),
            "post_push_peak_verr": torch.where(has_push, self.post_push_peak_err, torch.full_like(self.n_win, nan)),
            "term_pitch_rad": torch.where(self.survived, torch.full_like(self.n_win, nan), self.term_pitch),
            "term_roll_rad": torch.where(self.survived, torch.full_like(self.n_win, nan), self.term_roll),
            "term_rel_height_m": torch.where(self.survived, torch.full_like(self.n_win, nan), self.term_height),
            "slip_ema_at_end": self.slip_ema,
            "total_mass_kg": self.total_mass,
        }
        res = {k: v.detach().cpu().numpy() for k, v in out.items()}
        res["failure_type"] = [FAILURE_TYPES[i] for i in res.pop("failure_type_id").tolist()]
        return res


def success_from_rmse(velocity_rmse, command_speed, survived, base=0.25, slope=0.2, scale=1.0):
    """Tracking criterion S3 applied offline: e_v <= scale * (base + slope * |v_cmd|)."""
    tau = scale * (base + slope * abs(command_speed))
    ok = (velocity_rmse == velocity_rmse) and velocity_rmse <= tau  # NaN-safe
    return bool(survived) and ok


def wilson_interval(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h
