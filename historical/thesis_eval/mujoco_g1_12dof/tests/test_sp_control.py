"""Tests for thesis_eval/mujoco_g1_12dof/sp_control.py. Run: KMP_DUPLICATE_LIB_OK=TRUE python <this file>.
No real checkpoint needed. The plumbing test uses the TorchScript stand-in deploy/pre_train/g1/motion.pt when present
(copied to an ASCII temp path because torch.jit.load fails on non-ASCII Windows paths) and is skipped otherwise."""
import math
import os
import shutil
import sys
import tempfile
import unittest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(PKG))
for p in (PKG, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import ch5_mujoco as C5  # noqa: E402
import sp_control as S  # noqa: E402
from thesis_eval.analysis import sp_design as D  # noqa: E402

CFG = os.path.join(PKG, "configs", "short_paper_control.yaml")
STANDIN = os.path.join(ROOT, "deploy", "pre_train", "g1", "motion.pt")
DESIGN = dict(gamma=0.3, u_min=0.0, u_max=2.2,
              u_max_by_payload={"0": 2.2, "3": 2.2, "5": 2.2, "8": 2.0, "10": 2.0, "15": None})


class LinearPolicy(object):
    """Deterministic stateless stand-in: a fixed small linear map of the observation."""

    def __init__(self):
        self.W = np.random.RandomState(3).randn(12, 47) * 0.05

    def reset(self):
        pass

    def act(self, obs):
        return self.W.dot(np.asarray(obs, dtype=np.float64)), {}


def nominal(**kw):
    e = dict(payload_kg=0.0, friction=1.0, command_vx=1.0, command_vy=0.0, command_yaw=0.0, push_dir_deg=None,
             push_mag_mps=0.0, push_time_s=None, episode=0)
    e.update(kw)
    return e


class TestOuterLoop(unittest.TestCase):
    def plant_run(self, gk, T, cycles=60, g=0.7, c=1.0, u_max=5.0, est_bias=0.0, u_min=0.0):
        """Discrete first-order plant at the policy rate driven by OuterLoop; returns cycle means and u."""
        a = math.exp(-0.02 / T)
        loop = S.OuterLoop(dict(type="outer", gamma=gk / g, u_min=u_min, u_max=u_max, t_on_s=0.0, est_bias=est_bias))
        u = loop.reset(c)
        x, n, ys, us = 0.0, 0, [], []
        for _ in range(cycles * 40):
            x = a * x + (1 - a) * g * u
            n += 1
            new = loop.observe(x, n, n * 0.02)
            if new is not None:
                u = new
                ys.append(loop.log[-1][1])
                us.append(u)
        loop.x_final = x
        return np.array(ys), np.array(us), loop

    def test_converges_to_reference(self):
        ys, us, _ = self.plant_run(gk=0.3, T=0.5)
        self.assertAlmostEqual(ys[-1], 1.0, places=4)
        self.assertAlmostEqual(us[-1], 1.0 / 0.7, places=3)

    def test_decay_matches_lifted_design_model(self):
        gk, T = 0.6, 0.5
        ys, _, _ = self.plant_run(gk=gk, T=T, cycles=40)
        e = np.abs(ys - 1.0)
        k = np.arange(5, 25)
        rho_emp = math.exp(np.polyfit(k, np.log(e[k]), 1)[0])
        self.assertAlmostEqual(rho_emp, D.spectral_radius(gk, T, 0.0), delta=0.02)

    def test_unstable_above_bound(self):
        T = 0.5
        bound = D.design_gain(T, 0.0)["gk_upper"]
        ys, _, _ = self.plant_run(gk=1.5 * bound, T=T, cycles=40, u_min=-1e9, u_max=1e9)  # |pole| about 1.18
        e = np.abs(ys - 1.0)
        self.assertGreater(e[-1], 100.0 * e[5])
        ys, us, _ = self.plant_run(gk=1.5 * bound, T=T, cycles=60)  # clamped: bounded saturating oscillation
        self.assertTrue(np.all((us >= 0.0) & (us <= 5.0)))
        self.assertGreater(np.abs(ys[-10:] - 1.0).max(), 0.1)
        ys, _, _ = self.plant_run(gk=0.9 * bound, T=T, cycles=200)
        self.assertLess(abs(ys[-1] - 1.0), 1e-2)

    def test_anti_windup_and_reference_change(self):
        _, us, loop = self.plant_run(gk=0.3, T=0.5, c=2.0, u_max=2.2)  # g(u_max) = 1.54 < 2
        self.assertAlmostEqual(us[-1], 2.2, places=9)
        self.assertAlmostEqual(loop.z, 0.2, places=9)
        loop.set_reference(1.0)  # z clamps to the new bounds immediately
        self.assertLessEqual(loop.u, 2.2 + 1e-12)

    def test_estimate_bias_passes_to_steady_state(self):
        ys, _, loop = self.plant_run(gk=0.3, T=0.5, est_bias=0.05, cycles=80)
        self.assertAlmostEqual(ys[-1], 1.0, places=4)  # the measured mean is regulated to c
        self.assertAlmostEqual(loop.x_final, 0.95, places=4)  # so the true speed settles at c - bias

    def test_none_never_changes_u_and_discards_partial_cycle(self):
        loop = S.OuterLoop(dict(type="none"))
        loop.reset(1.0)
        out = [loop.observe(1.0, n, n * 0.02) for n in range(2, 121)]
        self.assertTrue(all(o is None for o in out))
        self.assertTrue(math.isnan(loop.log[0][1]))  # steps 2..40 = 39 samples
        self.assertEqual(len(loop.log), 3)


class TestConfig(unittest.TestCase):
    def test_expands_and_counts(self):
        cfg = S.load_config(CFG)
        exps = dict((e["name"], e) for e in S.expand(cfg, design=DESIGN))
        counts = dict((k, len(v["conds"])) for k, v in exps.items())
        self.assertEqual(counts, {"SP0_smoke_control": 2, "SP-B2_step_id": 15, "SP-B3_compensation": 48,
                                  "SP-B4_gain_sweep": 9, "SP-B5_payload_step": 4, "SP-B6_estimator": 15,
                                  "SP-B7_cap": 12})
        b2 = [c for _, c, _ in exps["SP-B2_step_id"]["conds"]]
        self.assertTrue(all(abs(abs(c["command_step"][1] - c["command_vx"]) - 0.25) < 1e-9 for c in b2))
        b4 = [ct["gamma"] for _, _, ct in exps["SP-B4_gain_sweep"]["conds"]]
        self.assertAlmostEqual(b4[2], 0.3)
        self.assertAlmostEqual(b4[-1], 3.6)

    def test_per_payload_cap_lookup(self):
        cfg = S.load_config(CFG)
        exps = dict((e["name"], e) for e in S.expand(cfg, design=DESIGN))
        caps = dict((cid, ct["u_max"]) for cid, _, ct in exps["SP-B3_compensation"]["conds"] if ct["type"] == "outer")
        self.assertEqual(caps["payload_kg=0.0|command_vx=1.0|controller=outer"], 2.2)
        self.assertEqual(caps["payload_kg=10.0|command_vx=1.0|controller=outer"], 2.0)
        self.assertEqual(caps["payload_kg=15.0|command_vx=1.5|controller=outer"], 1.5)  # None -> u_max = reference
        b5 = dict((cid, ct["u_max"]) for cid, _, ct in exps["SP-B5_payload_step"]["conds"] if ct["type"] == "outer")
        self.assertEqual(b5["payload_step=20:10|controller=outer"], 2.0)  # heaviest payload of the episode
        nocap = [ct["u_max"] for _, _, ct in exps["SP-B7_cap"]["conds"] if ct["u_max"] == 3.5]
        self.assertEqual(len(nocap), 6)

    def test_design_required_only_for_stage_s3(self):
        cfg = S.load_config(CFG)
        self.assertFalse(S.needs_design(cfg, ["SP0", "SP-B2"]))
        self.assertTrue(S.needs_design(cfg, ["SP-B3"]))


@unittest.skipUnless(os.environ.get("RUN_MUJOCO_SMOKE") == "1", "set RUN_MUJOCO_SMOKE=1")
class TestEpisode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = C5.Ch5Sim()

    def test_identical_to_ch5_without_control(self):
        pol = LinearPolicy()
        for payload in (0.0, 5.0):
            ref, _ = C5.run_episode(self.sim, pol, nominal(payload_kg=payload), 0, 0, 3.0, 1.0)
            cond = dict(S.CONDITION_DEFAULTS, payload_kg=payload)
            ctrl = S.resolve_controller(dict(controllers={"none": {"type": "none"}}), cond, None)
            res, _, _, _ = S.run_control_episode(self.sim, pol, cond, ctrl, 0, 0, 3.0, 1.0, 2.0)
            for k in C5.recorder_keys():
                a, b = np.asarray(ref[k]), np.asarray(res[k])
                if a.dtype.kind == "f":
                    self.assertTrue(np.array_equal(a, b, equal_nan=True), k)
                else:
                    self.assertTrue(np.array_equal(a, b), k)

    def test_payload_step_changes_mass_at_scheduled_step(self):
        calls = []
        orig = self.sim.apply_condition

        def spy(payload, friction, ground_friction=1.0):
            calls.append((payload, self.sim.env.n_steps))
            return orig(payload, friction, ground_friction)

        self.sim.apply_condition = spy
        try:
            cond = dict(S.CONDITION_DEFAULTS, payload_step=[0.5, 7.0])
            ctrl = dict(S.CONTROLLER_DEFAULTS, est_bias=0.0, est_noise_std=0.0)
            S.run_control_episode(self.sim, LinearPolicy(), cond, ctrl, 0, 0, 2.0, 1.0, 1.0)
        finally:
            del self.sim.apply_condition
        self.assertEqual(calls[1][0], 7.0)
        self.assertEqual(calls[1][1], 1 + 24)  # zero-action step + 24 recorded steps before policy step 25
        self.sim.apply_condition(0.0, 1.0)


@unittest.skipUnless(os.environ.get("RUN_MUJOCO_SMOKE") == "1", "set RUN_MUJOCO_SMOKE=1")
@unittest.skipUnless(os.path.isfile(STANDIN), "stand-in policy not present")
class TestPlumbingWithWalkingPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        dst = os.path.join(cls.tmp, "standin.pt")
        shutil.copy(STANDIN, dst)
        from policies12 import load_policy
        cls.base = load_policy(dict(kind="jit", path=dst))
        cls.sim = C5.Ch5Sim()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_policy_sees_u_recorder_gets_reference(self):
        seen = []
        base = self.base

        class Spy(object):
            def reset(self):
                base.reset()

            def act(self, obs):
                seen.append(float(obs[6]) / 2.0)  # obs[6] = commands[0] * 2
                return base.act(obs)

        cond = dict(S.CONDITION_DEFAULTS, command_vx=1.0, payload_kg=10.0)
        ctrl = dict(S.CONTROLLER_DEFAULTS, type="outer", gamma=0.3, u_max=2.2, t_on_s=2.0, est_bias=-0.2,
                    est_noise_std=0.0)
        res, res_s, ex, ts = S.run_control_episode(self.sim, Spy(), cond, ctrl, 0, 0, 10.0, 2.0, 6.0,
                                                   want_timeseries=True)
        self.assertTrue(bool(res["survived"][0]))
        self.assertGreater(ex["n_updates"], 5)
        self.assertGreater(ex["u_final"], 1.05)
        self.assertTrue(np.all(ts["c"] == 1.0))
        # the policy acted on u: the command in the observation of step k is the u in force during step k
        n = len(ts["u"])
        self.assertTrue(np.allclose(np.array(seen[:n]), ts["u"], atol=1e-6))
        self.assertGreater(len(set(np.round(ts["u"], 9))), 5)
        self.assertLess(res_s["window_s"][0], res["window_s"][0])
        ctrl0 = dict(ctrl, type="none")
        res0, _, _, _ = S.run_control_episode(self.sim, base, cond, ctrl0, 0, 0, 10.0, 2.0, 6.0)
        self.assertGreater(res["mean_vx"][0], res0["mean_vx"][0])

    def test_main_smoke_writes_outputs(self):
        out = os.path.join(self.tmp, "out")
        dst = os.path.join(self.tmp, "standin.pt")
        rc = S.main(["--config", CFG, "--local_jit", dst, "--out_root", out, "--only", "SP0", "--workers", "1",
                     "--episodes", "1"])
        self.assertEqual(rc, 0)
        exp = os.path.join(out, "_jit_standin", "SP0_smoke_control")
        with open(os.path.join(exp, "episodes_all.csv"), encoding="utf-8") as f:
            rows = f.read().strip().splitlines()
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(os.listdir(os.path.join(exp, "timeseries"))), 2)
        self.assertEqual(S.main(["--config", CFG, "--local_jit", dst, "--out_root", out, "--only", "SP0",
                                 "--workers", "1", "--episodes", "1"]), 0)  # skipped on the second call


if __name__ == "__main__":
    unittest.main()
