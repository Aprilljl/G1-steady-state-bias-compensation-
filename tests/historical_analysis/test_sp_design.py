"""Adapted historical SP design tests; all fixtures are synthetic and use temporary directories."""
import contextlib, csv, io, json, math, os, tempfile, unittest
import numpy as np
from historical.thesis_eval.analysis import sp_design as D


def closed_form_bound(T, Ts=0.8):
    a = math.exp(-Ts / T); b = (T / Ts) * (1 - a)
    return min((1 - a) / (b - a), 2 * (1 + a) / (1 + a - 2 * b))


class TestStatistics(unittest.TestCase):
    def test_poisson_interval_known_values(self):
        self.assertEqual(D.poisson_rate_interval(0, 1.0)[0], 0.0)
        self.assertAlmostEqual(D.poisson_rate_interval(0, 1.0)[1], 3.6889, places=3)
        lo, hi = D.poisson_rate_interval(5, 803.4)
        self.assertAlmostEqual(lo, 0.0020, places=4); self.assertAlmostEqual(hi, 0.0145, places=4)

    def test_u_max_rule_is_contiguous_from_below(self):
        rows = [dict(command_vx=c, early=0, hi=h) for c, h in [(1.8,.001),(1.9,.005),(2.0,.02),(2.1,.004)]]
        self.assertEqual(D.u_max_from_hazard(rows, .0124), 1.9)
        rows[0]["early"] = 1; self.assertIsNone(D.u_max_from_hazard(rows, .0124))


class TestCaps(unittest.TestCase):
    def test_payload_caps(self):
        cell = lambda m,u,ok: dict(payload_kg=m,u=u,n=20,n_survived=20 if ok else 19,g=u)
        smap=[cell(0,.25,False),cell(0,.5,True),cell(0,1,True),cell(0,2,True),cell(0,2.25,False),cell(15,.5,True),cell(15,1,False),cell(15,2,True),cell(8,.5,False)]
        self.assertEqual(D.payload_caps(smap,2.1,.5), {0:2.0,8:None,15:.5})

    def test_start_up_failures_ignored_only_with_cap_t0(self):
        rows=[dict(payload_kg="15",command_vx=u,survived="True",mean_vx="0.5",survival_time_s="30") for u in ("0.5","1.0","1.5") for _ in range(5)]
        rows[5]=dict(rows[5],survived="False",survival_time_s="1.78")
        self.assertEqual(D.payload_caps(D.static_map(rows),2,.5), {15.0:.5})
        self.assertEqual(D.payload_caps(D.static_map(rows,cap_t0=2),2,.5), {15.0:1.5})
        rows[6]=dict(rows[6],survived="False",survival_time_s="12.0")
        self.assertEqual(D.payload_caps(D.static_map(rows,cap_t0=2),2,.5), {15.0:.5})


class TestDesign(unittest.TestCase):
    def test_lifted_model_matches_closed_form_without_delay(self):
        for T in (.4,.6,.9):
            self.assertAlmostEqual(D.design_gain(T,0,n=3000)["gk_upper"],closed_form_bound(T),delta=.05*closed_form_bound(T))

    def test_three_state_matrix_matches_direct_simulation(self):
        gk,T,L=.8,.4,.3; d,a=int(round(L/.02)),math.exp(-.02/T); x,buf,u,ys=0.,[1.]*d,1.,[]
        for _ in range(30):
            acc=0.
            for _ in range(40): buf.append(u); x=a*x+(1-a)*buf.pop(0); acc+=x
            u-=gk*acc/40; ys.append(abs(u))
        rho_emp=math.exp(np.polyfit(np.arange(10,30),np.log(ys[10:30]),1)[0])
        self.assertAlmostEqual(rho_emp,D.spectral_radius(gk,T,L),delta=.02)

    def test_robust_design_is_worst_case(self):
        one=D.design_gain(.4,0); both=D.design_gain_robust([(.4,0),(.4,.4)])
        self.assertLess(both["gk_upper"],one["gk_upper"]); self.assertGreaterEqual(both["rho_star"],one["rho_star"])

    def test_delay_shrinks_margin(self): self.assertLess(D.design_gain(.5,.3)["gk_upper"],D.design_gain(.5,0)["gk_upper"])

    def test_fopdt_fit_recovers_parameters_with_ripple(self):
        t=np.arange(0,30,.02); T,L,ts=.55,.12,15.; s=t-ts-L
        v=1+.2*np.where(s>0,1-np.exp(-np.clip(s,0,None)/T),0.)+.06*np.sin(2*np.pi*t/.8)+.01*np.random.RandomState(0).randn(len(t))
        f=D.fopdt_fit(t,v,ts); self.assertAlmostEqual(f["T"],T,delta=.08); self.assertAlmostEqual(f["L"],L,delta=.06); self.assertAlmostEqual(f["dy"],.2,delta=.01)

    def test_end_to_end_on_synthetic_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            haz=os.path.join(tmp,"hazard.csv")
            with open(haz,"w",newline="") as f:
                w=csv.writer(f); w.writerow(["command_vx","survived","survival_time_s"])
                for c,k in [(1.8,0),(1.9,0),(2.,3),(2.1,80),(2.2,100)]:
                    for i in range(100): w.writerow([c,"False" if i<k else "True",10. if i<k else 120.])
            sta=os.path.join(tmp,"static.csv")
            with open(sta,"w",newline="") as f:
                w=csv.writer(f); w.writerow(["payload_kg","command_vx","survived","mean_vx"])
                for m in (0,10):
                    for u in (.5,1,1.5,2,2.25):
                        for _ in range(3): w.writerow([m,u,"True",(.95-.015*m)*u-.05*u*u])
            sdir=os.path.join(tmp,"step","timeseries"); os.makedirs(sdir); t=np.arange(.02,30,.02); s=t-15-.1
            vx=1+.2*np.where(s>0,1-np.exp(-np.clip(s,0,None)/.5),0.)
            np.savez(os.path.join(sdir,"a__ep0.npz"),t=t,vx=vx,condition=np.array(json.dumps(dict(command_step=[15.,1.25],command_vx=1.,payload_kg=0.))))
            out=os.path.join(tmp,"design","sp_design.json")
            with contextlib.redirect_stdout(io.StringIO()): rc=D.main(["--hazard",haz,"--static",sta,"--step_dir",os.path.join(tmp,"step"),"--out",out])
            self.assertEqual(rc,0)
            with open(out,encoding="utf-8") as f: d=json.load(f)
            self.assertEqual(d["u_max"],2.); self.assertEqual(d["u_max_by_payload"],{"0":2.,"10":2.}); self.assertGreater(d["gamma"],0.); self.assertGreater(d["gain_margin"],1.); self.assertAlmostEqual(d["T_s"],.5,delta=.05)
