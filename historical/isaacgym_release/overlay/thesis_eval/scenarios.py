"""Expand a scenario YAML into Isaac Gym launch specs (one JSON per launch).

One launch = one policy checkpoint on one terrain; every env runs exactly one
episode under its own condition. YAML schema:

    defaults:
      episodes: 100
      seed: 0
      episode_length_s: 20.0
      warmup_s: 2.0
      add_noise: false
      max_envs: 4096            # launches above this are chunked by episode ranges
      timeseries_episodes: 1    # episodes < this per condition save a timeseries
      checkpoint: final         # 'final' or an int iteration -> model_<iter>.pt in the same folder
      condition: {payload_kg: 0.0, friction: 1.0, command_vx: 1.0, command_vy: 0.0,
                  command_yaw: 0.0, push_dir_deg: null, push_mag_mps: 0.0, push_time_s: null}
      recorder_params: {}
    terrains:
      plane:     {mesh_type: plane, ground_friction: 1.0}
      rough_d05: {mesh_type: trimesh, difficulty: 0.5,
                  proportions: [0.2, 0.7, 0.0, 0.0, 0.1], ground_friction: 1.0}
    experiments:
      <name>:
        enabled: true
        policies: [<run name>, ...]
        policy_overrides: {}                  # optional
        policy_overrides_by_id: {<run>: {}}   # optional, merged on top
        terrains: [plane, rough_d05]
        condition_defaults: {push_time_s: 5.0}  # optional, merged onto defaults.condition
        grid: {payload_kg: [0.0, 1.5, 3.0]}   # cartesian product (axis `push`: list of [dir_deg, mag])
        conditions:                           # optional explicit list (after grid conditions)
          - {name: push_fwd, push_dir_deg: 0, push_mag_mps: 1.0, push_time_s: 5.0,
             terrain: plane}                  # `terrain` (name or list) restricts the condition
        # any of episodes/seed/episode_length_s/warmup_s/add_noise/max_envs/
        # timeseries_episodes/checkpoint/recorder_params may be overridden here

Environment order inside a launch: conditions in expansion order (grid first),
episodes inner, env_index contiguous from 0.
"""
import itertools
import json
import os

import yaml

CONDITION_KEYS = ("payload_kg", "friction", "command_vx", "command_vy",
                  "command_yaw", "push_dir_deg", "push_mag_mps", "push_time_s")
_FLOAT_KEYS = ("payload_kg", "friction", "command_vx", "command_vy",
               "command_yaw", "push_mag_mps")
_OPT_FLOAT_KEYS = ("push_dir_deg", "push_time_s")
# keys that experiments may override from `defaults`
_OVERRIDABLE = ("episodes", "seed", "episode_length_s", "warmup_s", "add_noise",
                "max_envs", "timeseries_episodes", "checkpoint", "recorder_params")
_BUILTIN_DEFAULTS = {
    "episodes": 100, "seed": 0, "episode_length_s": 20.0, "warmup_s": 2.0,
    "add_noise": False, "max_envs": 4096, "timeseries_episodes": 1,
    "checkpoint": "final", "recorder_params": {},
    "condition": {"payload_kg": 0.0, "friction": 1.0, "command_vx": 1.0,
                  "command_vy": 0.0, "command_yaw": 0.0, "push_dir_deg": None,
                  "push_mag_mps": 0.0, "push_time_s": None},
}


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "experiments" not in cfg or "terrains" not in cfg:
        raise ValueError("config %s must contain 'terrains' and 'experiments'" % path)
    return cfg


def _normalize_condition(cond):
    """Cast to plain floats / None and validate the push triple."""
    out = {}
    for k in CONDITION_KEYS:
        v = cond.get(k)
        if k in _FLOAT_KEYS:
            out[k] = float(v)
        else:
            out[k] = None if v is None else float(v)
    if out["push_mag_mps"] <= 0.0:
        out["push_mag_mps"] = 0.0
        out["push_dir_deg"] = None
        out["push_time_s"] = None
    elif out["push_dir_deg"] is None or out["push_time_s"] is None:
        raise ValueError("push_mag_mps > 0 requires push_dir_deg and push_time_s: %r" % (cond,))
    return out


def _fmt(v):
    return "None" if v is None else repr(float(v))


def _check_keys(d, ctx):
    bad = [k for k in d if k not in CONDITION_KEYS]
    if bad:
        raise ValueError("unknown condition keys %s in %s" % (bad, ctx))


def _grid_conditions(grid, base, exp_name):
    """Cartesian product of grid axes -> list of (condition_id, condition_dict)."""
    axes = []  # each axis: list of dict(key -> value)
    for key, values in grid.items():
        if key == "push":
            axes.append([{"push_dir_deg": p[0], "push_mag_mps": p[1]} for p in values])
        else:
            _check_keys({key: None}, "grid of %s" % exp_name)
            axes.append([{key: v} for v in values])
    result = []
    for combo in itertools.product(*axes):
        over = {}
        for part in combo:
            over.update(part)
        cond = dict(base)
        cond.update(over)
        cid = "|".join("%s=%s" % (k, _fmt(v)) for k, v in over.items())
        result.append((cid, cond))
    return result


def _explicit_conditions(items, base, exp_name):
    """Explicit condition list -> list of (condition_id, condition_dict, terrain_restriction)."""
    result = []
    for item in items:
        item = dict(item)
        name = item.pop("name", None)
        terr = item.pop("terrain", None)
        if isinstance(terr, str):
            terr = [terr]
        _check_keys(item, "conditions of %s" % exp_name)
        cond = dict(base)
        cond.update(item)
        if name is None:
            diff = [(k, cond[k]) for k in CONDITION_KEYS if cond[k] != base[k]]
            name = "|".join("%s=%s" % (k, _fmt(v)) for k, v in diff) or "nominal"
        result.append((name, cond, terr))
    return result


def _experiment_conditions(exp_name, exp, defaults):
    base = dict(defaults["condition"])
    base.update(exp.get("condition_defaults") or {})
    _check_keys(base, "condition defaults of %s" % exp_name)
    entries = []  # (cid, normalized cond, terrain restriction or None)
    if exp.get("grid"):
        for cid, cond in _grid_conditions(exp["grid"], base, exp_name):
            entries.append((cid, _normalize_condition(cond), None))
    for cid, cond, terr in _explicit_conditions(exp.get("conditions") or [], base, exp_name):
        entries.append((cid, _normalize_condition(cond), terr))
    if not entries:
        entries.append(("nominal", _normalize_condition(base), None))
    ids = [e[0] for e in entries]
    dup = sorted(set(i for i in ids if ids.count(i) > 1))
    if dup:
        raise ValueError("duplicate condition ids in %s: %s" % (exp_name, dup))
    return entries


def _select_experiments(config, only):
    names = list(config["experiments"].keys())
    if not only:
        return names
    selected = []
    for o in only:
        hits = [n for n in names if n == o or n.startswith(o)]
        if not hits:
            raise ValueError("--only %r matches no experiment; available: %s" % (o, names))
        selected.extend(n for n in hits if n not in selected)
    return selected


def _resolve_checkpoint(ckpt, row):
    """Return (path, sha256) for 'final' or an integer iteration."""
    final = row["final_checkpoint"]
    if ckpt == "final":
        return final, row.get("final_sha256", "") or ""
    return os.path.join(os.path.dirname(final), "model_%d.pt" % int(ckpt)), ""


def expand(config, registry, out_root, only=None):
    out_root = os.path.abspath(out_root)
    defaults = dict(_BUILTIN_DEFAULTS)
    defaults.update({k: v for k, v in (config.get("defaults") or {}).items() if k != "condition"})
    defaults["condition"] = dict(_BUILTIN_DEFAULTS["condition"])
    defaults["condition"].update((config.get("defaults") or {}).get("condition") or {})
    terrains = config["terrains"]

    specs = []
    missing = []
    for exp_name in _select_experiments(config, only):
        exp = config["experiments"][exp_name]
        if not exp.get("enabled", True):
            continue
        opts = dict(defaults)
        opts.update({k: exp[k] for k in _OVERRIDABLE if k in exp})
        entries = _experiment_conditions(exp_name, exp, defaults)
        exp_dir = os.path.join(out_root, exp_name)

        for policy in exp["policies"]:
            if policy not in registry:
                if policy not in missing:
                    missing.append(policy)
                continue
            overrides = dict(exp.get("policy_overrides") or {})
            overrides.update((exp.get("policy_overrides_by_id") or {}).get(policy) or {})
            ckpt, sha = _resolve_checkpoint(opts["checkpoint"], registry[policy])

            for tname in exp["terrains"]:
                if tname not in terrains:
                    raise KeyError("experiment %s uses undefined terrain %r" % (exp_name, tname))
                t = terrains[tname]
                terrain = {"name": tname, "mesh_type": t["mesh_type"],
                           "difficulty": t.get("difficulty"),
                           "proportions": t.get("proportions"),
                           "ground_friction": float(t.get("ground_friction", 1.0))}
                conds = [(cid, c) for cid, c, r in entries if r is None or tname in r]
                if not conds:
                    continue
                n_ep = int(opts["episodes"])
                ep_per_chunk = max(1, int(opts["max_envs"]) // len(conds))
                n_chunks = (n_ep + ep_per_chunk - 1) // ep_per_chunk
                for chunk in range(n_chunks):
                    ep_lo = chunk * ep_per_chunk
                    ep_hi = min(n_ep, ep_lo + ep_per_chunk)
                    envs = []
                    for cid, c in conds:
                        for ep in range(ep_lo, ep_hi):
                            env = {"env_index": len(envs), "condition_id": cid, "episode": ep}
                            env.update(c)
                            env["save_timeseries"] = ep < int(opts["timeseries_episodes"])
                            envs.append(env)
                    launch_id = "%s__%s__%s__c%d" % (exp_name, policy, tname, chunk)
                    specs.append({
                        "launch_id": launch_id,
                        "experiment": exp_name,
                        "policy_id": policy,
                        "checkpoint": ckpt,
                        "checkpoint_sha256": sha,
                        "policy_overrides": overrides,
                        "terrain": terrain,
                        "seed": int(opts["seed"]) + chunk,
                        "episode_length_s": float(opts["episode_length_s"]),
                        "warmup_s": float(opts["warmup_s"]),
                        "add_noise": bool(opts["add_noise"]),
                        "recorder_params": dict(opts["recorder_params"] or {}),
                        "envs": envs,
                        "out_csv": os.path.join(exp_dir, "episodes", launch_id + ".csv"),
                        "out_meta": os.path.join(exp_dir, "meta", launch_id + ".json"),
                        "out_timeseries_dir": os.path.join(exp_dir, "timeseries", launch_id) + os.sep,
                    })
    if missing:
        raise ValueError("policies not found in registry (%d): %s" % (len(missing), ", ".join(missing)))
    return specs


def write_specs(specs, out_root):
    paths = []
    for spec in specs:
        d = os.path.join(os.path.abspath(out_root), spec["experiment"], "specs")
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, spec["launch_id"] + ".json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=1)
        paths.append(p)
    return paths
