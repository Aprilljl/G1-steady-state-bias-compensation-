# Historical Isaac Gym offline plan

This is a selected historical overlay and offline audit aid, not a verified Isaac Gym release. See `../../docs/isaacgym_reproduction.md` for scope and blockers. `configs/short_paper_isaacgym.yaml` preserves the registered design values; `scenarios.py` uses the existing historical expander and prints launch specifications without starting a simulator, loading weights, or creating an output directory. `policy_arch.py` checks supplied metadata only; it does not inspect checkpoints and is not automatically enforced by the expander.

The real private evaluator and required environment files are present under `overlay/` at their original relative paths, especially `overlay/legged_gym/scripts/eval_batch.py`, `overlay/legged_gym/envs/base/legged_robot.py`, and `overlay/legged_gym/envs/g1/g1_config.py`. The top-level `eval_batch.py` remains a standalone import-safe offline stub and is deliberately not the runtime evaluator. Source commits and SHA-256 values are recorded in `overlay_manifest.json` and `source_manifest.json`.

From the project root:

```text
python -B -m unittest discover -s tests -p test_isaacgym_release.py -v
python -B historical/isaacgym_release/scenarios.py --help
python -B historical/isaacgym_release/scenarios.py --registry PATH\TO\registry.csv --out-root PATH\TO\planned-output --only SP-A2
```

The overlay has only passed static/hash/config checks. No Isaac Gym execution, GPU access, checkpoint deserialization, or formal experiment was performed. Public upstream G1 LSTM defaults are not the paper's recorded MLP. Apply the overlay only to a fresh, separately verified upstream sandbox as documented; do not modify the existing upstream checkout.
