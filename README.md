# g1-steady-state-bias-compensation

Reference implementation for *Steady-State Bias Compensation for Learned Bipedal Locomotion*. This repository contains controller and analysis code, MuJoCo and Isaac Gym evaluation code, F/G/R policy checkpoints, and a qualitative gait illustration.

## Setup

Create a Python environment and install `requirements.txt`. MuJoCo runs require separately obtained [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) G1 resources; set `UNITREE_RL_GYM_ROOT` to that checkout. See [MuJoCo notes](docs/mujoco_reproduction.md), [Isaac Gym notes](docs/isaacgym_reproduction.md) and [provenance](PROVENANCE.md).

## Checkpoints

The F/G/R checkpoints are in `checkpoints/`, and `model_manifest.json` records their identities. Verify hashes without loading them:

```sh
python -B check_model.py F checkpoints/F/model_10000.pt
python -B check_model.py G checkpoints/G/model_10000.pt
python -B check_model.py R checkpoints/R/model_10000.pt
python -B g1_runner.py check --policy F --experiment mujoco-smoke --f checkpoints/F/model_10000.pt
```

A simulation requires explicit trust and confirmation, plus a new output directory outside the repository:

```sh
python -B g1_runner.py run --policy F --experiment mujoco-smoke --f checkpoints/F/model_10000.pt --out ../g1-smoke-output --trust-checkpoint --confirm-run
```

Use `python -B g1_runner.py --help` for available commands.

## Tests

```sh
python -B -m pytest -p no:cacheprovider tests
```

## License and attribution

New code uses the BSD 3-Clause [LICENSE](LICENSE). The Isaac Gym evaluation code in `historical/isaacgym_release/overlay/` is derived from [legged_gym](https://github.com/leggedrobotics/legged_gym) (ETH Zurich, Nikita Rudin, NVIDIA); its license is retained at that directory. The author has authorized public disclosure of the F/G/R checkpoints; reuse terms for original material the author owns and has the right to license are CC BY 4.0 as specified in [DATA_LICENSE.md](DATA_LICENSE.md). Upstream and third-party terms are separate; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
