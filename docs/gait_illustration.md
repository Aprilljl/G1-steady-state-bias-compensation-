# Fig. 9: qualitative gait illustration

`historical/gait_illustration/simulation_gait_sequence.png` is the author-selected English-labelled figure. Its SHA-256 and recorded provenance are in `historical/gait_illustration/provenance.json`. It is a qualitative illustration of a Unitree G1 29-DoF policy in MuJoCo, not an F/G/R checkpoint result or numerical evidence for robustness.

The original model revision, policy checkpoint, trajectory, camera settings and montage-generation procedure are not established. This repository does not claim to reproduce the figure. The author authorized publication of the English image; applicable third-party model and asset rights remain separate.

Run `python -B -m pytest -q -p no:cacheprovider tests/test_gait_illustration.py` to check the retained image hash and dimensions.
