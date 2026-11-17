# Sources, attribution, and provenance

This file records only currently evidenced facts. It is not a license grant, publication authorization, or proof that the experiments can be rerun.

## Manuscript identity

- Author: Jinlong Liu.
- Title: *Steady-State Bias Compensation for Learned Bipedal Locomotion: Diagnosis, Frozen-Gain Integral Control, and Hazard-Derived Command Capping*.
- Status: unpublished CEP manuscript draft; no DOI or publication is assumed.
- Evidence locations in the source workspace: `short_paper/robustness_analysis/journal/paper_cep.md` and the CEP manuscript template. This release candidate keeps the paper identity separate from any future repository name.

## Code and upstream attribution

- Unitree Robotics, [unitreerobotics/unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym), checked-out commit `276801e46c5d433564f24658bac64f254b7d2d4b`. Its root BSD 3-Clause notice credits HangZhou YuShu TECHNOLOGY CO., LTD.; the upstream `legged_gym/LICENSE` has separate ETH Zurich/NVIDIA attribution. Do not imply endorsement.
- `historical/` is a selective snapshot from an author-authorized private checkout at commit `8df0dfc8669620b5ee058dbe117fe92b653aafc6`, not that entire repository. The retained `historical/LICENSE` and notices must remain with the applicable files. Adapted files and pre-adaptation source hashes are not interchangeable evidence of current-file identity.
- `controller.py` and `analysis.py` are new reference implementations of manuscript equations. They differ from the historical scripts and are not claimed to reproduce the original implementation line-for-line or all experiment behavior; a synthetic `OuterLoop` check does not establish equivalence for every schedule or trajectory.
- MuJoCo, Isaac Gym, `rsl_rl`, PyTorch, Unitree robot XML/meshes, datasets and checkpoints may each have independent terms. The XML and meshes are expected from a separate upstream checkout and are not a grant to redistribute them. Check notices and platform requirements before release.

## Simulator and data scope

Historical results and offline recomputation datasets are not included in this minimal repository. The launcher exposes selected historical MuJoCo configurations only. The Isaac Gym overlay has not been runtime-verified. No MATLAB execution is provided or verified.

The original SP-A3 periodic-orbit precheck failed, and its protocol prohibited the full run. A later smaller-step E3 run existed, but its ALL PASS status checked file consistency rather than the scientific gate; no independent replacement precheck record has been established. FTLE/return-map association with fall rate is not causal evidence, and a steady bias does not prove absence of integral action. These are submission-blocking author-review items, not validated Floquet or mechanism claims. The historical `cross_layer` calculation uses a final constant command approximation rather than integrating the actual command trajectory; its scientific interpretation also requires author review.

The Isaac overlay is present with retained license notices; Isaac import, GPU execution and result validity remain unverified. Earlier offline audits of historical records do not constitute new simulations or establish training reproducibility; the audited data and recomputation scripts are not included here.

The MuJoCo launcher and safety fixes are present. Mock workflow tests verify software behavior only, not the periodic-orbit scientific gate, which did not pass.

The paper and current release candidate retain these limitations: simulated results only; one training seed per checkpoint; outer-loop ground-truth velocity; post-hoc round-3 comparisons; a failed G RMSE criterion; unverified R training provenance; and failed periodic-orbit preconditions for interpreting Floquet quantities. These constraints prohibit hardware claims, broad cross-engine claims, or treating round-3/post-hoc material as preregistered confirmation.

## F/G/R checkpoint identity

The following are archived registry records and SHA-256 values, not a license or proof of safe deserialization:

| Policy | Run | Checkpoint | Recorded SHA-256 |
|---|---|---|---|
| F | `Mar11_13-47-46_` | `model_10000.pt` | `975dbcec99b81d9d467e4d6e4404b29a301a2b7a5f9b7744f5b9d4955b3b22a0` |
| G | `Mar11_17-23-39_` | `model_10000.pt` | `66c7cfb05388da0ba247a00e8bbff48ef89f70c05d16d8bb6b0da68e078f46b4` |
| R | `Mar11_13-45-41_` | `model_10000.pt` | `dd1bf3d60f26c9331b7ff93044e15926e5ab44d4a8d1ca38d667121598e0d383` |

The files under `checkpoints/{F,G,R}/model_10000.pt` match the recorded manifest hashes according to the repository's byte-hash check. The user has authorized public disclosure of data and F/G/R weights; reuse terms for original data and checkpoints owned and licensable by Jinlong Liu are CC BY 4.0 (see [DATA_LICENSE.md](DATA_LICENSE.md)). This hash match does not independently verify training logs, architecture, safety, or the full commit-to-output chain. The paper records a 512–256–128 ELU MLP; the public upstream LSTM configuration and `deploy/pre_train/g1/motion.pt` are not verified substitutes.

Historical loading uses `torch.load(..., weights_only=False)`. Never load an untrusted checkpoint, including a file with an expected name or digest. Before publication, author review must cover checkpoint/asset rights, private metadata, third-party notices, the scope of data and weight reuse terms, and the scientific limitations above. New project code is licensed under the BSD 3-Clause root LICENSE, copyright 2026 Jinlong Liu; third-party terms remain separate. This is a local release candidate; scientific author review remains pending.
