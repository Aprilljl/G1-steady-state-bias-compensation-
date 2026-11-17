# 历史 MuJoCo 复现路径

本目录只保留论文历史 MuJoCo 路径的可审查入口。`g1_runner.py` 不会下载、复制或反序列化权重；registry 阶段仅对用户指定的 F/G/R 文件做 SHA-256 校验。F/G/R 的哈希匹配不等于来源、许可或架构已确认。

## 版本、根目录与设计对应

所有直接命令均从项目根目录执行（以下以 `<PROJECT_ROOT>` 表示，不包含机器特定路径）；脚本 `mujoco_g1_12dof/sp_control.py` 中 `HERE` 是 `historical/thesis_eval/mujoco_g1_12dof`，`HERE.parents[1]` 因而是 `historical`，即代码中的 `REPO_ROOT`。配置、design 和 registry 的相对路径应从 `<PROJECT_ROOT>` 解析；`<OUT>` 必须是项目外且尚不存在的新目录。执行前应校验当前工作目录为 `<PROJECT_ROOT>`，或将所有参数改为绝对路径。

| 论文实验 | 历史 experiment | 脚本/配置 | design | 版本与输出 |
|---|---|---|---|---|
| B2 step identification | `SP-B2_step_id` | `mujoco_g1_12dof/sp_control.py` + `short_paper_control.yaml` | 无 | S2；`<OUT>/SP-B2_step_id/` |
| B3 compensation | `SP-B3_compensation` | `sp_control.py` + `short_paper_control.yaml` | `short_paper_design/sp_design.json`（v1） | S3；`<OUT>/SP-B3_compensation/` |
| B4 gain sweep | `SP-B4_gain_sweep` | 同上 | v1 | S3；`<OUT>/SP-B4_gain_sweep/` |
| B5 payload step | `SP-B5_payload_step` | 同上 | v1 | S3；`<OUT>/SP-B5_payload_step/` |
| B6 estimator | `SP-B6_estimator` | 同上 | v1 | S3；`<OUT>/SP-B6_estimator/` |
| B7 cap | `SP-B7_cap` | 同上 | v1 | S3；`<OUT>/SP-B7_cap/` |
| E1 gain onset | `SP-E1_gain_onset` | `sp_control.py` + `short_paper_followup.yaml` | v1；`analysis/sp_e1_rule.py` | follow-up；`<OUT>/SP-E1_gain_onset/` |
| E2a | `SP-E2a_b3_15kg` | `sp_control.py` + `short_paper_followup.yaml` | **v2** `sp_design_v2.json` | corrected cap；`<OUT>/SP-E2a_b3_15kg/` |
| E2b | `SP-E2b_b5_15kg` | 同上 | v2 | corrected cap；`<OUT>/SP-E2b_b5_15kg/` |
| E2c | `SP-E2c_b7_15kg` | 同上 | v2 | corrected cap；`<OUT>/SP-E2c_b7_15kg/` |
| E2d | `SP-E2d_b7_none` | 同上 | v2 config family, no compensation | corrected control；`<OUT>/SP-E2d_b7_none/` |
| A3 | `SP-A3` implementation | `mujoco_g1_12dof/sp_floquet.py` | no JSON design; periodic-orbit precheck | blocked unless precheck passes |

`SP-A1_speed_hazard` 与 `SP-B1_static_map` 是基础 MuJoCo hazard/static-map 实验，不是 B3；B3 是 `SP-B3_compensation`。`SP-B2_step_id` 是 B2。不要把 S1 的 A1/B1 命名映射为 B3。

## 设计边界与 gate

- `SP0_smoke_mujoco` / `SP0_smoke_control` 是本地极小 smoke，不是论文结果；只有显式 `--confirm-run` 才会启动 launcher 的历史 runner。
- B3-B7 使用冻结 v1 design；E1 明确使用 v1；E2 使用更正后的 v2。首次 15 kg cap 与 v2 的 15 kg cap 不得合并解释。
- S3 author-review gate 必须在正式 B3-B7/E1/E2 运行前通过；launcher 当前只执行配置/依赖/路径检查，不替代作者审核，也不自动运行 round3。
- `short_paper_round3_F.yaml` / `short_paper_round3_G.yaml` 与返回映射保留在历史目录；它们属于第三轮候选，未通过 S3 gate 前不应启动。R 没有匹配的历史 MuJoCo runner/config，入口只允许登记，不能套用 F/G。
- A3 的 `sp_floquet.py` 要求 stateless policy 和周期轨道、rho/FTLE 前置检查；失败即闭锁，不应把 Floquet 框架当作已验证机制。当前上游 `motion.pt` 是 LSTM stand-in，不可用于声称 A3 结果。
- 顶层 `g1_runner.py run` 的 smoke 使用用户指定、SHA-256 匹配的论文 checkpoint；它不是 TorchScript stand-in，也不会把上游 `motion.pt` 当作论文权重。需要 stand-in 时必须直接调用历史 runner 的 `--local_jit`，输出标记为 `@jit_standin`，不能声称复制论文结果。

## 计划展开（不执行仿真）

以下 `sp_control.py --dry_run` 只展开计划，不加载 checkpoint 或写入结果。注意 `ch5_mujoco.py --dry_run` 行为不同：它会写入 specs，因此不是纯只读；使用新的外部输出目录。所有 checks 仅证明 Python 模块存在、scene 存在及 registry/checkpoint 哈希匹配，不证明模型架构、观测维度或实际运行兼容。直接历史脚本的 `--force` 会允许重用已有输出，必须人工审查后使用；`--dry_run` 的写入行为也需按脚本分别核对。

```text
cd /d <PROJECT_ROOT>
python -B historical/thesis_eval/mujoco_g1_12dof/sp_control.py --config historical/thesis_eval/mujoco_g1_12dof/configs/short_paper_control.yaml --design historical/eval_results/short_paper_design/sp_design.json --registry <REGISTRY> --out_root <OUT> --only SP-B3_compensation SP-B4_gain_sweep SP-B5_payload_step SP-B6_estimator SP-B7_cap --workers 1 --dry_run
python -B historical/thesis_eval/mujoco_g1_12dof/sp_control.py --config historical/thesis_eval/mujoco_g1_12dof/configs/short_paper_followup.yaml --design historical/eval_results/short_paper_design/sp_design.json --registry <REGISTRY> --out_root <OUT_E1> --only SP-E1_gain_onset --workers 1 --dry_run
python -B historical/thesis_eval/mujoco_g1_12dof/sp_control.py --config historical/thesis_eval/mujoco_g1_12dof/configs/short_paper_followup.yaml --design historical/eval_results/short_paper_design/sp_design_v2.json --registry <REGISTRY> --out_root <OUT_E2> --only SP-E2a_b3_15kg SP-E2b_b5_15kg SP-E2c_b7_15kg SP-E2d_b7_none --workers 1 --dry_run
```

`<REGISTRY>` 必须由 `g1_runner.py registry` 生成且包含选定 policy 的真实哈希匹配路径；dry-run 仍需提供 registry 参数，但不会加载 checkpoint。registry CSV 含本机绝对路径，不应提交、发布或复制到共享仓库。

安全约束：历史 `sp_control.py` 直接调用现在拒绝 `historical/eval_results` 下的输出，并默认拒绝任何已存在输出目录；不要用 `--force` 绕过该保护。顶层 `g1_runner.py run` 会先独占创建新的项目外目录，再把其尚不存在的 `results` 子目录传给历史 runner，不传递 `--force`，不会覆盖旧结果。`g1_runner.py` 的 registry 阶段自身不反序列化权重；但 `run` 会启动历史子进程，而该子进程会加载 checkpoint，可能执行其中的代码。非 stand-in 运行只应使用可信且哈希核验的权重。

## 直接历史脚本示例（不执行）

B3-B7（v1）：

```text
python -B <PROJECT_ROOT>\historical\thesis_eval\mujoco_g1_12dof\sp_control.py --config <PROJECT_ROOT>\historical\thesis_eval\mujoco_g1_12dof\configs\short_paper_control.yaml --design <PROJECT_ROOT>\historical\eval_results\short_paper_design\sp_design.json --registry <REGISTRY> --out_root <OUT> --only SP-B3_compensation --workers 16
```

E1（v1）：

```text
python -B <PROJECT_ROOT>\historical\thesis_eval\mujoco_g1_12dof\sp_control.py --config <PROJECT_ROOT>\historical\thesis_eval\mujoco_g1_12dof\configs\short_paper_followup.yaml --design <PROJECT_ROOT>\historical\eval_results\short_paper_design\sp_design.json --registry <REGISTRY> --out_root <OUT_E1> --only SP-E1_gain_onset --workers 16
python -B <PROJECT_ROOT>\historical\thesis_eval\analysis\sp_e1_rule.py <OUT_E1>\SP-E1_gain_onset
```

E2（v2）：

```text
python -B <PROJECT_ROOT>\historical\thesis_eval\mujoco_g1_12dof\sp_control.py --config <PROJECT_ROOT>\historical\thesis_eval\mujoco_g1_12dof\configs\short_paper_followup.yaml --design <PROJECT_ROOT>\historical\eval_results\short_paper_design\sp_design_v2.json --registry <REGISTRY> --out_root <OUT_E2> --only SP-E2a_b3_15kg SP-E2b_b5_15kg SP-E2c_b7_15kg SP-E2d_b7_none --workers 16
```

A3（仅在周期轨道 precheck 通过且作者批准后）：

```text
python -B <PROJECT_ROOT>\historical\thesis_eval\mujoco_g1_12dof\sp_floquet.py --registry <REGISTRY> --out_root <OUT> --workers 16
```

上述命令均为文档示例，当前任务未执行正式实验。

## 只读登记

```text
python -B g1_runner.py registry --f <F_CHECKPOINT> --g <G_CHECKPOINT> --r <R_CHECKPOINT> --out <REGISTRY_CSV>
```

目标文件已存在时拒绝覆盖。登记只读字节；不要把生成的绝对路径 CSV 发布。

## 环境检查

```text
python -B g1_runner.py check --policy F --experiment mujoco-smoke --f PATH\F\model_10000.pt
python -B g1_runner.py check --policy G --experiment S2-step --g PATH\G\model_10000.pt
```

检查依赖 `numpy`、`torch`、`mujoco`、`yaml` 与同级 `unitree_rl_gym/resources/robots/g1_description/scene.xml`（或 `UNITREE_RL_GYM_ROOT`）。缺项即停止。

## 明确运行

仅在用户确认可信权重、理解 `torch.load(..., weights_only=False)` 的反序列化风险，并明确允许运行时：

```text
python -B g1_runner.py run --policy F --experiment mujoco-smoke --f <F_CHECKPOINT> --out <NEW_EXTERNAL_OUTPUT> --trust-checkpoint --confirm-run
```

输出必须是项目外、尚不存在的新目录；不会覆盖或合并历史结果。正式 B3-B7/E1/E2/A3 实验仍需逐项人工审核冻结 design、episode 长度、analysis window 与 S3 gate，不能以 smoke 代替。

## Stand-in smoke

若只验证 MuJoCo 与真实上游 TorchScript stand-in 的连接，可使用历史 runner 的 `--local_jit`，输出应位于新临时目录并带 `_jit_standin` 元数据。该路径不是论文 checkpoint 路径，也不提供论文数值证据。缺少上游 XML、stand-in 或 MuJoCo 依赖时测试应失败闭锁，而不是创建假文件。

## 验证命令

```text
python -B -m unittest discover -s tests -v
python -B -m unittest discover -s historical/thesis_eval/mujoco_g1_12dof/tests -v
```
