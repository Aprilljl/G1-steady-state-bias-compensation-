# Isaac Gym 平地评估：候选叠加与离线审查

`historical/isaacgym_release/overlay/` 包含授权私有仓 `8df0dfc8669620b5ee058dbe117fe92b653aafc6` 的白名单原始文件，包括真实 `legged_gym/scripts/eval_batch.py`、两个环境文件、`thesis_eval` 评估链及两份 BSD 许可。逐文件原始 SHA-256 见 `overlay_manifest.json`。这是 **SP-A2 / SP-B1g 平地场景的候选闭包**，不是已验证的可运行发行版；动态导入、模拟器和结果均未验证。上层 `isaacgym_release/eval_batch.py` 是离线 stub，不是真实入口。

## 离线计划

```text
python -B -m unittest discover -s tests -p test_isaacgym_release.py -v
python -B historical/isaacgym_release/scenarios.py --registry PATH\TO\registry.csv --out-root PATH\TO\planned-output --only SP-A2
```

规格展开不导入 Isaac Gym、torch、不检查 checkpoint 字节、不写输出。`g1_runner.py registry` 可在只读哈希校验后生成 registry（见项目 README）。配置仅用 F `Mar11_13-47-46_`：SP-A2 为 13 个速度、各 100 回合、60 s；SP-B1g 为 6 载荷 × 9 速度、各 20 回合、30 s；warmup 2 s、平地。不可将 G/R、上游 LSTM 或 `motion.pt` 当 F。

## 人工应用到独立沙箱（未执行）

1. 准备用户自己拥有的**全新、独立** `unitree_rl_gym` checkout，不修改现有上游或私有仓。历史记录的上游提交为 `276801e46c5d433564f24658bac64f254b7d2d4b`，推荐使用；若 `git rev-parse HEAD` 为其他提交（例如 `3a4cd236fc5428654c7a7b9442f463d4b9f2d667`），可继续，但须在运行记录中写明实际提交，且步骤 2 的两处上游文件哈希必须匹配。`git status --porcelain` 应为空。
2. 对照 `overlay_manifest.json` 计算每个 overlay 文件 SHA-256；对两个待替换的上游文件 `legged_gym/envs/base/legged_robot.py` 和 `legged_gym/envs/g1/g1_config.py`，计算原始字节 SHA-256 并与 manifest 的 `upstream_sha256` 对照。匹配说明沙箱与历史上游一致；不匹配（例如 `3a4cd236` 上实测为 `9777a260…` 和 `4ecbf0be…`）时可继续，但须在运行记录中写下实际哈希，并先把这两个文件备份为 `*.orig`，再用 overlay 版本整体替换。此时上游在这两个文件上的改动会被覆盖，其余上游文件与私有版本的兼容性需自行确认。对每个新增 `thesis_eval` 和 `eval_batch.py` 路径，必须确认目标不存在；不得覆盖已有用户文件。检查目录不是链接到原 checkout。
3. 仅在沙箱中按相同相对路径复制 manifest 中的 Python 和 YAML 文件；两个环境文件在步骤 2 严格验证之后才替换。BSD 许可作为来源记录保留在 overlay，勿覆盖沙箱许可。逐个复算目标文件哈希与 `private_sha256` 一致，核对 `git diff`。manifest 的 null 表示未核实或不存在，不能推断允许覆盖。
4. 在授权、兼容的 Linux Isaac Gym 环境中另行核对 Isaac Gym、PyTorch、`rsl_rl` 1.0.2、NumPy、PyYAML、G1 URDF 和所有 mesh 的来源、版本与许可；确认训练模型可信及 SHA。`torch.load` 可执行代码。运行前检查私有与上游 base config、G1 env、task registry、helpers 和 terrain 差异：这些候选未整体覆盖，可能阻断或改变结果。不可因 smoke 通过即认定等价。
5. 历史实际入口（**只作待审核说明，本机未运行**）：

```text
python thesis_eval/run_eval.py --config thesis_eval/configs/short_paper_isaacgym.yaml --registry PATH/TO/registry.csv --out_root PATH/TO/NEW_RESULTS --only SP0_smoke_isaac
```

该命令会真正启动模拟并写文件；仅在明确获准、环境依赖和科学设置已验证后运行。先核对历史 runner 不会覆盖已有结果，禁止将 smoke 结果当论文结果。

## 差异及限制

历史 evaluator 执行 per-env 质量/摩擦 monkeypatch、checkpoint 架构推断和严格 state_dict 加载；私有 `legged_robot.py` 比上游增加足部状态等逻辑，上游差异约 215 行新增、33 行删除。上游 G1 默认 `ActorCriticRecurrent` LSTM；论文登记为 `ActorCritic` MLP `[512,256,128]` ELU。历史 evaluator 虽推断网络，但不会自动强制其与论文登记一致；运行前必须独立验证记录和加载后的架构。其他依赖闭包、实际物理结果及 checkpoint 架构未验证。BSD 许可仅覆盖相应代码；Isaac、rsl_rl、资产、权重及新增内容许可仍须核查。
