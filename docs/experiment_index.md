# 实验范围与复现入口

本仓库保留参考控制器、选择性历史仿真代码、设计配置、F/G/R 权重及英文版 Fig. 9。历史实验结果和离线复算数据未收录；无法使用本仓库直接重算论文表格或完整复现实验。

## 入口

- MuJoCo：见 [MuJoCo 说明](mujoco_reproduction.md)，`g1_runner.py` 仅支持选定实验；外部机器人资产需单独取得。
- Isaac Gym：见 [Isaac Gym 说明](isaacgym_reproduction.md)。overlay 仅通过静态/离线检查，尚未验证 Isaac 导入、GPU 执行或结果有效性。
- 图 9：`historical/gait_illustration/simulation_gait_sequence.png` 是定性英文版图像；模型、策略和轨迹的完整生成链尚未确认。
- 默认测试：`python -B -m pytest -p no:cacheprovider tests`。测试通过不代表重新仿真或训练。

## 科学边界

结果仅来自仿真，每个 checkpoint 对应单训练 seed，外环使用真值速度；不得声称实机验证。G 的 RMSE 判据失败，cap 经修订，15 kg 条件不可达；Round-3 是后验候选，尚待作者审核。原始 Floquet 周期轨道前置检查失败，不能声称机制已验证；cross-layer 分析使用最终命令近似，并非对真实命令轨迹积分。离线审计不是 simulator rerun。有关权重、许可和安全限制参见 [README](../README.md)、[来源](../PROVENANCE.md) 及 [第三方通知](../THIRD_PARTY_NOTICES.md)。
