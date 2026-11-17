# 发布前检查清单

- [ ] 按当前 Git 树及拟发布的历史逐项审查隐私；已删除文件仍可能存在于历史中。不要依赖 `.gitignore` 作为隐私边界。
- [ ] 核实第三方源码、机器人资源及权重的适用许可和版权通知；保留 [第三方通知](../THIRD_PARTY_NOTICES.md) 和上游许可证。原创权重的许可见 [DATA_LICENSE.md](../DATA_LICENSE.md)。
- [ ] 确认 F/G/R 文件与 `model_manifest.json` 哈希匹配。哈希不是可信来源、安全反序列化或训练溯源的证明；PyTorch pickle 和 TorchScript 只加载可信文件。
- [ ] 明确历史实验结果和离线复算数据不在当前仓库；不声称它们可由本仓库复算。保留单训练 seed、真值速度、G RMSE 失败、cap 修订、Floquet 前置条件失败、cross-layer 最终命令近似和 Round-3 作者审核等科学限制。
- [ ] Isaac Gym overlay 仅经过静态/离线检查；不得声称已验证 Isaac 导入、GPU 运行或结果。Fig. 9 英文图为定性说明，完整生成链未确认。
- [ ] 直接历史脚本可能绕过上层安全检查并覆盖输出；执行前检查依赖、许可、输入和全新输出目录。不要将 smoke 结果称为正式实验。
- [ ] 在干净副本运行 `python -B -m pytest -p no:cacheprovider tests`，记录跳过项及环境；发布由作者自行决定和执行。
