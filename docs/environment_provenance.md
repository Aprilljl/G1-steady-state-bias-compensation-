# G1 资产与环境溯源记录

日期：2026-11-17。只记录只读核验；不复制机器人资产、不加载权重、不联网、不安装依赖、不运行仿真。

## 来源与范围

- 已发布来源：`unitree_rl_gym` 的仓库 remote 记录为 `https://github.com/unitreerobotics/unitree_rl_gym.git`，记录提交为 `276801e46c5d433564f24658bac64f254b7d2d4b`；本次未联网独立确认上游提交、tag 或发布包。
- 私有对照源：`private_source` 的记录提交为 `8df0dfc8669620b5ee058dbe117fe92b653aafc6`；精确上游来源尚未建立。其本地目录不写入发布清单，由测试在可用时显式发现。
- 入口是 `resources/robots/g1_description/scene.xml`。生成器递归解析 XML `<include>` 与 file-backed `<mesh>`，当前闭包为 29 个文件（2 个 XML、27 个 mesh）。清单逐项记录相对路径和 SHA-256，见 `historical/assets_manifest.json`。

## 生成与测试

生成器 `historical/assets_manifest_generator.py` 不含私有绝对路径：默认仅使用脚本位置推导的相邻 `unitree_rl_gym`；私有比较源必须显式传入 `--comparison-root`；输出必须使用新的 `--output` 路径，已存在文件会被拒绝，不会覆盖既有 manifest。

```text
python historical/assets_manifest_generator.py --source-root <unitree_rl_gym> --comparison-root <private_source> --output <new_manifest.json>
python -B -m pytest -q -p no:cacheprovider tests/test_assets_provenance.py
```

测试仅依赖已发布 manifest；私有源存在时逐项核对其文件 SHA，缺失时 skip。测试不导入 MuJoCo，不复制资产或加载权重。

## 环境与限制

- OS：Windows (`win32`)
- Python：`3.13.9 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 19:09:58) [MSC v.1929 64 bit (AMD64)]`
- 当前本地副本的 SHA-256 一致，只证明本次核验时文件字节一致，不能单独证明历史实验使用的模型资产、上游来源或授权链。上游独立核验曾因 `Upstream request failed` 未完成；机器人 XML/meshes 的许可及再分发权利仍待确认。
