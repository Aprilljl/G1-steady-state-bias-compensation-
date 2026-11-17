# Third-party notices and rights boundaries

本文件是事实性通知，不是法律意见。新代码使用根目录 BSD 3-Clause LICENSE；数据、权重和论文材料的再利用条款独立处理。

## Unitree Robotics / unitree_rl_gym

已登记的上游为 Unitree Robotics 的 `unitreerobotics/unitree_rl_gym`，历史快照中的 Unitree 代码带有 BSD 3-Clause notice。选择性历史快照不是整个私有仓库。再分发 Unitree 代码时必须保留适用版权、条件和免责声明，并避免暗示 Unitree 背书。发布前应随实际分发内容核对上游 commit、许可证文本和是否包含额外资产条款。

## legged_gym / ETH Zurich / NVIDIA

`legged_gym` 的版权/许可证链与 Unitree notice **独立处理**；历史 provenance 标明其许可证版权归属 ETH Zurich、Nikita Rudin 和 NVIDIA。不能把 `legged_gym` 误写成 Unitree 代码、不能用 Unitree BSD notice 代替其独立 notice，也不能因引用上游就声称获得机器人资源或训练资产的公开分发权。若公开代码包含其修改或衍生部分，应随实际文件保留相应 LICENSE/版权标记。

## Isaac Gym、rsl_rl、MuJoCo 与机器人资源

Isaac Gym、rsl_rl、MuJoCo、G1 XML/mesh/material/motion 资源及其他依赖可能有各自许可、注册、下载和再分发条件。本候选目录不把外部 XML/mesh 自动变成可公开打包内容；`UNITREE_RL_GYM_ROOT` 指向的外部资源应由发布者单独核验。不要下载、捆绑或替换缺失资源来制造“可复现”。

## 论文、脚本、数据与历史快照

CEP manuscript source 是未发表稿；论文文本、图、表及历史快照的版权与第三方权利需分别核查。`PROVENANCE.md` 和 `historical/SOURCE_HASHES.json` 是来源记录，不是许可证授予。历史实验结果和离线复算数据未包含在本仓库；仅 Jinlong Liu 拥有并有权许可的原创实验数据适用 [CC BY 4.0](DATA_LICENSE.md)，其他权利仍需核查。

## F/G/R checkpoints

`checkpoints/{F,G,R}/model_10000.pt` 的哈希与历史登记匹配；用户已授权公开权重，并为其拥有且有权许可的原创权重指定 CC BY 4.0，见 [DATA_LICENSE.md](DATA_LICENSE.md)。这些文件已纳入仓库；训练日志来源、第三方权利和架构独立验证仍需核查。

## New project code and data

本项目新代码采用 BSD 3-Clause，Copyright (c) 2026, Jinlong Liu，详见根目录 LICENSE。上游和其他第三方代码保留其独立版权和许可通知。用户已授权公开数据及 F/G/R 权重，只有其拥有并有权许可的原创数据和权重适用 CC BY 4.0；新代码许可证和数据许可证均不自动覆盖论文材料或第三方资源。

## Citation checklist

发布前至少独立核对并引用：Unitree `unitree_rl_gym`；`legged_gym` 及其 ETH Zurich/Nikita Rudin/NVIDIA 版权链；Isaac Gym（若实际使用）；rsl_rl；MuJoCo；G1 资产提供方；以及论文所依据的方法和数据。引用是学术归属要求，不是再分发许可。
