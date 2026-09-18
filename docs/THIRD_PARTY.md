# 来源、引用与许可范围

## NKAI 自有贡献

根目录 MIT LICENSE 仅适用于 NKAI 拥有权利的新增代码、发布工具与项目文档。其范围包括本次权重下载/校验、输入清单与检查工具、融合/打包组织，以及团队有权授权的 Jittor 新增实现。原仓库中其他作者的文件或派生部分保留上游适用许可；MIT 不覆盖第三方数据、模型权重，也不替代未经明确许可的上游授权。

本公开版本的自有贡献以 MIT 提供。历史复审文档保留当时的来源记录，当前自有新增部分以根目录 LICENSE 为准；第三方部分按各自适用条款使用。

## 参考模型与框架

| 工作 | 使用关系 | 来源与许可说明 |
|---|---|---|
| 3DMambaIPF, AAAI 2025 | 四阶段 EdgeConv–Mamba 网络设计与来源链 | [项目](https://github.com/TsingyuanChou/3DMambaIPF) · [论文](https://arxiv.org/abs/2404.05522)；随包审计未发现上游独立 LICENSE，不推定额外授权 |
| IterativePFN, CVPR 2023 | 四阶段迭代过滤设计 | [项目](https://github.com/ddsediri/IterativePFN) · [论文](https://arxiv.org/abs/2304.01529)；上游 MIT，参见原仓库 |
| StraightPCF, CVPR 2024 | 官方 Jittor baseline 及参考 VM/ROT | [论文](https://arxiv.org/abs/2405.08322)；随包官方 baseline 未附独立许可证，引用不等于另行授权 |
| Mamba | Selective state-space 方程与模块设计 | [项目](https://github.com/state-spaces/mamba)；上游 Apache-2.0；本包 selective scan 为 Jittor `jt.code` 实现，不包含上游 CUDA/Triton kernel |
| Jittor | 模型训练与推理框架 | [项目](https://github.com/Jittor/jittor)；按其上游许可证使用 |

详细历史来源链保留于 [Mamba LICENSES](../solution/code/mamba/jittor_core/LICENSES.md) 和 [历史复审 LICENSES](../solution/a_board/code/LICENSES.md)。依赖通过包管理器安装，不在仓库中重新分发。

## 权重与数据

Releases 的四个 checkpoint 是团队提供的最终比赛模型，文件保持原始字节，并公开 SHA-256 供核对。模型使用与再分发遵循来源许可和赛事/数据条款。数据集通过赛事渠道取得。训练入口与配置见 [TRAINING.md](TRAINING.md)。

## 学术引用

请按实际使用引用 3DMambaIPF、IterativePFN、StraightPCF、Mamba 和 Jittor 原始工作。引用本仓库可使用 GitHub 的 “Cite this repository” 或 [CITATION.cff](../CITATION.cff)。团队的成员组织、工程实现及融合方案与上游架构贡献应分别表述。
