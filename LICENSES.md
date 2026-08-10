# 来源、许可证与引用

本文件用于代码复审中的来源披露，不构成对任何第三方代码的额外许可授权。

## 1. PLR-003 Jittor port

`plr3d/`、`train.py`、`infer.py`、`assemble*.py`、`run_plr003.py` 及工具脚本是为本次比赛复现整理的 Jittor 实现。其用途是提交方的比赛代码复审；除赛事规则所要求的评审、运行和存档权限外，不在本包中另行声明通用开源许可证。

## 2. 3DMambaIPF

模型结构和历史 checkpoint 来源于：

- Qingyuan Zhou, Weidong Yang, Ben Fei, Jingyi Xu, Rui Zhang, Keyi Liu, Yeqi Luo, Ying He,
  “3DMambaIPF: A State Space Model for Iterative Point Cloud Filtering via Differentiable Rendering,” AAAI 2025.
- Project: <https://github.com/TsingyuanChou/3DMambaIPF>
- arXiv: <https://arxiv.org/abs/2404.05522>

审计时上游仓库未提供独立 LICENSE 文件，因此本包不推定其为某种开放源代码许可证。这里包含的是为比赛正式 runtime 编写的 Jittor port，不包含上游 PyTorch、Triton、`mamba_ssm` 或 `causal_conv1d` 文件。

建议引用：

```bibtex
@inproceedings{zhou20253dmambaipf,
  title={3DMambaIPF: A State Space Model for Iterative Point Cloud Filtering via Differentiable Rendering},
  author={Zhou, Qingyuan and Yang, Weidong and Fei, Ben and Xu, Jingyi and Zhang, Rui and Liu, Keyi and Luo, Yeqi and He, Ying},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39}, number={10}, pages={10843--10851}, year={2025}
}
```

## 3. Mamba

Selective state-space equations和 Mamba block 设计参考：

- Albert Gu and Tri Dao, “Mamba: Linear-Time Sequence Modeling with Selective State Spaces,” 2023.
- Project: <https://github.com/state-spaces/mamba>
- Upstream license: Apache License 2.0。

本包没有复制上游 CUDA/Triton kernel；`plr3d/ops/selective_scan.py` 是基于公开方程编写的 Jittor `jt.code` forward/backward。

## 4. IterativePFN

3DMambaIPF 上游声明其部分建立在 IterativePFN 上：

- Project: <https://github.com/ddsediri/IterativePFN>
- Upstream license: MIT License, Copyright (c) 2023 ddsediri。

本包没有直接复制 IterativePFN 仓库文件，但在此保留其学术和许可证来源链。

## 5. Jittor StraightPCF / ROT

`rot_jittor/` 由比赛提供的 Jittor Track2 StraightPCF baseline 裁剪而来，只保留冻结 ROT inference 所需的 Jittor 模型文件；数据 loader、配置框架和 calibration CLI 已移除，calibration 公式在 `generate_rot.py` 中以 Numpy 重写。提供的 baseline 中未发现独立 LICENSE 文件；该子树按赛事代码复审目的提交，不推定额外的开放源代码授权。

请同时引用 StraightPCF 对应论文/基线说明及 Jittor：

- Jittor project: <https://github.com/Jittor/jittor>
- Jittor 按其上游仓库许可证使用。

## 6. Python 依赖

正式运行只安装 Jittor 和 Numpy；它们不被 vendored 到本包，分别遵循各自上游许可证。若在源码包外迁移历史 checkpoint，原研究环境可能使用 PyTorch，但该 reader、依赖和 runtime 均不在复审 ZIP 中。
