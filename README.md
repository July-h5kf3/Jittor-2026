# Track2 点云去噪（Jittor）

本仓库只保留当前最佳方案、复现实验所需的核心代码与关键实验结论。大规模数据集、模型权重、预测结果和提交包仅保存在训练服务器，不进入 Git。

截至 2026-07-13，项目记录中的最佳线上成绩仍为 **75.51**，对应 **CVM-002：Graph StraightPCF + FiLM + stage-velocity target + deep supervision + multi-scale distance head**。最新 local2 提交候选是在同一权重上使用 `predict_alpha=1.05`，得分 **72.82**；它只比 `alpha=1.0` 高 0.05，尚不能替代线上已验证的默认方案。

## 当前最佳方法

| 部分 | 最终选择 | 说明 |
|---|---|---|
| 主干 | StraightPCF，4 个 velocity modules | module=4 是 16GB 显存条件下验证过的容量上限 |
| 编码/解码 | EdgeConv 编码器 + graph decoder | 图卷解码是最稳定、最大的结构收益来源 |
| 条件 | FiLM | 将噪声相关条件注入特征 |
| CVM 目标 | `stage_velocity` | 每个阶段学习自己的局部速度，而不是重复预测完整残差 |
| 监督 | `cvm_deep_sup: true` | 对中间阶段增加监督；这是 CVM-002 超过 CVM-001 的关键 |
| Distance head | multi-scale encoder | 改善剩余步长估计 |
| 训练分布 | `sigma=0.008–0.014`，含 2% outliers | 与线上噪声分布对齐比单纯扩大模型更有效 |
| 推理 | 单次推理，`alpha=1.0`，无 TTA/融合 | 多趟推理和普通 TTA 曾出现本地上涨、线上下降 |

核心配置：

- `configs/task/train_spcfgfncvm002_cvm.yaml`
- `configs/task/train_spcfgfncvm002.yaml`
- `configs/task/predict_spcfgfncvm002.yaml`
- `configs/model/spcfgfncvm002_{cvm,spcf}.yaml`

local2 提交候选配置：

- `configs/task/predict_spcfgfncvm002a105.yaml`
- `configs/task/predict_spcfgfncvm002a105_local2.yaml`
- `configs/model/spcfgfncvm002a105_spcf.yaml`

## 服务器端保留的最佳产物

这些文件被 `.gitignore` 排除，不上传 GitHub：

| 用途 | 路径 |
|---|---|
| CVM-002 初始化权重 | `experiments/_bak_official_75.01/cvm_checkpoint_best.pkl` |
| CVM-002 最佳 CVM 权重 | `experiments/spcfgfncvm002_cvm/checkpoint_best.pkl` |
| CVM-002 最佳完整权重 | `experiments/spcfgfncvm002_spcf/checkpoint_best.pkl` |
| 最佳线上提交包 | `submission_results/result_cvm002_spcfgfncvm002.zip` |

## 环境与运行

```bash
conda create -n jittor python=3.8 -y
conda activate jittor
pip install -r requirements.txt
```

训练数据目录约定：

```text
dataset_train/shapenet/<synset>/<model>/models/model_normalized.obj
dataset_test_noisy/shapenet/<synset>/<model>/noisy.npy
```

训练最佳方案：

```bash
GPU_LIST=0,1,2,3 NP=4 bash scripts/run_best_pipeline.sh
```

生成新的提交包（默认使用 CVM-002 alpha 1.05 候选）：

```bash
GPU=0 bash scripts/package_submission.sh
```

提交到 Educoder Track2 A 榜前，先安装独立的提交依赖并执行只读检查：

```bash
pip install -r requirements-submit.txt
export EDUCODER_COOKIE='从浏览器复制的 Cookie，仅用于当前 shell'
python scripts/submit_educoder.py --dry-run
```

确认输出中的 ZIP、账号、队伍、阶段和历史提交均正确后，新增一条提交记录：

```bash
python scripts/submit_educoder.py --yes
unset EDUCODER_COOKIE
```

脚本不会删除或覆盖历史记录；它拒绝同名重复提交，并要求 ZIP 内恰好有 200 个
`float32 (50000, 3)` 数组。Cookie 只从环境变量读取，禁止写入 `.env`、日志或 Git。

运行配置合同测试：

```bash
python -m unittest tests.test_best_configs -v
```

## 关键实验结果

线上成绩以平台反馈为准；`local2` 是 62 个样本的内部代理集，小于 0.1 的差距不应被当成可靠排序。

| 方法 | 主要变化 | local2 | 线上 | 结论 |
|---|---|---:|---:|---|
| Graph StraightPCF (`spcfg`) | 图卷解码器 | 旧代理 62.87 | 73.71 | 奠定主干 |
| `spcfgm` | velocity modules 2→4 | 旧代理 64.66 | 74.66 | 容量放大有效 |
| C-noise extended | FiLM + 噪声对齐 + 延长训练 | 72.06 | 75.01 | 首次稳定突破 75 |
| MS-A106 | multi-scale distance，alpha=1.06 | 72.15 | 75.47 | P2S 较强 |
| CVM-001 | stage velocity | 72.54 | 74.90 | 单独换 target 不稳定 |
| **CVM-002** | **stage velocity + deep supervision** | **72.77** | **75.51** | **当前最佳** |
| CVM-002 A105 | `predict_alpha=1.05`，单次推理 | 72.82 | 未提交 | local2 +0.05，小于可靠排序阈值；作为低风险提交候选 |
| LDC matched-unroll | 冻结 velocity trunk，学习 distance/stage adapter，并匹配两步展开 | 70.55 | 未提交 | P2S 86.60，但 CD 降至 54.50；否定 |
| CVM-006 | 75% stage-velocity blend | 72.72 | 未提交 | 最值得补线上验证 |
| CVM-008 | multi-scale velocity encoder | 72.60 | 未记录 | 没有超过 CVM-002 |
| CVM-011 | 更强噪声区间 | 71.97 | 未提交 | 简单加大噪声反而退化 |
| PD-001～004 | 逐点 distance gate 系列 | 71.03～71.40 | 未提交 | 当前实现整体失败 |

更完整但已压缩的实验记录见 [EXPERIMENTS.md](EXPERIMENTS.md)。

## 后续最值得尝试

| 优先级 | 方法 | 状态 | 理由与风险 |
|---:|---|---|---|
| 1 | CVM-002 A105 线上验证 | 配置和提交脚本已准备 | local2 72.82，但仅高 0.05；一次线上提交即可判断是否迁移 |
| 2 | 法向/表面感知 endpoint loss | 未实现 | 官方一半分数来自 P2S，数据管线已有法向；比继续加宽网络更直接对齐指标 |
| 3 | 坐标图 + 特征图双图解码器 | 未实现 | 针对薄面跨表面误连和尖锐边过度平滑，属于中风险结构改进 |
| 4 | CVM-006 复训后线上验证 | 历史 local2 72.72，权重已清理 | 验证 stage/full blend 是否比纯 stage target 更稳 |

## 主要经验

1. 图卷解码、module=4、噪声分布对齐和 deep supervision 是已验证的正向因素。
2. attention、普通 TTA、多趟推理、逐点 distance gate 和单纯加宽模型均未带来可靠线上收益。
3. local2 可用于排除明显失败的方法，但不能可靠判断 0.1 分以内的线上排序。
4. 后续实验应优先改变速度场监督和 CD/P2S Pareto，而不是继续堆叠后端模块。
5. distance/stage adapter 的单步验证劣于基线，matched-unroll 又显著损害 CD；后续不再优先投入这条 conditioning 路线。

## 参考文献

- Wu et al., **StraightPCF: Straight Point Cloud Filtering**, CVPR 2024. [arXiv:2405.08322](https://arxiv.org/abs/2405.08322)
- Wang et al., **Dynamic Graph CNN for Learning on Point Clouds**, TOG 2019. [arXiv:1801.07829](https://arxiv.org/abs/1801.07829)
- Perez et al., **FiLM: Visual Reasoning with a General Conditioning Layer**, AAAI 2018. [arXiv:1709.07871](https://arxiv.org/abs/1709.07871)
- de Silva Edirimuni et al., **IterativePFN: True Iterative Point Cloud Filtering**, CVPR 2023. [arXiv:2304.01529](https://arxiv.org/abs/2304.01529)
- Luo and Hu, **Score-Based Point Cloud Denoising**, ICCV 2021. [arXiv:2107.10981](https://arxiv.org/abs/2107.10981)
- Rakotosaona et al., **PointCleanNet: Learning to Denoise and Remove Outliers from Dense Point Clouds**, CGF 2020. [arXiv:1901.01060](https://arxiv.org/abs/1901.01060)
- Zhang et al., **Pointfilter: Point Cloud Filtering via Encoder-Decoder Modeling**, TVCG 2021. [arXiv:2002.05968](https://arxiv.org/abs/2002.05968)
- Luo and Hu, **Differentiable Manifold Reconstruction for Point Cloud Denoising**, ACM MM 2020. [arXiv:2007.13551](https://arxiv.org/abs/2007.13551)
- Zhao et al., **Point Transformer**, ICCV 2021. [arXiv:2012.09164](https://arxiv.org/abs/2012.09164)
