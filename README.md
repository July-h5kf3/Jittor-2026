# Track2 点云去噪（Jittor）

仓库只保留当前最佳方案、复现实验所需代码和影响决策的关键结论。数据集、checkpoint、预测、提交包和编译 cache 只保存在训练服务器，不进入 Git。

截至 2026-07-14：

- 线上最佳：**76.03**，CVM-002 A105，CD/P2S = **64.57 / 87.49**。
- 最佳离线提交候选：CVM-002 A105 + **按云 mean/var 自适应位移校准**。
- Educoder 提交脚本已经完成真实 API dry-run；尚未执行上传。

## 当前最佳方案

| 部分 | 最终选择 | 说明 |
|---|---|---|
| 主干 | StraightPCF，4 个 velocity modules | 16GB V100 上验证过的容量上限 |
| 编码/解码 | EdgeConv encoder + graph decoder | 最稳定、最大的结构收益来源 |
| 条件 | FiLM | 注入噪声/阶段相关条件 |
| CVM 目标 | `stage_velocity` + deep supervision | CVM-002 超过早期 CVM 的关键 |
| Distance head | multi-scale encoder | 改善剩余步长估计 |
| 训练噪声 | sigma=`0.008～0.014`，含 2% 三倍 Laplace outlier | 比简单扩大模型更有效 |
| 基础推理 | 单次，`predict_alpha=1.05`，无 TTA/融合 | 对应线上 76.03 |
| 最佳离线后处理 | 每云 mean/var Ridge，alpha 裁剪到 `[0.97,1.10]` | 不读取 clean/mesh；独立集 CD/P2S 同升 |

核心配置：

- `configs/task/train_spcfgfncvm002_cvm.yaml`
- `configs/task/train_spcfgfncvm002.yaml`
- `configs/task/predict_spcfgfncvm002a105.yaml`
- `configs/model/spcfgfncvm002_{cvm,spcf}.yaml`
- `configs/model/spcfgfncvm002a105_spcf.yaml`
- `scripts/calibrate_predictions.py`

## 服务器保留产物

以下路径被 `.gitignore` 排除：

| 用途 | 路径 |
|---|---|
| 官方 75.01 初始化权重 | `experiments/_bak_official_75.01/cvm_checkpoint_best.pkl` |
| CVM-002 最佳 CVM 权重 | `experiments/spcfgfncvm002_cvm/checkpoint_best.pkl` |
| CVM-002 最佳完整权重 | `experiments/spcfgfncvm002_spcf/checkpoint_best.pkl` |
| 最佳离线提交包 | `submission_results/result_cvm002a105_adaptive_meanvar_a110.zip` |
| 最终每云 alpha 清单 | `submission_results/cvm002a105_adaptive_meanvar_a110.tsv` |
| 提交包 SHA256 | `073f4b23bed59ce5dc237a6a2e4aaba7d17e13f4b2e68843532ca98e71c30564` |

## 环境与运行

```bash
conda create -n jittor python=3.8 -y
conda activate jittor
pip install -r requirements.txt
```

训练数据约定：

```text
dataset_train/shapenet/<synset>/<model>/models/model_normalized.obj
dataset_test_noisy/shapenet/<synset>/<model>/noisy.npy
```

训练并生成基础预测：

```bash
GPU_LIST=0,1,2,3 NP=4 bash scripts/run_best_pipeline.sh
GPU=0 bash scripts/package_submission.sh
```

服务器根分区空间很小。多卡启动器会把 OpenMPI 临时目录放到 `/dev/shm`，并支持 `JITTOR_CACHE_PER_RANK=1`，避免多个 rank 同时编译新算子时争用同一 cache。

## 自适应校准与打包

校准器只使用每云预测位移模长的均值和方差：

```bash
python scripts/calibrate_predictions.py \
  --pred-dir submission_results/raw_prediction/shapenet \
  --noisy-dir dataset_test_noisy/shapenet \
  --out-dir submission_results/adaptive_prediction/shapenet \
  --profile mean-var \
  --expected-count 200

SKIP_PREDICT=1 \
OUT_DIR=submission_results/adaptive_prediction \
ZIP_PATH=submission_results/result_adaptive.zip \
bash scripts/package_submission.sh
```

验证结果：

| 口径 | 固定/原始 | adaptive | 增益 |
|---|---:|---:|---:|
| local2 五折留出 | 72.87189 | 72.99553 | **+0.12364** |
| 独立 20 云 | 77.35058 | 77.76446 | **+0.41388** |

独立 20 云上 CD 与 P2S 同时提高。当前 ZIP 仍是离线候选，平台确认前线上最好仍记为 76.03。

## Educoder 提交脚本

依赖安装到数据盘，Cookie 只通过当前 shell 的环境变量传入：

```bash
export SUBMIT_DEPS=/root/data-tmp/submit_deps
export TMPDIR=/root/data-tmp/tmp
mkdir -p "$SUBMIT_DEPS" "$TMPDIR"

python -m pip install --no-cache-dir --upgrade \
  --target "$SUBMIT_DEPS" -r requirements-submit.txt

export EDUCODER_COOKIE='从浏览器复制的 Cookie，仅用于当前 shell'
PYTHONPATH="$SUBMIT_DEPS" python scripts/submit_educoder.py --dry-run
```

脚本会在上传前验证：

- ZIP 恰好包含 200 个 `denoised.npy`；
- 每个数组为 `float32 (50000, 3)` 且数值有限；
- 账号、队伍、赛段、历史文件名和 OSS 临时令牌均有效；
- `--dry-run` 不上传，真实上传必须显式传入 `--yes`。

确认后才可执行：

```bash
PYTHONPATH="$SUBMIT_DEPS" python scripts/submit_educoder.py --yes
unset EDUCODER_COOKIE
```

当前没有执行真实上传。

## 关键实验汇总

近期严格 control：raw `72.82574093`，mean/var adaptive `73.00041870`。

| 方法 | adaptive 总分 | 相对 control | 结论 |
|---|---:|---:|---|
| **mean/var 自适应校准** | **73.00042** | OOF **+0.12364** | 当前最佳离线改进 |
| Normal auxiliary | 72.99793 | -0.00249 | CD 小升、P2S 下降 |
| SIMPC mirror consistency | 72.96870 | -0.03172 | 覆盖与表面距离 Pareto 变差 |
| HybridPF short residual | 72.93875 | -0.06167 | CD/P2S 均下降 |
| ROB010 Huber endpoint | 72.99279 | -0.00763 | 难点梯度被过度裁剪 |
| CORE256 中心监督 | 72.99724 | -0.00317 | 硬掩码忽略真实拼接输出 |
| CORE384 中心监督 | 73.00200 | +0.00158 | 96.5% 覆盖仍远低于保留门槛 |
| TSTRATA 端点分层采样 | 72.99515 | -0.00527 | P2S 上升但 CD 下降 |
| feature + position 双图 | 72.98807 | -0.01235（V5 内部 -0.00693） | 严格同配方 control 下无收益，删除 |
| UGD-lite pristine GMM | 72.70767 | -0.29275（V5 mean/var -0.28733） | 21/62 云选 alpha 下界，明显过度收缩 |
| bilateral IMLS | OOF 72.86087 | -0.01102 | 表面损失无独立收益 |
| TV-PC / geometry / tangent repulsion | - | - | 过平滑、过拟合或 CD/P2S 互换，均删除 |

完整数值和置信区间见 [EXPERIMENTS.md](EXPERIMENTS.md)。

## 后续最值得尝试

| 优先级 | 方向 | 当前状态 | 理由 |
|---:|---|---|---|
| 1 | 不确定性驱动的每云/每 patch 步长 | 未尝试 | mean/var 是目前唯一跨口径稳定迁移的增益，可进一步学习置信度 |
| 2 | 曲率感知、可学习的点分布项 | 未尝试 | 手工 repulsion 能提高 CD，但必须联合守住 P2S |
| 3 | 法向/曲率域消息传递 | 部分探索 | 纯坐标双图已否定；后续若尝试，应显式构造切平面或曲率邻接 |
| 4 | U-CAN / Noise2Noise 一致性预训练 | 未尝试 | 可利用 noisy-only 数据扩大分布，但训练成本较高 |
| 5 | CVM-006 线上补测 | 尝试但未提交 | 历史 local2 72.72，优先级低于当前 adaptive ZIP |

## 测试

```bash
python -m unittest tests.test_best_configs -v
python -m unittest tests.test_adaptive_alpha tests.test_submit_educoder -v
git diff --check
```

## 参考文献

- Wu et al., **StraightPCF: Straight Point Cloud Filtering**, CVPR 2024. [arXiv:2405.08322](https://arxiv.org/abs/2405.08322)
- Wang et al., **Dynamic Graph CNN for Learning on Point Clouds**, TOG 2019. [arXiv:1801.07829](https://arxiv.org/abs/1801.07829)
- Perez et al., **FiLM: Visual Reasoning with a General Conditioning Layer**, AAAI 2018. [arXiv:1709.07871](https://arxiv.org/abs/1709.07871)
- de Silva Edirimuni et al., **IterativePFN: True Iterative Point Cloud Filtering**, CVPR 2023. [arXiv:2304.01529](https://arxiv.org/abs/2304.01529)
- Luo and Hu, **Score-Based Point Cloud Denoising**, ICCV 2021. [arXiv:2107.10981](https://arxiv.org/abs/2107.10981)
- Wang et al., **Adaptive and Iterative Point Cloud Denoising with Score-Based Diffusion Model**. [arXiv:2509.14560](https://arxiv.org/abs/2509.14560)
- Zhang et al., **SIMPC: Learning Self-Induced Mirror-Point Consistency for Unsupervised Point Cloud Denoising**. [arXiv:2605.26894](https://arxiv.org/abs/2605.26894)
- de Silva Edirimuni et al., **Hybrid Long and Short Range Flows for Point Cloud Filtering**. [arXiv:2508.08542](https://arxiv.org/abs/2508.08542)
- Wei et al., **Noise2Score3D: Tweedie's Approach for Unsupervised Point Cloud Denoising**. [arXiv:2503.09283](https://arxiv.org/abs/2503.09283)
- Zhou et al., **U-CAN: Unsupervised Point Cloud Denoising with Consistency-Aware Noise2Noise Matching**. [arXiv:2510.25210](https://arxiv.org/abs/2510.25210)
- Li et al., **Learning Normals of Noisy Points by Local Gradient-Aware Surface Filtering**. [arXiv:2507.03394](https://arxiv.org/abs/2507.03394)
- Xu et al., **Gradient-based Point Cloud Denoising with Uniformity**. [arXiv:2207.10279](https://arxiv.org/abs/2207.10279)
- Na et al., **A Lennard-Jones Layer for Distribution Normalization**. [arXiv:2402.03287](https://arxiv.org/abs/2402.03287)
- **GD-GCN: Geometry-Driven Graph Convolutional Network for Point Cloud Denoising**. [arXiv:2411.14158](https://arxiv.org/abs/2411.14158)
- **UGD: Unsupervised Point Cloud Denoising via a Learned Pristine Geometry Prior**. [arXiv:2604.16976](https://arxiv.org/abs/2604.16976)
- **PQDT: Pseudo-Query Dual Transformer for Point Cloud Denoising**. [arXiv:2605.25127](https://arxiv.org/abs/2605.25127)
