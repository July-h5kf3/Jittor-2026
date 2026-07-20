# Track2 点云去噪（Jittor）

仓库只保留当前最佳方案、复现实验所需代码和影响决策的关键结论。数据集、checkpoint、预测、提交包和编译 cache 只保存在训练服务器，不进入 Git。

截至 2026-07-20：

- 当前研究目标：在只使用官方数据并保持 Jittor 可复现的前提下，从 **77.87** 继续推进到 **85+**。
- 线上最佳：**77.87**，ROT-001 单 checkpoint + mean/var + **CV-adaptive two-pass**；CD/P2S = **65.91 / 89.83**，相对上一最佳 76.04 提升 **+1.83**（request_id `2026071921373714762837`）。
- 上一线上最佳：**76.04**，CVM-002 A105 与独立重训模型 1:1 输出集成，再做 mean/var 校准；CD/P2S = **64.60 / 87.47**（submission `39221`）。
- 当前 local2 最好：ROT-001 与 PNX-001 的 CV two-pass 按 `70/30` 固定输出集成，**73.83070**；相对 ROT 单模 73.75323 提高 `+0.07747`，95% CI `[+0.03542,+0.12108]`，LOCO 全正。它仍是离线候选，线上最佳保持 77.87。
- Educoder 提交脚本已完成多次真实上传，支持结果等待、临时 502 重试和不确定回调去重。

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
| 上一线上集成 | 原最佳与独立重训 raw 预测 1:1 平均，再做 mean/var | 线上 76.04；CD/P2S = 64.60/87.47 |
| 当前线上最佳 | ROT-001 单 checkpoint + mean/var + CV-adaptive two-pass | 用第二次修正模长的 `std/mean` 调 beta；线上 77.87，CD/P2S = 65.91/89.83 |

核心配置：

- `configs/task/train_spcfgfncvm002_cvm.yaml`
- `configs/task/train_spcfgfncvm002.yaml`
- `configs/task/predict_spcfgfncvm002a105.yaml`
- `configs/model/spcfgfncvm002_{cvm,spcf}.yaml`
- `configs/model/spcfgfncvm002a105_spcf.yaml`
- `scripts/calibrate_predictions.py`
- `scripts/ensemble_predictions.py`
- `scripts/calibrate_two_pass.py`
- `scripts/submit_educoder.py`

## 服务器保留产物

以下路径被 `.gitignore` 排除：

| 用途 | 路径 |
|---|---|
| 官方 75.01 初始化权重 | `experiments/_bak_official_75.01/cvm_checkpoint_best.pkl` |
| CVM-002 最佳 CVM 权重 | `experiments/spcfgfncvm002_cvm/checkpoint_best.pkl` |
| CVM-002 最佳完整权重 | `experiments/spcfgfncvm002_spcf/checkpoint_best.pkl` |
| 独立重训 raw 权重 | `experiments/spcfgfncvm002ema_spcf/checkpoint_best.pkl` |
| ROT-001 最佳完整权重 | `experiments/spcfgfnrot001_spcf/checkpoint_best.pkl` |
| PNX-001 最佳完整权重 | `experiments/spcfgfnpnx001_spcf/checkpoint_best.pkl` |
| 线上 77.87 提交包 | `submission_results/result_rot001_a105_cv_twopass_g050_20260719.zip` |
| 线上 77.87 提交包 SHA256 | `1cdf56febb033f1a24d1bc6456933cb4584c7532c3126a7eaeb54d3a597369da` |
| ROT/PNX 70/30 离线候选包 | `submission_results/result_rot001_pnx001_twopass_ens_p030_20260720.zip` |
| ROT/PNX 70/30 候选包 SHA256 | `ebcec509073ac86078a08639441e6c6ec1ed6c713b92575b04bf4b1a149b3279` |
| 线上 76.04 提交包 | `submission_results/result_baseline_retrain_ensemble_adaptive.zip` |
| 线上 76.04 提交包 SHA256 | `e8bce43d21a458ab16b621713f96a141b2cbbc6364c8671a84858f2b1f25fad6` |
| 上一版候选 | `submission_results/result_baseline075_retrain025_cv_twopass_g050.zip` |
| 上一版候选 SHA256 | `8b171a4e92a18a1360d10b86783c02c302bf01235e5919ad9a0809a3e719c272` |

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

独立 20 云上 CD 与 P2S 同时提高。该结果用于建立 mean/var 校准基线；它曾支撑 baseline/retrain 1:1 输出集成取得 76.04，当前线上最好已更新为 ROT-001 CV-adaptive two-pass 的 77.87。

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

已完成真实上传。当前关键记录：`39221=76.04`；ROT-001 request_id `2026071921373714762837=77.87`；`39222` 长时间停留待计算；`39225` 因团队每日两次提交上限被平台拒绝。不要重复上传仍处于 `status=0` 的记录。

## 关键实验汇总

近期严格 control：raw `72.82574093`，mean/var adaptive `73.00041870`。

| 方法 | adaptive 总分 | 相对 control | 结论 |
|---|---:|---:|---|
| **mean/var 自适应校准** | **73.00042** | OOF **+0.12364** | 当前 ensemble/two-pass 的基础校准 anchor |
| **1:1 独立轨迹输出集成 + mean/var** | **73.03443** | +0.03852 | 线上 76.04，已确认微小提升 |
| **75/25 集成 + fixed two-pass** | **73.58214** | 相对 1:1 two-pass +0.00811 | 200 云包已生成；今日额度已满 |
| **75/25 集成 + conservative two-pass** | **73.62371** | 相对 fixed +0.04157 | 上一 local2 最好；已被 CV 调度稳定超过 |
| **75/25 集成 + CV-adaptive two-pass** | **73.67212** | 相对 conservative +0.04841 | CD/P2S 同升；95% CI `[+0.01172,+0.08920]`，上一版已校验候选 |
| FPS 起点变化自集成 + CV two-pass | 73.64981 | -0.02231 | 95% CI 跨零，否定并删除专用实现 |
| baseline/retrain/seed456 = 2:1:1 + clipped-alpha gate | 73.68104 | +0.00892 | 单云在 0.97 裁剪边界触发不连续，旧结果不再作为候选 |
| **baseline/retrain/seed456 = 55/10/35 + raw-alpha gate** | **73.71935** | **+0.04723** | CD/P2S 同升；95% CI `[+0.02507,+0.07160]`，12 类 leave-one-out 全正 |
| + seed789 第四轨迹（10%） | 73.72168 | +0.00233 | CI 为正但远低于 +0.05；不增加 A 榜候选复杂度 |
| multi-EMA 0.99/0.995/0.999 | 72.90402～72.90787 | -0.08803～-0.09188 | 三个 decay 均否定；独立 raw 仅用于集成 |
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
| PNX-001 单模 CV two-pass | 73.48412 | 相对 ROT -0.26912 | P2S 稳定回撤，单模否定 |
| **ROT/PNX 70/30 two-pass 集成** | **73.83070** | **相对 ROT +0.07747** | CD +0.15606、P2S 基本不变；95% CI 与 LOCO 全正，保留离线候选 |

完整数值和置信区间见 [EXPERIMENTS.md](EXPERIMENTS.md)。

PNX-001 已完成四卡训练与 canonical local2 评测：其 CV two-pass 为 `73.48412`，相对 ROT 稳定下降 `-0.26912`（95% CI `[-0.40457,-0.13796]`），因此单模否定；但 70/30 ROT-PNX 集成达到 `73.83070`，相对 ROT 提高 `+0.07747`，作为架构多样性候选保留。

## 后续最值得尝试

| 优先级 | 方向 | 当前状态 | 理由 |
|---:|---|---|---|
| 1 | UNI-001 旋转不变局部间距损失 | 已实现并通过 CPU/CUDA smoke，待四卡训练 | 训练期匹配 clean-kNN 邻距以改善 CD 覆盖；不约束边方向，避免重复 edge-vector 与手工 repulsion 的 P2S 回撤 |
| 2 | PNX-001 架构多样性集成 | 单模否定；70/30 候选包已验证 | local2 相对 ROT `+0.07747`，CI 与 LOCO 全正；200 云 ZIP 已严格校验，必须取得用户批准后才能提交 |
| 3 | 法向/曲率域消息传递 | 部分探索 | 纯坐标双图已否定；后续若尝试，应显式构造切平面或曲率邻接 |
| 4 | U-CAN / Noise2Noise 一致性预训练 | 未尝试 | 可利用 noisy-only 数据扩大分布，但训练成本较高 |
| 5 | 55/10/35 + raw-alpha gate | 离线保留，不优先消耗提交机会 | local2 低于 ROT two-pass；作为后续集成多样性素材保留 |
| 6 | CVM-006 线上补测 | 尝试但未提交 | 历史 local2 72.72，优先级低于当前 77.87 主方案 |

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
- Koo et al., **P2P-Bridge: Diffusion Bridges for 3D Point Cloud Denoising**. [arXiv:2408.16325](https://arxiv.org/abs/2408.16325)
- **Point Cloud Resampling with Learnable Heat Diffusion**. [arXiv:2411.14120](https://arxiv.org/abs/2411.14120)
- Izmailov et al., **Averaging Weights Leads to Wider Optima and Better Generalization**. [arXiv:1803.05407](https://arxiv.org/abs/1803.05407)
- Bahri et al., **Test-Time Adaptation in Point Clouds: Leveraging Sampling Variation with Weight Averaging**. [arXiv:2411.01116](https://arxiv.org/abs/2411.01116)
- **GD-GCN: Geometry-Driven Graph Convolutional Network for Point Cloud Denoising**. [arXiv:2411.14158](https://arxiv.org/abs/2411.14158)
- **UGD: Unsupervised Point Cloud Denoising via a Learned Pristine Geometry Prior**. [arXiv:2604.16976](https://arxiv.org/abs/2604.16976)
- **PQDT: Pseudo-Query Dual Transformer for Point Cloud Denoising**. [arXiv:2605.25127](https://arxiv.org/abs/2605.25127)
