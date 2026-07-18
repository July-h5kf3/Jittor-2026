# Track2 点云去噪（Jittor）

仓库只保留当前最佳方案、复现实验所需代码和影响决策的关键结论。数据集、checkpoint、预测、提交包和编译 cache 只保存在训练服务器，不进入 Git。

截至 2026-07-14：

- 线上最佳：**76.04**，CVM-002 A105 与独立重训模型 1:1 输出集成，再做 mean/var 校准；CD/P2S = **64.60 / 87.47**（submission `39221`）。
- 当前 local2 最好：baseline/retrain/seed456=`55/10/35` 输出集成 + mean/var + **raw-alpha gate + CV-adaptive two-pass**，**73.71935**。seed789 第四轨迹已完成，最佳四轨迹扫描只到 73.72168（+0.00233），未达到保留门槛。
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
| 线上确认集成 | 原最佳与独立重训 raw 预测 1:1 平均，再做 mean/var | 线上 76.04；CD 提升 0.03，P2S 下降 0.02 |
| 待线上验证 | 55/10/35 三轨迹集成 + raw-alpha gate + CV-adaptive two-pass | 用第二次修正模长的 `std/mean` 调 beta；local2 73.71935，CD/P2S 同升 |

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

独立 20 云上 CD 与 P2S 同时提高。该结果用于建立 mean/var 校准基线；当前已确认的线上最好为 baseline/retrain 1:1 输出集成 + mean/var，成绩 76.04（submission 39221）。

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

已完成真实上传。当前关键记录：`39221=76.04`；`39222` 长时间停留待计算；`39225` 因团队每日两次提交上限被平台拒绝。不要重复上传仍处于 `status=0` 的记录。

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

完整数值和置信区间见 [EXPERIMENTS.md](EXPERIMENTS.md)。

## 后续最值得尝试

| 优先级 | 方向 | 当前状态 | 理由 |
|---:|---|---|---|
| 1 | 55/10/35 + raw-alpha gate 线上验证 | 待打包 | AID 的方差调度启发；local2 CD/P2S 同升，独立 local3 保持保护行为 |
| 2 | ROT-001 真实 sampled-point SO(3) 增强 | 四卡训练中 | 发现历史线性增强只变换 mesh、对训练点实际为空操作；先隔离旋转变量 |
| 3 | PointNeXt-lite 层次化局部残差编码器 | 候选设计 | 普通 attention 已否定；层次化局部聚合更直接针对 CD 覆盖和 B 榜扩展性 |
| 4 | 曲率感知、可学习的点分布项 | 未尝试 | 手工 repulsion 能提高 CD，但必须联合守住 P2S |
| 5 | 法向/曲率域消息传递 | 部分探索 | 纯坐标双图已否定；后续若尝试，应显式构造切平面或曲率邻接 |
| 6 | U-CAN / Noise2Noise 一致性预训练 | 未尝试 | 可利用 noisy-only 数据扩大分布，但训练成本较高 |
| 7 | CVM-006 线上补测 | 尝试但未提交 | 历史 local2 72.72，优先级低于当前 adaptive ZIP |

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
