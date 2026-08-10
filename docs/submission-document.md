# 第六届计图人工智能挑战赛 赛题二 A榜代码提交说明文档

## 一、团队信息与 A 榜成绩

- 队伍名称：**NKAI**
- A 榜排名：**31**（提交文件名使用三位字段 `031`）
- A 榜最优成绩：**79.23 / 67.42 / 91.04（Total / CD / P2S）**
- 主要联系人：李佳璞，电话/微信：18883873159
- 团队成员：刘迪乘，电话/微信：18570499315
- 团队成员：曲恒睿，电话/微信：17606383128

最终代码检查文件名：`contest2_NKAI_031.zip`。

## 二、项目概述

本项目解决赛题二 50K 三维点云降噪。主体模型为 Jittor 原生四阶段 3DMambaIPF 风格网络，使用 EdgeConv 动态图提取局部几何，使用 selective state-space block 建模长程上下文，并逐阶段预测 displacement。最终 A 榜算法 PLR-003 由随机初始化训练得到的 Full 基线、类别专家、StraightPCF ROT noisy-only reference，以及三层冻结路由组成。

核心组件：

1. Jittor 原生 3DMambaIPF：四阶段 point displacement refinement；
2. `jt.code` selective scan CUDA forward/backward：不调用 PyTorch、Triton、`mamba_ssm` 或 `causal_conv1d`；
3. mixed-domain、pure-Laplace 与 density-aware Chamfer continuation：构建 DCD、table、sofa、airplane 和七个 tail 类别专家；
4. Jittor StraightPCF：从随机初始化执行 VM → CVM → SPCF 三阶段训练，并在推理时做 pass-1、mean-var calibration、pass-2 和 CV-adaptive fusion；
5. 冻结 noisy-only 路由：PLR-001 类别专家、PLR-002 table 1.25 倍外推、PLR-003 tail 标准门控；
6. 输出强制保持 50,000 点、原索引顺序、finite 和 float32。

系统不读取测试集 clean point cloud、mesh、ground truth、隐藏答案或官方指标。

## 三、提交包结构

```text
contest2_NKAI_031.zip
├── code/
│   ├── main.py                       # 统一 doctor/delegate/pipeline 入口
│   ├── train.py                      # 3DMambaIPF 随机初始化与 continuation
│   ├── train_rot.py                  # StraightPCF VM/CVM/SPCF 原生训练
│   ├── infer.py                      # 3DMambaIPF 50K patch inference
│   ├── generate_rot.py               # adaptive two-pass ROT
│   ├── assemble_incumbent.py         # MBI-009 等价 incumbent
│   ├── assemble_full.py              # Full noisy-only route
│   ├── assemble.py                   # PLR-001/002/003 route
│   ├── package_results.py            # canonical prediction ZIP
│   ├── plr3d/models/                 # Mamba、EdgeConv、DenoiseNet
│   ├── plr3d/ops/                    # KNN/FPS、selective scan CUDA
│   ├── plr3d/data/、plr3d/losses.py  # patch 数据与 DCD loss
│   ├── rot_jittor/                   # 精简 Jittor StraightPCF 模型
│   ├── configs/reproduce_full.json   # 39-stage 从原始数据复现链
│   ├── configs/lists/                # 公开训练 key 与确定性 seed
│   ├── tools/                        # 数据准备、key、审计工具
│   └── tests/                        # source policy、route、CUDA smoke
├── requirements.txt                  # 最小正式 runtime
├── environment.yaml                  # 组委会完整检查环境
└── 提交说明文档.pdf                  # 本文档
```

代码包不含 `.pth/.pt/.ckpt/.pkl/.npy/.npz` 权重、数据或预测，也不含日志、缓存、Cookie、SSH key、密码或私有绝对路径。

## 四、环境配置

目标检查环境：

- Ubuntu 22.04；
- NVIDIA RTX 4090 24GB（sm_89）；
- CUDA Toolkit 12.4；
- cuDNN 8.9.7；
- Python 3.10.14；
- Jittor 1.3.11.0（满足 Jittor `>=1.3.10`）；
- NumPy 1.24.4。

最小 Python 安装（系统需已提供 CUDA 12.4 与 cuDNN 8.9）：

```bash
conda create -n nkai-track2 python=3.10.14 -y
conda activate nkai-track2
pip install -r requirements.txt
```

组委会完整检查环境：

```bash
conda env create -f environment.yaml
conda activate nkai-track2-plr003
```

建议将 Conda、pip 和 Jittor cache 放入大容量数据盘：

```bash
export CONDA_PKGS_DIRS=/root/autodl-tmp/conda-pkgs
export PIP_CACHE_DIR=/root/autodl-tmp/pip-cache
export JITTOR_HOME=/root/autodl-tmp/jittor-cache
export cache_path=/root/autodl-tmp/jittor-cache
```

环境与 CUDA 算子自检：

```bash
cd code
python main.py doctor
python main.py doctor --deep
```

`main.py` 会自动兼容 Conda CUDA Toolkit 布局：把环境 `bin/` 加入 `PATH`，并在需要时创建 `lib64 -> lib` 链接，供 Jittor 1.3.11 定位 `libcudart.so`。

`requirements.txt` 只列实际正式 Python 依赖 Jittor 与 NumPy。`environment.yaml` 额外固定 CUDA Toolkit 12.4、cuDNN 8.9.7，以及赛事示例检查环境中的 Pandas、SciPy、scikit-learn 和 JittorGeometric；正式算法不调用这些额外 Python 包。

## 五、原始数据格式

训练根目录必须包含原始 ShapeNet mesh：

```text
<TRAIN_ROOT>/shapenet/<synset>/<model>/models/model_normalized.obj
```

若数据附带 `model_normalized.obj.cache.npz`，程序优先读取其中的 vertices/faces 以保持历史 mesh 顺序；无 cache 时使用标准库 OBJ parser。

测试根目录：

```text
<TEST_ROOT>/shapenet/<synset>/<model>/noisy.npy
```

正式测试集要求 200 个唯一 key，每个 noisy array 为 finite `float32 (50000,3)`。输出路径为相同 key 下的 `denoised.npy`。

随包 `configs/lists/train_*.txt` 仅包含公开训练集 key，用于固定 A 榜训练划分；不包含点坐标、标签、预测或权重。

## 六、完整训练与推理命令

设置外部原始数据和大容量工作目录：

```bash
cd code
export NKAI_TRAIN_ROOT=/data/track2/dataset_train
export NKAI_TEST_ROOT=/data/track2/dataset_test
export NKAI_WORK_ROOT=/root/autodl-tmp/nkai-track2-reproduction
```

先查看 39 个 stage 的完整展开命令：

```bash
python main.py pipeline \
  --config configs/reproduce_full.json \
  --log-dir "$NKAI_WORK_ROOT/logs" \
  --dry-run
```

正式执行从随机初始化到预测 ZIP 的完整复现：

```bash
python main.py doctor --deep
python main.py pipeline \
  --config configs/reproduce_full.json \
  --log-dir "$NKAI_WORK_ROOT/logs"
```

最终结果：

```text
$NKAI_WORK_ROOT/result_plr003.zip
$NKAI_WORK_ROOT/manifests/result_zip_audit.json
$NKAI_WORK_ROOT/manifests/source_policy_audit.json
```

Pipeline 的 `skip_if_exists` 指向完成 checkpoint 或 manifest，重复执行会跳过完成阶段。StraightPCF 训练使用 `--resume`，可继续未完成的 VM/CVM/SPCF stage。

## 七、训练链与关键超参数

### 7.1 训练数据构建

从原始 OBJ 确定性采样 50K clean cloud：

- base：15,733 个训练 mesh；
- MBI-009：650（13 类各 50）；
- DCD-001：3,000；
- table：1,200；
- airplane：1,000；
- sofa：1,000；
- tail：1,885（七类）。

### 7.2 3DMambaIPF 链

共同设置：`patch_size=1000`、`patch_ratio=1.2`、每卡 micro-batch 4、gradient clip 1.0、四个 denoise modules。Continuation 单卡使用 `grad_accum_steps=4`（effective batch 16）；base 使用 `grad_accum_steps=8`（effective batch 32）。

主要 stage：

- base-random：100 epochs，15,733×4 samples，`lr=1e-4`，mixed Gaussian/Laplace，DCD weight 0；
- MBI-009：3 epochs，650×4 samples，`lr=1e-5`，Gaussian probability 0.5，outlier fraction 0.02；
- DCD-001：2 epochs，3,000×1 samples，`lr=5e-6`，DCD weight 0.5；
- MBI-011 table：2 epochs，1,200×2 samples，`lr=5e-6`；
- TSD-003：2 epochs，`lr=2e-6`，DCD weight 0.5；
- TSD-004 strong：2 epochs，`lr=1e-6`，DCD weight 1.0；
- SSD-009 strong：2 epochs，`lr=2e-6`，DCD weight 1.0；
- PLR airplane/table/sofa：2 epochs，pure Laplace、无 outlier、`lr=1e-6`；
- PLR tail：2 epochs，1,885×4 samples，七类冻结 sigma map、DCD weight 0.5。

Airplane sigma `[0.012489692407870212, 0.014616960844581714]`；table sigma `[0.009048618504872253, 0.012903604290095811]`；sofa sigma `[0.008374454205880252, 0.011346570064822026]`。Tail 精确 sigma 位于 `configs/plr003_tail_sigma.json`。

### 7.3 StraightPCF ROT

`train_rot.py` 从 mesh 随机初始化训练 VM → CVM → SPCF：

- 默认 epochs：3 / 30 / 40；
- 每 epoch 10,000 samples；
- batch size 32；
- surface samples 32,768，其中最多 1,024 个原 mesh vertices；
- train noise sigma `[0.008,0.014]` Laplace，2% outliers，outlier scale 3；
- train patch 1,000；
- Adam `lr=1e-4`；
- validation sigma 0.011；
- VM best 初始化四个 CVM velocity modules；CVM best 初始化 SPCF velocity modules。

### 7.4 50K 推理

3DMambaIPF 专家：`patch_size=2000`、`seed_k=6`、`seed_k_alpha=20`、`num_modules=4`、seed 2020。每个 50K cloud 使用 150 个 FPS seed，按归一化 seed 距离为每个原点选择唯一 patch prediction。

ROT 执行 StraightPCF pass-1、mean/variance adaptive alpha、pass-2、CV robust score 和 `gamma=0.5` residual fusion。

## 八、Full 与 PLR-003 冻结路由

标准门控：

```text
ratio = q95(||member-noisy|| / max(||ROT-noisy||, float64_tiny))
ratio <= 4.0:
    output = float32(0.25*ROT + 0.75*member)
else:
    output = ROT
```

- Full：DCD direct 五类、table、sofa 专家与 incumbent fallback；
- Full sofa：分别门控 DCD source 与 SSD strong，再计算 `source + 1.25*(strong-source)`；
- PLR-001：airplane/table/sofa 专家，其余回退 Full；
- PLR-002：table 计算 `Full + 1.25*(PLR001-Full)`；
- PLR-003：七个 tail 类别走标准门控，其余回退 PLR-002。

历史 MBI-020 后续分支在最终 Full/PLR 可达图中全部被覆盖，因此 `assemble_incumbent.py` 按 dead-branch pruning 只重建最终可达的 MBI-009 noisy-only incumbent，不改变最终可达输出。

## 九、预测 ZIP 生成与审计

完整 pipeline 自动执行；也可单独运行：

```bash
python main.py package-results -- \
  --root "$NKAI_WORK_ROOT/predictions/plr003" \
  --key-list "$NKAI_WORK_ROOT/lists/test_all.txt" \
  --zip "$NKAI_WORK_ROOT/result_plr003.zip" \
  --audit "$NKAI_WORK_ROOT/manifests/result_zip_audit.json" \
  --expected-count 200 --expected-points 50000
```

ZIP 内部必须严格为：

```text
shapenet/<synset>/<model>/denoised.npy
```

成员按 key 排序、唯一、无目录穿越；每个数组为 finite `float32 (50000,3)`。audit 记录 ZIP SHA256、大小、每个成员 file SHA256 与 array SHA256。

## 十、验证与一致性

已完成的核心验证：

- selective scan CUDA forward max error `3.7252903e-09`，gradient max error `1.8626451e-09`；
- Mamba max error `4.7683716e-07`；
- FeatureExtraction max error `2.6822090e-07`；
- 四阶段 denoise max error `1.8998980e-07`；
- patch/stitch max error `2.7567148e-07`；
- training loss difference `9.3132257e-10`；
- sampled gradient max error `4.6938658e-07`；
- Full official200 assembly 与冻结 A 榜 Full 输入 200/200 文件逐字节一致；
- PLR-003 official200 assembly 与实际 A 榜提交 200/200 文件逐字节一致。

- Ubuntu 22.04 / Python 3.10.14 / CUDA 12.4.99 / cuDNN 8.9.7 / RTX 4090：`doctor --deep` 通过；
- 4090 上通过随机初始化训练一步、DCD 训练一步、gradient accumulation、checkpoint save/load；
- 4090 上通过 StraightPCF VM→CVM→SPCF 三阶段训练与 resume；
- 4090 上通过真实 50K 3DMambaIPF inference、真实 50K adaptive two-pass ROT、assembly 和 prediction ZIP packaging。

目标环境的命令、耗时、SHA、磁盘占用和已修复问题详见 `code/VALIDATION.md`。

## 十一、已知问题与复现边界

1. 权重、数据和预测不随代码包提交，必须运行完整训练链生成；
2. 首次运行会编译 Jittor CUDA 算子，应将 cache 放在大容量数据盘；
3. 完整 surface 数据约 15GB，建议工作分区至少预留 60GB；
4. selective scan 仅支持 float32 和 1～32 个 SSM states，冻结模型使用 16；
5. 动态 KNN near-tie、scatter/CUDA reduction、随机初始化与 GPU 架构差异可能导致 checkpoint 和 prediction 不能逐字节相同；算法、结构、数据划分、超参数和冻结 route 保持一致；
6. 所有输出仍必须通过 canonical shape/dtype/index/finite 和 route manifest 审计；
7. 不得读取测试 ground truth、clean cloud 或 mesh；
8. 4090 验收覆盖每种正式路径，但未等待完整 100-epoch base、全部 continuation 和 803 个 50K inference 结束；79.23 为历史实际 A 榜成绩，完整新重训指标只能由官方评测确认；
9. B 榜阶段继续使用本文披露的算法、模型与路由，不做未披露修改。

## 十二、来源与许可证

模型学术来源包括 3DMambaIPF、Mamba、IterativePFN、StraightPCF 和 Jittor。完整来源、引用和许可证边界见 `code/LICENSES.md` 与 `code/NOTICE`。
