# NKAI 赛题二 PLR-003：Jittor 原生完整复现代码

本目录为队伍 **NKAI** 的赛题二 A 榜代码检查材料。A 榜排名 **031**，官方成绩为 **79.23 / 67.42 / 91.04（Total / CD / P2S）**。

代码包只包含源码、配置、公开训练样本标识列表和文档；**不包含模型权重、数据集、预测结果、缓存或日志**。完整流水线从赛事原始 ShapeNet 训练 mesh 和随机初始化开始，依次完成训练数据构建、3DMambaIPF 系列训练、StraightPCF ROT 训练、200 个测试点云推理、Full/PLR-003 assembly 以及预测 ZIP 生成。

## 1. 合规范围

正式路径只使用 **Jittor、NumPy 和 Python 标准库**：

- 不导入或调用 PyTorch、Triton、`mamba_ssm`、`causal_conv1d`；
- 不依赖 SciPy、trimesh、OmegaConf、tqdm 或预编译第三方点云算子；
- `plr3d/ops/selective_scan.py` 使用 Jittor `jt.code` 实现 CUDA forward/backward；
- `train_rot.py` 直接实现 OBJ/cache 读取、mesh surface sampling、噪声、旋转、patch 构建和 VM → CVM → SPCF 三阶段训练；
- 包内没有历史 PyTorch checkpoint reader，也没有任何祖先 checkpoint。

随包提供的 `configs/lists/train_*.txt` 仅为公开训练集中的 `shapenet/<synset>/<model>` 标识，用于冻结 A 榜训练划分；它们不是点云数据、标签、预测或权重。`train_tail_surface_seeds.json` 仅记录确定性 surface sampling seed。

## 2. 目标环境

最终检查目标：

- Ubuntu 22.04；
- NVIDIA RTX 4090 24GB；
- CUDA Toolkit 12.4；
- cuDNN 8.9.7；
- Python 3.10；
- Jittor 1.3.11.0（满足 `>=1.3.10`）；
- NumPy 1.24.4。

安装方式二选一：

```bash
# 最小正式 Python runtime（要求系统已提供 CUDA 12.4 与 cuDNN 8.9）
pip install -r requirements.txt

# 或按赛事检查环境建立 Conda 环境（推荐；文件位于 ZIP 根目录）
conda env create -f ../environment.yaml
conda activate nkai-track2-plr003
```

建议把环境和缓存放在大容量数据盘：

```bash
export CONDA_PKGS_DIRS=/root/autodl-tmp/conda-pkgs
export PIP_CACHE_DIR=/root/autodl-tmp/pip-cache
export JITTOR_HOME=/root/autodl-tmp/jittor-cache
export cache_path=/root/autodl-tmp/jittor-cache
```

检查环境和 CUDA 自定义算子：

```bash
python main.py doctor
python main.py doctor --deep
```

`main.py` 会在检测到 Conda CUDA Toolkit 时把环境 `bin/` 加入 `PATH`，并在需要时创建 `lib64 -> lib` 兼容链接；这是 Jittor 1.3.11 在 Conda CUDA 12.4 布局下查找 `libcudart.so` 所需的兼容处理。首次运行会编译 CUDA 算子；selective scan 支持 float32、SSM state 数 1～32，冻结网络使用 `d_state=16`。

## 3. 原始数据布局

训练根目录必须能按 key 找到原始 OBJ：

```text
<TRAIN_ROOT>/shapenet/<synset>/<model>/models/model_normalized.obj
```

若原始数据同时提供 `model_normalized.obj.cache.npz`，代码优先读取其中的 `vertices`/`faces` 以精确保持历史 mesh 解析顺序；没有 cache 时使用包内标准库 OBJ parser。

测试根目录：

```text
<TEST_ROOT>/shapenet/<synset>/<model>/noisy.npy
```

正式测试集应有 200 个 key，每个 `noisy.npy` 为 finite `float32 (50000,3)`。程序不读取测试集 clean、mesh、ground truth 或官方指标。

## 4. 一条命令完成从原始数据复现

```bash
cd code
export NKAI_TRAIN_ROOT=/data/track2/dataset_train
export NKAI_TEST_ROOT=/data/track2/dataset_test
export NKAI_WORK_ROOT=/root/autodl-tmp/nkai-track2-reproduction

python main.py doctor --deep
python main.py pipeline \
  --config configs/reproduce_full.json \
  --log-dir "$NKAI_WORK_ROOT/logs"
```

先检查全部 39 个 stage 的展开命令而不执行：

```bash
python main.py pipeline \
  --config configs/reproduce_full.json \
  --log-dir "$NKAI_WORK_ROOT/logs" \
  --dry-run
```

流水线的 `skip_if_exists` 指向每个已完成的 manifest 或 checkpoint；中断后重复同一命令会跳过完成阶段。`train_rot.py` 始终以 `--resume` 进入，能继续未完成的 VM/CVM/SPCF 阶段。若需要故意重跑某阶段，应先删除该阶段的完成产物及其输出目录。

最终生成：

```text
$NKAI_WORK_ROOT/result_plr003.zip
$NKAI_WORK_ROOT/manifests/result_zip_audit.json
$NKAI_WORK_ROOT/manifests/source_policy_audit.json
```

预测 ZIP 含 200 个排序且唯一的 `shapenet/.../denoised.npy`，每个输出均为 finite canonical `float32 (50000,3)`。

## 5. 完整流水线内容

`configs/reproduce_full.json` 固定以下链：

1. 从原始 OBJ 确定性采样 clean 50K 数据：base、MBI-009 650、DCD-001 3000、table 1200、airplane 1000、sofa 1000、tail 1885；
2. `base-random`：随机初始化 3DMambaIPF base training；
3. MBI-009 mixed-domain continuation；
4. DCD-001、MBI-011、TSD-003、TSD-004、SSD-009 训练链；
5. PLR-001 airplane/table/sofa pure-Laplace 专家和 PLR-003 tail 专家；
6. StraightPCF VM → CVM → SPCF 随机初始化训练；
7. 自动发现 200 个测试 key，并按冻结 synset 生成 route 子列表；
8. ROT、MBI-009、Full 三组专家和 PLR 四组专家推理；
9. incumbent、Full、PLR-003 assembly；
10. 分别执行 canonical 结果 ZIP 审计与源码 policy 审计。

单卡 4090 的 continuation 使用 `train.py --grad-accum-steps 4` 模拟历史四卡全局 batch 16；base 使用 accumulation 8 对齐原始八卡全局 batch 32。训练 surface 数据约 15GB；还应为 Conda/CUDA/Jittor 编译缓存、checkpoint、预测和日志预留空间，建议 `NKAI_WORK_ROOT` 所在分区至少有 60GB 可用空间。

## 6. 统一入口

```bash
python main.py --help
```

主要子命令：

```text
prepare-keys          从外部数据目录发现 key
filter-keys           按 synset 生成冻结 route key list
prepare-data          从原始 OBJ 构建训练 clean cloud
train                 3DMambaIPF 随机初始化或 checkpoint continuation
train-rot             StraightPCF VM/CVM/SPCF 从原始 mesh 训练
infer                 单个 3DMambaIPF checkpoint 推理
generate-rot          StraightPCF adaptive two-pass ROT 推理
assemble-incumbent    MBI-009 等价 noisy-only incumbent
assemble-full         冻结 Full route
assemble-plr003       冻结 PLR-001 → PLR-002 → PLR-003 route
package-results       生成 canonical prediction ZIP
audit-source          检查源码及结果 ZIP
pipeline              执行可恢复 JSON 流水线
```

每个 delegate 入口可用 `python main.py <name> -- --help` 查看底层脚本参数。例如仅运行 tail continuation：

```bash
python main.py train -- \
  --checkpoint "$NKAI_WORK_ROOT/training/dcd001/dcd001-freq-dcd-epoch02.pkl" \
  --data-root "$NKAI_WORK_ROOT/data/tail_1885" \
  --train-list "$NKAI_WORK_ROOT/lists/tail_1885.txt" \
  --sigma-map-json configs/plr003_tail_sigma.json \
  --output-root "$NKAI_WORK_ROOT/training/plr_tail_manual" \
  --checkpoint-prefix plr003-tail \
  --epochs 2 --samples-per-shape 4 --batch-size 4 --grad-accum-steps 4 \
  --gaussian-probability 0 --outlier-fraction 0 \
  --lr 1e-6 --dcd-weight 0.5 --device cuda
```

`main.py` 的 delegate 子命令必须用独立的 `--` 将底层脚本参数与统一入口参数分隔；JSON pipeline 会直接调用底层脚本，因此配置中的 `args` 不需要这个分隔符。

## 7. 冻结路由公式

所有 gate 仅使用 noisy、ROT 和 member prediction：

```text
ratio = q95(||member-noisy|| / max(||ROT-noisy||, float64_tiny))
ratio <= 4.0 时：output = float32(0.25*ROT + 0.75*member)
否则：output = ROT
```

- PLR-001：airplane/table/sofa 专家；其余回退 Full；
- PLR-002：table 使用 `Full + 1.25 * (PLR001 - Full)`；
- PLR-003：七个 tail 类别使用标准门控；其余回退 PLR-002；
- Full sofa：先门控 DCD source 和 SSD strong，再用 `source + 1.25*(strong-source)`。

`assemble_incumbent.py` 保留最终可达的 MBI-009 noisy-only incumbent。历史 MBI-020 的后续分支在最终 Full/PLR 路由中全部被覆盖，因此按 dead-branch pruning 不再训练或生成；最终可达输出不变。

## 8. 测试与源码审计

```bash
python -m compileall -q .
python -m unittest tests.test_source_policy tests.test_routing
python tests/smoke_jittor.py
python tools/audit_source_archive.py --root .
```

打包前必须删除 `__pycache__` 和 `.pyc`。审计会拒绝权重/数据/预测后缀、日志、缓存、私有绝对路径、密钥以及禁止的 runtime import。

## 9. 已验证边界

- `assemble_full.py` 在冻结 official200 输入上与 A 榜 Full candidate **200/200 文件逐字节一致**；
- `assemble.py` 在冻结 official200 输入上与实际 PLR-003 提交 **200/200 文件逐字节一致**；
- selective scan forward/backward、Mamba、FeatureExtraction、四阶段网络、patch/stitch、训练 loss 和 sampled gradient 已与历史实现做真实权重数值对齐；
- 动态 KNN near-tie、scatter/CUDA reduction 和不同 GPU 架构可能造成非逐位浮点漂移，因此从随机初始化完整重训不承诺 checkpoint 或 prediction byte-for-byte 相同；算法、模型结构、超参数、数据划分和冻结路由保持一致；
- 所有正式输出仍强制满足 key、点数、原索引、dtype、shape 和 finite 约束。

更详细的模型与路由见 `DESIGN.md`，训练参数见 `WEIGHTS.md`，验证记录见 `VALIDATION.md`，许可证与来源见 `LICENSES.md` 和 `NOTICE`。
