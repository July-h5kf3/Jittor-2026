# NKAI PLR-003 Jittor 验证记录

## 1. CUDA 12.4 / RTX 4090 目标环境验收

目标主机：

```text
OS: Ubuntu 22.04 / Linux 5.15 / glibc 2.35
GPU: NVIDIA GeForce RTX 4090 24GB, compute capability 8.9
Driver: 580.105.08
CUDA Toolkit: 12.4.99
cuDNN: 8.9.7.29
Python: 3.10.14
Jittor: 1.3.11.0
NumPy: 1.24.4
```

Conda 环境、CUDA Toolkit、cuDNN、pip/conda cache、Jittor cache 和测试产物均位于 `/root/autodl-tmp`。Jittor 报告：

```text
has_cuda=true
nvcc_path=/root/autodl-tmp/envs/nkai310/bin/nvcc
cuda key=cu12.4.99_sm_89
Found cuda archs: [89]
```

Conda CUDA 把 `libcudart.so` 放在 `$CONDA_PREFIX/lib`，而 Jittor 1.3.11 默认检查 `lib64`。`main.py` 已加入安全兼容处理：检测环境 nvcc/cudart 后自动把环境 `bin` 加入 PATH，并在缺失时创建 `lib64 -> lib`。`environment.yaml` 同时固定 `cuda-toolkit=12.4.0`、`cuda-version=12.4` 和 `cudnn=8.9.7.29`。

### 1.1 Doctor 和 selective scan

```bash
python main.py doctor --deep
```

结果：

```text
python_3_10_or_newer=true
jittor_1_3_10_or_newer=true
cuda_available=true
JITTOR_SMOKE_OK 1280 16278412
```

该 smoke 在 CPU reference 与 sm_89 CUDA 上执行 selective scan forward/backward，并构造完整 1280-key `DenoiseNet`。

### 1.2 Python、unit 和 source policy

目标环境完成：

- `python -m compileall -q .`：通过；
- `python -m unittest tests.test_source_policy tests.test_routing`：5/5 通过；
- 删除 compile cache 后 `tools/audit_source_archive.py --root .`：通过；
- source tree 无 PyTorch/Triton/`mamba_ssm`/`causal_conv1d`/SciPy/trimesh/OmegaConf/tqdm runtime import。

### 1.3 从原始 mesh 的确定性数据准备

在目标 4090 环境对 tail key
`shapenet/02871439/104874322c6f7a75aba93753eed86c0a` 从原始 OBJ/cache 重新采样 50,000 点：

```text
surface_seed=1613388104
produced clean.npy SHA256=1004a9c724d68b1024ec06e94ce2f2f31547dc8c4d7a79dd4178a303eb04a0a9
historical clean.npy SHA256=1004a9c724d68b1024ec06e94ce2f2f31547dc8c4d7a79dd4178a303eb04a0a9
match=true
```

本地还对 MBI-009、DCD-001、table、airplane、sofa 各抽取一个历史 entry 检查，surface seed 和 `clean.npy` SHA256 均完全一致。

### 1.4 3DMambaIPF 训练、DCD 和 accumulation

目标环境通过以下真实 CUDA update：

1. 随机初始化，DCD weight 0，patch 128，1 micro-batch/1 optimizer step；
2. 从保存 checkpoint 重新加载，DCD weight 0.5，density-aware Chamfer CUDA/scatter 路径；
3. 2 micro-batches、`grad_accum_steps=2`、1 optimizer step。

代表结果：

```text
random-init loss=2.7756009102, checkpoint save success
DCD loss=0.9846314788, total loss=2.2175652981
accumulation: micro_batches=2, steps=1, mean_loss=2.3479722738
```

输出 checkpoint 能被 `infer.py` 严格重新加载。

### 1.5 StraightPCF VM → CVM → SPCF 训练

使用两个真实原始 ShapeNet mesh、每 stage 1 epoch/1 train step/1 validation step 运行：

```text
VM train/val:   2.5885539055 / 2.4320330620
CVM train/val: 18.3955898285 / 15.3954029083
SPCF train/val:143.1239776611 / 135.4595489502
spcf-final.pkl SHA256=5f5577d76ecd8ca1e73b1acff99713fdc08f19df150e8c9837ac8ae8cef4d232
```

同一命令再次以 `--resume` 执行时复用三个 completed stage，并生成相同 final SHA。

验收中发现并修复了 Jittor 1.3.11 checkpoint chaining 的一个隐蔽问题：直接把同一组 checkpoint Var 反复传给 `load_state_dict` 会使广播后的部分 CVM module 缺失梯度。当前实现先 materialize 独立 contiguous NumPy copy，再调用每个子模块的 `load_parameters`。修复后四个 CVM velocity module 的抽样梯度平方和均为非零：

```text
[0.0293912794, 0.0281293858, 0.0185012780, 0.0043981979]
```

### 1.6 真实 50K 3DMambaIPF inference

使用上述随机初始化 smoke checkpoint 对一个官方格式 50K noisy cloud 执行完整 `patch_size=2000` inference：

```text
shape=(50000,3)
dtype=float32
finite=true
stitching_fallback_count=0
model elapsed=304.9186 s（包含新算子编译）
process elapsed=312 s
output file SHA256=1999e6ca9b79e829d8fdfbb7831806ab7fa5638dc115b05268e4293416b2fbe9
```

该检查使用 smoke 权重，仅验证目标环境的完整 50K CUDA/runtime 路径，不代表 A 榜预测质量。

### 1.7 真实 50K ROT、assembly 与 packaging

使用从原始 mesh 完成 VM/CVM/SPCF smoke 后的 checkpoint，对同一个 50K cloud 执行 adaptive two-pass ROT：

```text
shape=(50000,3)
dtype=float32
finite=true
elapsed=68 s
output file SHA256=e1d145961aabd6a4f8cbd28a8bb0fe2c973a5b2bb56f66164c97740ba8e9a39d
```

随后依次通过：

- `assemble_incumbent.py`；
- `assemble_full.py`；
- `assemble.py`；
- `package_results.py`。

单 key canonical prediction ZIP：

```text
size=553560 bytes
SHA256=389bc55daf6f5c8266ed310d2f99b8c5a7c2a68a1452b0641648e377c98aa3c5
```

### 1.8 磁盘验收

完成环境与全部 smoke 后：

```text
/root/autodl-tmp: 100GB total, 14GB used, 87GB available
Conda env: 7.8GB
Conda package cache: 5.1GB
Jittor cache: 299MB
```

满足环境、cache 和测试产物放在 100GB 数据盘的要求。

## 2. 历史真实权重结构和数值 parity

历史 V100/CUDA 11.3 验证用于端到端迁移数值证据，不能替代上面的 4090 验收。

| 验证层级 | max absolute error | relative L2 / 说明 |
|---|---:|---:|
| selective scan random forward | `3.7252903e-09` | CUDA 与 Jittor reference |
| selective scan random gradients | `1.8626451e-09` | 八组输入梯度 |
| Mamba real-weight forward | `4.7683716e-07` | `1.1589873e-07` |
| FeatureExtraction real-weight | `2.6822090e-07` | `2.0371630e-06` |
| four-stage denoise real-weight | `1.8998980e-07` | `5.4200960e-07` |
| 256-point patch + stitching | `2.7567148e-07` | `4.7044546e-07` |
| training loss | `9.3132257e-10` | absolute difference |
| sampled parameter gradient | `4.6938658e-07` | max absolute difference |

Jittor state 为 1280 keys、16,278,412 values；历史 PyTorch state 多 48 个 BatchNorm `num_batches_tracked`，其余名称、shape 和数值 identity。

## 3. 历史 50K 跨框架检查

真实 Full-DCD 权重的一条 50K Jittor inference：

```text
shape=(50000,3), dtype=float32, finite=true, fallback=0
V100 elapsed=236.51 s
mean absolute coordinate difference vs historical PyTorch=8.111061e-05
relative L2=0.0010510734
```

ROT relative L2 为 `0.0006549766`。FPS seed 和 final best patch assignment 一致；少量 KNN near-tie 与 scatter/CUDA reduction 顺序造成跨框架浮点漂移。

## 4. 冻结 official200 route 验证

`assemble_full.py` 使用冻结 official200 输入：

- 200/200 输出与 A 榜 Full candidate 逐字节一致；
- route：`dcd=54`、`table=92`、`sofa=30`、`inherit=24`。

`assemble.py` 使用产生实际提交的冻结 official200 expert 输入：

- 200/200 输出与 PLR-003 A 榜提交逐字节一致；
- final route：`plr003_tail_standard=40`、`plr002_fallback=160`；
- changed vs Full：161。

因此 gate 阈值 4.0、0.25/0.75 blend、table 1.25× 和 sofa 1.25× 公式均被精确保留。

## 5. 完整重训边界

完整随机初始化训练与 803 个 50K inference 的计算量远大于代码检查 smoke；本次目标环境验收覆盖每种正式路径，但未在 4090 上等待完整 100-epoch base、全部 continuation 和 official200 全量重跑结束。动态 KNN、随机初始化和 CUDA reduction 也意味着完整重训不承诺 checkpoint/prediction byte-for-byte 相同。

提交材料严格保证：完整训练 DAG 可达、无外部 checkpoint/prediction 依赖、目标 CUDA 路径可运行、输出 canonical、冻结算法/结构/数据划分/超参数/route 不变。A 榜 79.23 是历史实际提交成绩；新一轮完整重训的官方指标只能由赛事评测系统确认。
