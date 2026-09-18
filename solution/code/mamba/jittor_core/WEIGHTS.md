> 历史复审文档，保留用于追溯。当前最终方案、权重及运行命令以[仓库首页](../../../../README.md)和[复现指南](../../../../docs/REPRODUCE.md)为准；此文不代表最终训练链已完整重现。

# 权重与训练链说明

## 1. 权重策略

提交包不含任何 `.pth/.pt/.ckpt/.pkl/.npz` 模型资产。正式复现不需要外部 checkpoint，也不提供历史 checkpoint migration 路径；所有权重都由 `configs/reproduce_full.json` 在 `NKAI_WORK_ROOT` 中从随机初始化生成。

完整命令：

```bash
export NKAI_TRAIN_ROOT=/data/track2/dataset_train
export NKAI_TEST_ROOT=/data/track2/dataset_test
export NKAI_WORK_ROOT=/root/autodl-tmp/nkai-track2-reproduction
python main.py pipeline --config configs/reproduce_full.json \
  --log-dir "$NKAI_WORK_ROOT/logs"
```

## 2. 3DMambaIPF checkpoint DAG

```text
base-random-epoch100
└── mbi009-ft-mix-epoch03
    ├── dcd001-freq-dcd-epoch02
    │   ├── ssd009-strong-epoch02
    │   │   └── plr001-sofa-epoch02
    │   ├── plr001-airplane-epoch02
    │   └── plr003-tail-epoch02
    └── mbi011-cat04379243-heavy-epoch02
        └── tsd003-dcd-epoch02
            └── tsd004-strong-epoch02
                └── plr001-table-epoch02
```

输出位于：

```text
$NKAI_WORK_ROOT/training/base/
$NKAI_WORK_ROOT/training/mbi009/
$NKAI_WORK_ROOT/training/dcd001/
$NKAI_WORK_ROOT/training/mbi011_table/
$NKAI_WORK_ROOT/training/tsd003/
$NKAI_WORK_ROOT/training/tsd004/
$NKAI_WORK_ROOT/training/ssd009/
$NKAI_WORK_ROOT/training/plr_airplane/
$NKAI_WORK_ROOT/training/plr_table/
$NKAI_WORK_ROOT/training/plr_sofa/
$NKAI_WORK_ROOT/training/plr_tail/
```

每个 Jittor checkpoint 包含严格的 `DenoiseNet.state_dict()` 和 metadata；metadata 记录初始化模式、base checkpoint SHA256、训练 list SHA256、sigma map SHA256、结构、超参数、epoch 与 loss。

## 3. 单卡全局 batch 复现

历史 continuation 使用四个 rank、每 rank batch 4；目标 4090 单卡 continuation 使用：

```text
batch_size=4
world_size=1
grad_accum_steps=4
effective_batch_size=16
```

`train.py` 每四个 micro-batch 执行一次 continuation optimizer update；最后不足四个 micro-batch 的 group 使用实际 group size 归一化。Base 对齐原始八卡预训练的 global batch 32，使用 `samples_per_shape=4` 和 `grad_accum_steps=8`。`--expected-steps-per-epoch` 检查 optimizer update 数，而不是 micro-batch 数。

## 4. 数据与共同参数

训练 clean cloud 由 `tools/prepare_reproduction_data.py` 从原始 OBJ 确定性采样 50,000 点。共同参数：

```text
patch_size=1000
patch_ratio=1.2
frame_knn=32
num_modules=4
noise_decay=4.0
weight_decay=0
grad_clip=1.0
dcd_alpha=100
dcd_n_lambda=0.5
```

### Base random initialization

```text
train_count=15733
samples_per_shape=4
epochs=100
sigma=[0.005,0.020]
gaussian_probability=0.5
outlier_fraction=0.02
outlier_scale=3
scale=[0.8,1.2]
lr=1e-4
dcd_weight=0
seed=2020
grad_accum_steps=8
effective_batch_size=32
```

### MBI-009

```text
train_count=650 (13 categories × 50)
samples_per_shape=4
epochs=3
optimizer_steps_per_epoch=163
sigma=[0.005,0.020]
gaussian_probability=0.5
outlier_fraction=0.02
scale=[0.8,1.2]
lr=1e-5
dcd_weight=0
seed=20260814
```

### DCD-001

```text
train_count=3000
samples_per_shape=1
epochs=2
optimizer_steps_per_epoch=188
lr=5e-6
dcd_weight=0.5
seed=20261303
```

### Table chain

```text
MBI-011: 1200 shapes × 2 samples, 2 epochs, 150 steps, lr=5e-6, dcd=0
TSD-003: 1200 × 1, 2 epochs, 75 steps, lr=2e-6, dcd=0.5
TSD-004: 1200 × 1, 2 epochs, 75 steps, lr=1e-6, dcd=1.0
```

这些 stage 使用 sigma `[0.005,0.020]`、Gaussian probability 0.4、outlier fraction 0.02、scale `[0.8,1.2]`。

### Sofa strong base

```text
train_count=1000
samples_per_shape=1
epochs=2
optimizer_steps_per_epoch=63
sigma=[0.005,0.020]
gaussian_probability=0.4
outlier_fraction=0.02
scale=[0.8,1.2]
lr=2e-6
dcd_weight=1.0
seed=20262011
```

## 5. PLR pure-Laplace experts

共同设置：Gaussian probability 0、outlier fraction 0、scale 1、2 epochs、每 shape 4 samples、`lr=1e-6`。

| Expert | Base | Count | Steps/epoch | Sigma | DCD weight | Seed |
|---|---|---:|---:|---|---:|---:|
| airplane | DCD-001 | 1000 | 250 | `[0.012489692407870212,0.014616960844581714]` | 0.5 | 20262701 |
| table | TSD-004 | 1200 | 300 | `[0.009048618504872253,0.012903604290095811]` | 1.0 | 20262702 |
| sofa | SSD-009 | 1000 | 250 | `[0.008374454205880252,0.011346570064822026]` | 1.0 | 20262703 |
| tail | DCD-001 | 1885 | 472 | `configs/plr003_tail_sigma.json` | 0.5 | 20262731 |

## 6. StraightPCF ROT 权重

`train_rot.py` 只依赖 Jittor、NumPy 和标准库，从原始 mesh 训练：

```text
VM (default 3 epochs)
  -> copy one velocity net into four CVM modules
CVM (default 30 epochs)
  -> copy four velocity nets into SPCF
SPCF (default 40 epochs)
  -> output spcf-final.pkl
```

共同参数：

```text
samples_per_epoch=10000
batch_size=32
surface_points=32768
vertex_samples=1024
patch_size=1000
train Laplace sigma=[0.008,0.014]
outlier_fraction=0.02
outlier_scale=3
validation sigma=0.011
Adam lr=1e-4
```

每个 stage 保存 `latest`、`best` 和 `stage_state.json`；`--resume` 可从 latest 权重继续。最终 raw Jittor state 位于：

```text
$NKAI_WORK_ROOT/training/rot/spcf-final.pkl
```

## 7. 历史 source lock（仅审计参考）

以下 SHA 仅用于说明 A 榜研究资产谱系，不是运行输入：

- Full DCD：`406a7fa3b2781d84dc4c676fcb808faef40cb5ec24a8fe1d42a6349a5905b6bc`
- airplane：`198f57c0ecd543f1b04b1fab08b530203e866c10e88d5a20841d2ca404b4afa7`
- table：`6c1a19653adc420dd8d4ec3ebdea2e38da9974dcc847d61a1ca5307fb9466e89`
- sofa：`53b32b42de5dd0b011a169d9a3cc885c72ba086cbe57da51b4ec40bf846eff37`
- tail：`38bea336f017439d71cab77e8fa05bc42c06168802eaa25f20eab0e41a4c34cb`

Jittor 重训 checkpoint 的 SHA 会因初始化、GPU 和 reduction 顺序而不同，必须以本次训练 manifest 中实际 SHA 为准。
