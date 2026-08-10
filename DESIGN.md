# PLR-003 设计说明

## 1. 目标与边界

PLR-003 面向 50K 点云降噪。比赛输出必须保持每个输入点的原索引与点数，不允许重采样后改变对应关系。因此系统只预测逐点位移，并将所有路由限定为 noisy-only displacement 统计：不读取 clean point cloud、mesh、答案或官方指标。

PLR-003 不是单一 checkpoint，而是：

1. 3DMambaIPF Full 与类别专家；
2. Jittor StraightPCF 的 ROT reference；
3. 三层冻结路由 PLR-001 → PLR-002 → PLR-003。

官方成绩为 79.23 / 67.42 / 91.04（Total / CD / P2S）。

## 2. 3DMambaIPF Jittor 网络

### 2.1 输入和输出

输入 patch 为 `P∈R^(B×N×3)`。网络预测逐点 displacement，并输出：

```text
P_denoised = P + displacement
```

模型包含四个串联 denoise modules。每个 module 重新在当前点坐标上提取特征并估计一个位移，因此能够迭代修正噪声。

### 2.2 FeatureExtraction

每个阶段使用动态图 EdgeConv：

1. 对 3D 坐标做 KNN；
2. 构造 edge feature `[x_i, x_j-x_i]`；
3. MLP + BatchNorm + LeakyReLU；
4. 邻域 max aggregation；
5. 对高维特征重新计算 pairwise squared distance 和 top-k；
6. 将多层 EdgeConv 特征连接为点特征。

坐标 KNN 使用 Jittor `jt.misc.knn`；高维 KNN 使用显式 pairwise distance + `jt.topk`。这一划分是保持历史模型图一致的必要条件。

### 2.3 Mamba block

每个点特征序列经过：

- input projection；
- depthwise causal Conv1d；
- SiLU；
- input-dependent `Δ, B, C` projection；
- selective state-space scan；
- gated output projection；
- residual connection和 RMSNorm。

冻结结构参数包括 `d_state=16`、`d_conv=4`、`expand=2`。代码保留历史 state-dict 名称与 shape，以便 identity checkpoint migration。

## 3. Jittor selective scan

### 3.1 前向方程

对 batch `b`、channel `d`、state `n`、time `t`：

```text
Δ_t = softplus(delta_t + delta_bias)
a_t = exp(Δ_t A)
b_t = Δ_t B_t u_t
x_t = a_t ⊙ x_(t-1) + b_t
y_t = Σ_n x_t[n] C_t[n] + D u_t
out_t = y_t · silu(z_t)
```

实现位于 `plr3d/ops/selective_scan.py`：

- CPU 路径：只使用 Jittor tensor op 的 reference recurrence；
- CUDA 路径：`jt.code` 自定义 forward 和 backward；
- 不依赖 Triton、`mamba_ssm` 或 `causal_conv1d`；
- 每 16 个 time step 保存一次 state checkpoint，backward 按区间重算中间状态，避免保存整个 `B×D×L×N` 状态张量；
- 当前限制 float32、`1<=N<=32`。

### 3.2 显式 backward

CUDA backward 反向遍历 recurrence。对

```text
x_t = a_t x_(t-1) + b_t
```

维护 `g_x_t`，并计算：

```text
g_a_t        += g_x_t x_(t-1)
g_b_t        += g_x_t
g_x_(t-1)    += g_x_t a_t
```

再通过 `a_t=exp(Δ_t A)`、`b_t=Δ_t B_t u_t`、softplus 和 SiLU 链式回传到 `u, delta, A, B, C, D, z, delta_bias`。随机输入下 forward 与所有输入梯度均已和 Jittor reference 对齐。

## 4. 训练设计

### 4.1 pure-Laplace continuation

训练数据只从 clean training cloud 合成：

1. 使用确定性 seed 从 clean cloud 选择 patch；
2. 对每个坐标加入零均值 Laplace 噪声；
3. 每个 patch 的 sigma 从冻结区间采样；
4. 每个 epoch/worker/cloud/patch 的随机性由显式 seed 派生。

这保证训练不访问测试 clean/mesh，同时能精确复现专家类别与噪声范围。

### 4.2 NN stages loss

四阶段预测均参与 nearest-neighbor reconstruction loss，越后阶段权重越高。每个预测点寻找 target 最近点，计算逐阶段 L2 距离，再按冻结 stage weight 求和。

### 4.3 Density-aware Chamfer (DCD)

最后阶段加入 density-aware Chamfer。对双向最近邻距离先做：

```text
score = 1 - exp(-alpha · distance)
```

然后按目标点被匹配的 multiplicity 做 density reweight，以减少点聚集和局部覆盖塌缩。训练总 loss 为：

```text
L = L_nn_stages + λ_dcd L_dcd
```

`λ_dcd` 由 stage 配置：base/MBI-009/MBI-011 为 0，DCD-001/TSD-003/airplane/tail 为 0.5，TSD-004/SSD-009/table/sofa strong 为 1.0。

### 4.4 Jittor 优化和 checkpoint

- Adam；
- gradient clipping；
- 可选 cosine decay；
- Jittor MPI 多进程训练；
- 每个 epoch 保存 Jittor checkpoint 与 JSON summary；
- BatchNorm state 不包含 PyTorch 的 `num_batches_tracked`。

## 5. 50K patch inference

直接对 50K 点运行高维动态图会超过显存，因此采用和历史方案相同的 patch/stitching：

1. 将全云归一化到 unit sphere；
2. `num_patches = seed_k × seed_k_alpha = 6×20 = 120`，并使用最小覆盖约束得到冻结数量；官方 50K 配置实际取 150 个 FPS seed；
3. 对每个 seed 取 2000 个最近点；
4. 每个 patch 独立归一化并执行四阶段推理；
5. 对原云中的每个点，在覆盖它的 patches 中选择归一化 seed distance 最小的 prediction；
6. 极少数未覆盖点回退原输入；
7. 反归一化，并按原 index 写回 `float32 (N,3)`。

`infer.py` 会检查 shape、finite、count，并在 manifest 记录 fallback 数。GPU 上仍分 patch 执行，避免同时缓存全部 FeatureExtraction 图。

## 6. ROT reference

PLR gate 需要一个与 3DMambaIPF 不同的 noisy-only reference。`generate_rot.py` 调用包内 Jittor StraightPCF：

1. pass-1，`predict_alpha=1.05`；
2. 根据 cross-view consistency 做 adaptive calibration；
3. 对 pass-1 输出执行 pass-2；
4. 用 `gamma=0.5` 融合 pass-1 和 pass-2；
5. 重新 canonicalize 到原 key 和 index。

ROT 仅作为 displacement reference，不参与 clean-based selection。

## 7. 冻结路由

定义逐点 displacement-ratio；分母对每个点用 float64 `tiny` 防止除零：

```text
r_i = ||expert_i - noisy_i||_2 / max(||rot_i - noisy_i||_2, tiny)
ratio = quantile_0.95({r_i})
```

### 7.1 标准门控

标准门控用于 airplane、table、sofa、Full 专家和 tail：

```text
if ratio <= 4.0:
    output = float32(0.25 × rot + 0.75 × expert)
else:
    output = rot
```

阈值固定为 4.0。门控只读取 noisy、ROT 和 prediction；没有 clean/mesh/metric 输入。

### 7.2 PLR-001

- airplane：airplane expert 经标准门控；
- table：table expert 经标准门控；
- sofa：分别得到 `source=standard(DCD)` 与 `strong=standard(sofa)`，再计算 `source + 1.25 × (strong-source)`；
- 其他类别：Full fallback。

Official200 route 数为 airplane 35、table 92、sofa 30、Full fallback 43。

### 7.3 PLR-002

table 执行冻结外推：

```text
PLR002_table = Full + 1.25 × (PLR001 - Full)
```

非 table 回退 PLR-001。Official200 为 table 92、fallback 108。

### 7.4 PLR-003

七个 tail 类别使用 tail expert 与标准门控；其他类别回退 PLR-002。Official200 最终为 tail 40、fallback 160；相对 Full 改变 161 个云。

## 8. Full 重建

`assemble_full.py` 实现冻结 Full noisy-only route：

- `02691156/03046257/03642806/04330267/04468005`：DCD expert 标准门控；
- table `04379243`：table expert 标准门控；
- sofa `04256520`：分别门控 DCD/sofa expert，再做 `source + 1.25 × (strong-source)`；
- 其他类别：Full incumbent fallback。

这一步使用 `assemble_incumbent.py` 从本次随机初始化训练得到的 MBI-009 prediction 与 ROT 重建 incumbent，再使用同一流水线训练得到的 DCD/TSD/SSD expert 重建 Full；不依赖包外 prediction 或 checkpoint。历史 MBI-020 的最终不可达分支按 dead-branch pruning 省略。

## 9. 从随机初始化生成权重

`configs/reproduce_full.json` 固定完整 checkpoint DAG：

```text
base-random
└── MBI-009
    ├── DCD-001 ─┬─ SSD-009 ── PLR sofa
    │             ├─ PLR airplane
    │             └─ PLR tail
    └── MBI-011 ── TSD-003 ── TSD-004 ── PLR table
```

StraightPCF 独立从随机初始化训练 VM → CVM → SPCF。所有 checkpoint 都生成在 `NKAI_WORK_ROOT`，提交包不提供历史 checkpoint migration 工具或外部权重。历史参数 key/value 对齐仅作为 Jittor port 的验证证据，不属于正式运行输入。

## 10. 可复现性与数值边界

冻结 seed、key 排序、FPS 首点、patch 数、stage 数、route 阈值/系数与训练 sigma 均写入代码或配置。assembly 对历史 official200 专家输出已达到 200/200 文件逐字节一致。

Jittor 与 PyTorch 的 CUDA reduction、动态图 KNN near-tie 和 scatter atomic 顺序可能不同，因此模型 inference 不承诺跨框架逐位一致。验证重点为：

- 参数/结构 identity；
- selective scan forward/backward；
- block、feature、四阶段、loss 与梯度数值对齐；
- canonical shape/dtype/finite/index；
- 冻结 route 与 assembly 完全一致。

详见 `VALIDATION.md`。
