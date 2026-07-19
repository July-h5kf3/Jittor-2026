# 关键实验记录

本文只保留影响最终决策的结果。逐小时日志、失败 checkpoint、中间预测和编译 cache 均不进入仓库，并在实验定案后从服务器删除。

## 评测口径

- 官方分：`0.5 × CD_score + 0.5 × P2S_score`。
- `local2`：62 个内部样本，用于严格配对淘汰；近期实验统一与 V4CTRL 比较。
- 新方法只有在 adaptive 总分相对 control 至少提高 `+0.05` 时才保留。
- 已观察到本地与线上幅度不一致；local2 小差距只能用于排除，最终仍以线上反馈为准。
- patch 推理的随机状态会随样本顺序推进；严格对照必须使用同一 canonical 列表顺序，不能拼接倒序或分片推理结果。

## 2026-07-18～19：ROT-001（完成，线上 77.87，当前最佳）

- 假设：`AugmentLinear` 只更新 mesh 顶点，未更新已经采样出的 clean/noisy 点，因此历史配置中的旋转与缩放实际为空操作。EdgeConv 不是旋转等变网络；让 paired clean/noisy 点真正接受随机旋转，可能改善跨方向的表面覆盖和 CD，同时保持 P2S。
- 唯一变量：修复 `Asset.transform()` 的采样点语义，并在独立配置中只启用 `rotate_p=0.5`；`scale_p=0.0`，噪声、模型、优化器、四卡 batch 与 seed 均保持 CVM-002 配方。
- 训练链：`train_spcfgfnrot001_cvm.yaml` -> `train_spcfgfnrot001.yaml`，正式训练固定 `GPU_LIST=0,1,2,3 NP=4 --seed 123`。
- 训练结果：CVM 最优 epoch 22，`val/loss_sum=2.133761`；SPCF 最优 epoch 83，`val/loss_sum=4.084705`，训练在 epoch 113 按 30 epoch 无提升停止。SPCF checkpoint SHA256 为 `60bc3e8a73c866c06950461db43dbae64ab54887ed67796b190f6ff23a64b82a`；峰值显存 GPU0/1/2/3 为约 `7605/6453/6337/6337 MiB`。
- 推理验证：固定 seed 123、`localtest2/local2.txt` canonical 顺序；raw 与 pass2 均生成 62/62 个 `(50000, 3)`、`float32`、有限数组，点数与 noisy 输入逐云一致。raw 推理约 28 分 53 秒，pass2 约 27 分 41 秒。
- local2 结果（同一 `baseline_adaptive` reference）：

  | 输出 | CD | P2S | 总分 | 相对 reference |
  |---|---:|---:|---:|---:|
  | ROT raw | 56.86644119 | 88.59776568 | 72.73210344 | -0.26380475 |
  | ROT mean/var | 57.11178932 | 88.70349807 | 72.90764369 | -0.08826450 |
  | ROT CV two-pass | **57.30330576** | **90.20316221** | **73.75323399** | **+0.75732580** |

- 两遍融合相对 reference 的 paired bootstrap 95% CI 为 `[+0.42636114,+1.08752123]`；相对当前 seed456 `55/10/35` CV two-pass（73.71934868）高 `+0.03388531`。提高云数 `49/62`，12 个类别 leave-one-category-out 总分范围 `[+0.39418172,+0.85561842]`；最大类别 `04379243` 留出后仍为 `+0.39418172`。CD 单独增益 CI 跨零（`[-0.20079385,+0.45553536]`），总分提升主要由 P2S `+1.38018658` 驱动，因此不宣称已解决 CD 瓶颈。
- A 榜提交：固定官方 200 云 canonical 顺序和 seed 123，raw、mean/var、pass2 与最终融合均生成 200/200 个 `float32 (50000,3)` 有限数组。提交 ZIP 大小 `110775768` 字节，SHA256 `1cdf56febb033f1a24d1bc6456933cb4584c7532c3126a7eaeb54d3a597369da`；远端与本地分别完成严格归档复验。
- 官方结果：request_id `2026071921373714762837`，总分 **77.87**，CD/P2S = **65.91 / 89.83**；相对上一最佳 76.04 提升 **+1.83**，CD/P2S 分别提升 `+1.31 / +2.36`。mean CD/P2S pred 为 `0.000084 / 0.000054`，对应 noisy 为 `0.000246 / 0.000196`。
- 决定：ROT-001 升级为当前 A 榜主方案，并保留单 checkpoint、canonical 两遍推理和后处理链作为可复现基线。local2 相对 seed456 候选仅 `+0.03389`，线上却提升 `+1.83`，再次说明 local2 小差距不能直接线性映射到线上幅度；后续仍优先改善 CD，同时保护本次 P2S 增益。

## 2026-07-18：seed789 第四轨迹（否定）

- checkpoint：`experiments/spcfgfncvm002seed789_spcf/checkpoint_best.pkl`，SHA256 `234f61d80bd14365178d5dbc3ac5e29cfa87946e5e897562440ef8f66a5ef408`。
- canonical local2 推理：单张 V100，62 云，28 分 37 秒，推理显存约 10.0 GiB。
- 单模型 raw CD/P2S/总分：`56.96503 / 88.69052 / 72.82778`。
- 单模型 mean/var：`57.19747 / 88.79085 / 72.99416`；相对同次重建 baseline 为 CD `+0.02863`、P2S `-0.03213`、总分 `-0.00175`，总分 95% CI `[-0.02336,+0.02182]`。这是近中性的 CD/P2S 交换，不是单模型提升。
- 四轨迹扫描固定原 `55/10/35` 比例，只把 seed789 权重从 5% 扫到 40%，raw-alpha gate、CV gamma 和 two-pass residual 全部不变。10% 时最高：CD/P2S/总分 `57.26989 / 90.17347 / 73.72168`，相对 73.71935 为 `+0.00233`，95% CI `[+0.00104,+0.00366]`，LOCO 范围 `[+0.00132,+0.00262]`；15% 后 CD 开始回撤，35% 以上总分显著下降。
- 决定：增益比 `+0.05` 保留门槛低一个数量级，不生成提交包、不把第四模型加入 A 榜候选；checkpoint 和 canonical raw 预测保留作误差研究证据。

## PNX-001：PointNeXt-lite 层次化残差编码器（CUDA smoke 通过）

- 假设：普通全局注意力、Point Transformer 和 RoPE 已被否定，但当前编码器始终停留在 1000 点单尺度图；显式 coarse context 可能减少局部 patch 的覆盖收缩，从而优先改善 CD。
- 唯一结构变量：在已验证的 EdgeConv 特征后加入窄通道层次残差分支。按距离排序的 patch 每 4 点确定一个 deterministic coarse seed，64 维、k=16 的位置 kNN 聚合 fine 特征，在 coarse 图做一次残差更新，再用 inverse-distance 3-NN 回插；不引入全局 attention。单卡 batch=32 smoke 曾使 PNX 与原 CVM-002 control 同样 OOM；检查 Jittor `Dataset` 源码确认正式全局 batch=32 会被四 rank 切为每 GPU 8，因此该 OOM 不构成结构否定，显存判定必须使用 per-rank batch=8。
- 风险控制：最终融合投影零初始化，因此现有 checkpoint 加载后的初始输出与基线 bitwise 相同。CPU 测试覆盖非等长 query/source gather、17 点非整除 stride、分支恒等性和基线 `state_dict` 迁移，3 项均通过。V100 CUDA 的 per-rank `batch=8, N=1000` forward/backward 也通过：allocator 为 `5.98 GiB`，control 为 `5.61 GiB`；单步约 `0.85s` 对 `0.63s`，增加约 0.37 GiB、34% 时间，满足 16GB 与 B 榜固定 patch 约束。
- 对照：`train_spcfgfnpnx001_cvm.yaml` 与 ROT-001 使用同一初始化、真实旋转 transform、四卡 batch、seed 和优化器；唯一变量是 `encoder_type=pointnext_lite`。下一门槛是正式四卡 CVM/SPCF 与 canonical local2 指标。
- B 榜属性：50,000 点整云仍走固定 1000 点 patch；新增计算只在 250 点 coarse set 上进行，复杂度与整云点数近似线性扩展。

## 2026-07-19：COND-001 显式 remaining-time/stage 条件（否定）

- 假设与唯一变量：在每个 velocity encoder 的 FiLM 中加入 `[remaining_time, normalized_stage_index]`，让不同 stage 显式感知残余时间与迭代位置；其余初始化、真实旋转、噪声、模型、优化器、四卡 batch 与 seed 对齐 ROT-001。remaining time 只来自 distance head，stage index 是公开模块序号，不读取 clean、mesh 或隐藏信息。
- 风险控制：condition 的最终 FiLM 层零初始化；旧 checkpoint 注入后，任意非零 condition 的初始预测与 control bitwise 相同。`tests.test_velocity_condition`、zero-init checkpoint equivalence，以及 V100 `B=8, N=1000` forward/backward smoke 均通过。
- 训练结果：CVM 最优 epoch 27，`val/loss_sum=2.094119`，checkpoint SHA256 `759f1accb33fc4c2c61197f7fb1dd6f5d497cce28263adfc6f1644954327d049`；SPCF 最优 epoch 66，`val/loss_sum=3.629969`，epoch 96 early stop，checkpoint SHA256 `a4a1e9719814208fab26329357619fadbe312fc1ce9646ddf864ff394ed5cf4b`。SPCF 采样峰值显存约 `6.1/5.1/5.1/5.1 GiB`。
- 推理验证：固定 seed 123 与 canonical 文件顺序；raw、mean/var 与 CV two-pass 都生成 62/62 个 `float32`、有限数组，shape 与 noisy 逐云一致。pass2 用时 27 分 52 秒；writer 会保留一层源目录前缀，因此融合时以 `results_local2_spcfgfncond001a105_pass2/results_local2_spcfgfncond001a105` 作为 canonical 第二遍根目录，没有重跑或重排样本。
- local2 结果（同一 `baseline_adaptive` reference）：

  | 输出 | CD | P2S | 总分 | 相对 reference | 总分 95% CI |
  |---|---:|---:|---:|---:|---:|
  | COND raw | 54.79508634 | 86.77322508 | 70.78415571 | -2.21175248 | `[-2.89437294,-1.56916672]` |
  | COND mean/var | **55.16793266** | **86.99171222** | **71.07982244** | **-1.91608575** | `[-2.49844738,-1.36174951]` |
  | COND CV two-pass | 53.88796292 | 87.52303132 | 70.70549712 | -2.29041107 | `[-3.14392905,-1.48020129]` |

- 相对 ROT CV two-pass，raw/mean-var/two-pass 的配对总分差分别为 `-2.96907828`、`-2.67341155`、`-3.04773687`，95% CI 分别为 `[-3.59014283,-2.38614507]`、`[-3.20418414,-2.18173080]`、`[-3.71979772,-2.42293750]`。相对 seed456 best 则为 `-2.93519297`、`-2.63952624`、`-3.01385156`，95% CI 分别为 `[-3.62730987,-2.28417993]`、`[-3.24840554,-2.06784361]`、`[-3.76650957,-2.30091432]`。
- 最好的 mean/var 轨迹相对 reference 的 CD/P2S 分别下降 `-2.00090809`、`-1.83126341`，两项 CI 都全负；仅 `13/62` 云提高。12 个类别中只有单样本类别 `02876657` 为正，最大类别 `04379243` 仍下降 `-2.25229473`；leave-one-category-out 总分差始终为负，范围 `[-2.07246425,-1.63920777]`。因此失败是全局泛化回撤，不是少数类别或极端样本驱动。
- 决定：明确否定 COND-001，不制作线上包、不把 checkpoint 加入集成。更低的训练 validation loss 没有转化为 local2 泛化，表明该条件实现与当前验证目标存在严重泛化失配；阶段捷径或条件分支过拟合是可能机制，但现有证据不足以单独归因。下一正式结构线转向 PNX-001 的层次局部几何编码。

## 已有线上反馈

| 方法 | local2 | 线上总分 | 线上 CD | 线上 P2S | 结论 |
|---|---:|---:|---:|---:|---|
| Graph StraightPCF | 旧代理 62.87 | 73.71 | - | - | 图卷解码带来最大单次结构收益 |
| module=4（`spcfgm`） | 旧代理 64.66 | 74.66 | - | - | module scaling 可迁移；module=5 OOM |
| C-noise extended | 72.06 | 75.01 | - | - | 噪声对齐和更长训练有效 |
| MS-A106 | 72.15 | 75.47 | 63.95 | 86.98 | 多尺度 distance，P2S 较强 |
| CVM-001 | 72.54 | 74.90 | 63.59 | 86.20 | stage target 单独使用不稳定 |
| CVM-002（alpha=1.0） | 72.77 | 75.51 | 64.15 | 86.88 | stage target + deep supervision 有效 |
| **CVM-002 A105** | **72.82** | **76.03** | **64.57** | **87.49** | **前一线上最佳；相同权重只把推理 alpha 调到 1.05** |
| **1:1 baseline/retrain ensemble + mean/var** | **73.03443** | **76.04** | **64.60** | **87.47** | **submission 39221；本地收益在线上缩小，但确认超过 76.03** |

## 2026-07-14：EMA、独立轨迹集成与 two-pass

同一训练轨迹同时保存 raw、EMA 0.99、0.995、0.999。best validation 位于 epoch 26，`val/loss_sum=4.169385`，epoch 56 在连续 30 个 epoch 无提升后停止。

| 方法 | local2 总分 | 相对 mean/var baseline | 95% CI | 结论 |
|---|---:|---:|---:|---|
| mean/var baseline | 72.99591 | - | - | 严格 reference |
| retrain raw + mean/var | 72.96777 | -0.02814 | `[-0.05237,-0.00542]` | 单模型否定，但误差可用于集成 |
| EMA 0.99 + mean/var | 72.90664 | -0.08927 | `[-0.15262,-0.02370]` | 否定 |
| EMA 0.995 + mean/var | 72.90787 | -0.08803 | `[-0.15131,-0.02802]` | 否定 |
| EMA 0.999 + mean/var | 72.90402 | -0.09188 | `[-0.15955,-0.02297]` | 否定 |
| baseline/retrain 1:1 ensemble + mean/var | 73.03443 | +0.03852 | `[+0.02113,+0.05618]` | 线上 76.04，保留 |
| 1:1 ensemble + baseline two-pass residual | 73.57402 | 相对原 two-pass +0.01130 | `[+0.00189,+0.02124]` | submission 39222 待计算 |
| 75/25 ensemble + fixed two-pass | 73.58214 | 相对 1:1 combo +0.00811 | `[+0.00282,+0.01326]` | 包已生成；submission 39225 因每日上限被拒 |
| 75/25 ensemble + conservative beta schedule | **73.62371** | 相对 fixed +0.04157 | `[-0.00400,+0.09171]` | 上一 local2 最好；已被 CV 调度超过 |
| 75/25 ensemble + CV-adaptive beta schedule | **73.67212** | 相对 conservative +0.04841 | `[+0.01172,+0.08920]` | CD/P2S 同升；上一版 200 云候选包已校验 |
| 55/10/35 baseline/retrain/seed456 + raw-alpha gate + CV two-pass | **73.71935** | 相对 75/25 CV +0.04723 | `[+0.02507,+0.07160]` | 当前 local2 最好；CD/P2S 同升，待生成 200 云包 |

补充否定结论：逐点 checkpoint disagreement 校准为 72.66690（-0.32900）；高残差云增加 beta 的 adaptive schedule 为 73.51723（-0.06491）。local3 的 40 云全部命中 alpha 下界 gate，固定与 conservative beta 都使用 `-0.30`，因此该集合对 regular-beta 调度无区分力。

CV-adaptive 调度来自 AID（arXiv:2509.14560）“用预测 score 模长方差估计噪声并安排迭代步长”的思想。这里不引入新模型，只用第二次修正模长的变异系数 `CV=std/mean` 做鲁棒标准化，再令 `beta=clip(0.45*(1+0.50*z), 0.30, 0.60)`；该阶段仍沿用 clipped-alpha 下界 gate，后续三轨迹扫描才升级为 raw-alpha gate。历史完整 beta 网格的五折回放相对 fixed 为 +0.12966，五折选择的 gamma 均为正（0.425～0.600）；在当前 75/25 锚点上的严格实测增益为 +0.04841。按类别 leave-one-out 时 12 个留出结果仍全部为正（+0.03782～+0.05894），因此收益不是由单个类别驱动。

- 提交包：`submission_results/result_baseline075_retrain025_cv_twopass_g050.zip`
- SHA256：`8b171a4e92a18a1360d10b86783c02c302bf01235e5919ad9a0809a3e719c272`

75/25 baseline/retrain 权重空间 model soup 被否定：soup adaptive / fixed two-pass / conservative two-pass 分别为 72.78194 / 73.37710 / 73.39222，相对当时 73.62371 reference 为 -0.84177 / -0.24661 / -0.23148，三个配对区间均全负。说明两个独立轨迹虽可在输出空间降低误差，参数线性插值会破坏已训练好的 surface-distance 几何。

Sampling Variation（arXiv:2411.01116）启发的 FPS 起点变化自集成也被否定：第二个 FPS 起点单独做 adaptive / fixed / CV two-pass 分别为 73.14340 / 73.51155 / 73.60107；再与原始 FPS 的 baseline/retrain 集成结果平均后做 CV two-pass 为 73.64981，仍比 73.67212 reference 低 0.02231，95% CI `[-0.08424,+0.01549]`。因此不提交、不保留专用推理接口和重预测目录，只保留 `experiments/sampling_variation/local2_comparison.tsv` 作为否定证据。

seed456 独立轨迹的单模型 adaptive 为 72.89614。直接使用 clipped-alpha gate 时，baseline/retrain/seed456 的初步 2:1:1 集成为 73.68104，仅比 73.67212 高 0.00892，且 CI 跨零。进一步检查发现，seed456 权重达到 10% 后有一云的裁剪前 alpha 仅从 0.97004 降到 0.96981，却因 `alpha==0.97` 使 beta 从正向 CV 调度瞬间切为 -0.30，该云单独损失 1.85 分。这是裁剪边界造成的推理不连续，不是模型集成本身退化。

校准 manifest 现同时保存 clipped alpha 与 `raw_alpha`，保护 gate 改为 `raw_alpha<=0.96`。阈值存在明确的独立分布间隔：最终 local2 候选的最低 raw alpha 为 0.96881，local3 的 40 云最高仅 0.94972；因此阈值在 `(0.94972,0.96881)` 内移动时两套结果均逐文件不变，0.96 不是单点调参。local3 上完全取消 gate 会从 69.89368 降到 66.58939（-3.30429），而 raw-alpha gate 与旧保护输出逐文件一致、分数完全相同；因此它只修复边界误判，不移除低质量云保护。

固定 raw-alpha gate 与 CV gamma=0.50 后，seed456 权重从 5% 增到 35% 时收益平滑上升，并在 35%～45% 形成平台；最终选择 baseline/retrain/seed456=`55/10/35`，而不是数值只高 0.00082 的 `50/10/40`，因为前者 CD/P2S 同升且高权重端 P2S 回撤更小。最终 local2 为 CD/P2S/总分 `57.26900 / 90.16970 / 73.71935`，相对 73.67212 为 +0.04723，95% CI `[+0.02507,+0.07160]`；44/62 云提高，12 个类别 leave-one-out 增益全部为正（+0.04124～+0.05367）。

Educoder 当日有效额度为团队每天两次：`39221` 已完成，`39222` 待计算；第三次 `39225` 返回“今日已达提交上限 2 次”。

## 历史校准基线：按云 mean/var 校准

固定 alpha 对不同噪声强度并非都最优。最终校准器只读取 noisy 和模型预测，使用：

1. `log(mean(||prediction-noisy||))`；
2. `log(var(||prediction-noisy||))`。

Ridge 输出每云 alpha，并裁剪到 `[0.97, 1.10]`。它不读取 clean 或 mesh，也不修改 checkpoint。

| 验证口径 | 原始/固定 alpha | mean/var adaptive | 增益 |
|---|---:|---:|---:|
| local2 五折留出 | 72.87189 | 72.99553 | **+0.12364**，95% CI `[+0.07856,+0.17084]` |
| 独立 20 云 | 77.35058 | 77.76446 | **+0.41388**；CD/P2S 均提高 |
| V4CTRL 完整 62 云 | 72.82574 raw | 73.00042 adaptive | 作为近期训练消融 control |

最终 200 云候选 alpha 范围为 `0.97～1.08950`，均值 `1.05542`，没有样本触及 1.10 上限。

- ZIP：`submission_results/result_cvm002a105_adaptive_meanvar_a110.zip`
- SHA256：`073f4b23bed59ce5dc237a6a2e4aaba7d17e13f4b2e68843532ca98e71c30564`
- 状态：保留为 ensemble/two-pass 的校准 anchor；不再作为最终单独提交候选。

## 2026-07-13～14 严格训练消融

统一 control：

- raw：CD/P2S/总分 `56.93656507 / 88.71491679 / 72.82574093`
- adaptive：`57.17678685 / 88.82405055 / 73.00041870`

| 方法 | raw 总分 | adaptive 总分 | adaptive-control | 总分 95% CI | 决策 |
|---|---:|---:|---:|---:|---|
| Normal auxiliary head | 72.82839 | 72.99793 | -0.00249 | `[-0.01561,+0.01171]` | CD 小升、P2S 显著下降，否定 |
| SIMPC mirror consistency | 72.81372 | 72.96870 | -0.03172 | `[-0.07603,+0.01490]` | CD/P2S Pareto 变差，否定 |
| HybridPF short residual | 72.71474 | 72.93875 | -0.06167 | `[-0.09919,-0.02629]` | CD/P2S 均下降，否定 |
| ROB010 Huber endpoint | 72.82483 | 72.99279 | -0.00763 | `[-0.02319,+0.00870]` | 裁剪难点梯度损害 P2S，否定 |
| CORE256 中心监督 | 72.82799 | 72.99724 | -0.00317 | `[-0.01701,+0.01148]` | 忽略约 13% 实际拼接输出，否定 |
| CORE384 中心监督 | 72.82765 | 73.00200 | +0.00158 | `[-0.00427,+0.00773]` | 覆盖约 96.5% 输出仍无实质收益，否定 |
| TSTRATA 端点分层采样 | 72.81380 | 72.99515 | -0.00527 | `[-0.03428,+0.02165]` | P2S 上升但 CD 下降，否定 |

### GD-GCN / UGD 轻量迁移

为排除训练配方影响，双图解码器与普通 graph decoder 使用相同初始化、AdamW、warmup/cosine 和固定验证随机种子。双图新增 position-kNN 消息分支，并通过零初始化投影保证初始函数与 control 一致。

| 方法 | raw 总分 | adaptive 总分 | 相对对应 control | 总分 95% CI | 决策 |
|---|---:|---:|---:|---:|---|
| V5 graph control | 72.82322 | 72.99500 | - | - | 训练配方未超过 V4CTRL |
| V5 feature + position 双图 | 72.82248 | 72.98807 | -0.00693 | `[-0.02200,+0.00900]` | 未达到 `+0.05` 门槛，删除实现与权重 |
| UGD-lite pristine GMM 选 alpha | - | 72.70767 | -0.28733（相对 V5 mean/var） | `[-0.38192,-0.19491]` | 21/62 云选择下界 0.97，明显过度收缩，删除 |

双图的 raw 差值为 -0.00074，95% CI `[-0.01622,+0.01540]`；说明轻量 position graph residual 并未减少足以反映到 CD/P2S 的跨表面误连。UGD-lite 只用手工局部描述子，不能替代论文中学习到的 pristine prior 与质量预测器。

其他表面监督：

| 方法 | 关键结果 | 决策 |
|---|---|---|
| correspondence normal | SURF-050 比同配方 SURF-000 仅 +0.00077，P2S -0.00751 | 无可靠收益 |
| bilateral IMLS | 五折 IMLS-control = -0.01102，95% CI `[-0.02443,+0.00322]` | 否定 |
| Virtual Normal | smoke test 后未进入完整训练 | 法向/IMLS 已无收益，不继续消耗预算 |

## 推理后处理消融

| 方法 | 结果 | 决策 |
|---|---|---|
| 六特征 geometry Ridge | local2 OOF 较高，但独立 20 云迁移弱于 mean/var | 疑似过拟合，删除 |
| tangent repulsion | local2 CD 上升但 P2S 等量下降；独立集总分约 -0.01 | 手工更新不稳定，删除 |
| 高位移点逐点 alpha | OOF 相对 mean/var -0.00701 | 高位移也包含真实边缘，删除 |
| Noise2Score3D 式 TV-PC 选 alpha | 62/62 云都偏向更大 alpha，52 云直接选上界 1.10 | 偏向过度平滑，删除 |
| 单特征/二次 Ridge | 最多只比 mean/var 高约 0.006，且没有独立集证据 | 不为微小闭环差异增加实现 |

## 历史未提交或未完成

| 方法 | local2/状态 | 结论 |
|---|---|---|
| LDC matched-unroll | 70.55，CD 54.50 / P2S 86.60 | 明显损害覆盖，未提交 |
| CVM-006 | 72.72，历史权重已清理 | 最值得补历史线上验证，但优先级低于当前 adaptive 候选 |
| CVM-008 | 72.60，曾生成 ZIP，无线上记录 | 低于 CVM-002 |
| CVM-009/010 conditioning | 因 NCCL 下载 HTTP 429 未完成/未启动 | 不是模型结论，但 conditioning 路线已降级 |
| PD-001～004 | 71.03～71.40 | pointwise distance gate 路线否定 |

## 下一步优先级

1. **带不确定性的自适应步长**：mean/var 是目前唯一跨口径稳定迁移的增益，可进一步让置信度头预测每云或每 patch 步长，并用 CD/P2S 联合验证。
2. **曲率感知、可学习的点分布项**：手工 tangent repulsion 证明 CD 仍有提升空间，但必须让网络联合约束 P2S。
3. **法向/曲率域消息传递**：纯 position-kNN 双图已否定；若再探索图结构，应直接构造切平面或曲率邻接，而不是增加另一个坐标图分支。
4. **U-CAN/Noise2Noise 一致性预训练**：利用 noisy-only 数据扩大训练分布，成本较高，排在校准改进之后。
5. 提交预算允许时，优先线上验证当前 adaptive ZIP；不再优先投入普通 attention、多趟推理、TTA 或更宽网络。
