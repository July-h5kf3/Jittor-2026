# 关键实验记录

本文只保留影响最终决策的实验。完整的逐小时运行日志、W&B 离线目录、重复 checkpoint 和中间预测已经清理。

## 评测口径

- 官方分：`0.5 × CD_score + 0.5 × P2S_score`。
- `local2`：62 个内部样本，用于快速淘汰明显失败方案。
- 已观察到本地与线上背离，因此小于 `0.1` 的 local2 差距不用于确定最终排序。

## 已有线上反馈

| 方法 | local2 | 线上总分 | 线上 CD | 线上 P2S | 结论 |
|---|---:|---:|---:|---:|---|
| Graph StraightPCF | 旧代理 62.87 | 73.71 | - | - | 图卷解码带来最大单次结构收益 |
| module=4 (`spcfgm`) | 旧代理 64.66 | 74.66 | - | - | module scaling 可迁移，module=5 OOM |
| C-noise extended | 72.06 | 75.01 | - | - | 噪声对齐和更长训练有效 |
| MS-A103 | 72.09 | 75.13 | - | - | alpha 小幅变化有效但有限 |
| MS-A106 | 72.15 | 75.47 | 63.95 | 86.98 | distance alpha=1.06，P2S 优势明显 |
| CVM-001 | 72.54 | 74.90 | 63.59 | 86.20 | stage target 单独使用不稳定 |
| CVM-002 (`alpha=1.0`) | 72.77 | 75.51 | 64.15 | 86.88 | stage target 与 deep supervision 的组合此前最佳 |
| **CVM-002 A105** | **72.82** | **76.03** | **64.57** | **87.49** | **相同权重仅校准推理步长，CD/P2S 同时提高，当前最佳** |

## 2026-07-13：LDC 与推理步长校准

### LDC / conditioning 消融

| 方法 | 训练口径 | 最佳验证 | local2 | CD | P2S | 决策 |
|---|---|---:|---:|---:|---:|---|
| LDC matched-unroll | 两步展开 + distance/stage adapter，冻结 velocity trunk | 4.3136（与历史单步口径不可比） | 70.55 | 54.50 | 86.60 | 明显损害点分布覆盖，否定 |
| LDC alpha 后处理 | 对 LDC 位移扫 `0.75～1.05` | - | 最高 70.96（alpha 0.90） | 55.29 | 86.64 | 不是简单步长过大，否定 |
| LDC-1 消融 | 恢复单步训练，仅保留 adapter | 4.4436 | 未评测 | - | - | 历史 CVM-002 同口径为 3.7942，提前停止 |

结论：matched-unroll 和 adapter 均未显示收益。LDC 的 zero-init 保证了安全初始化，但微调后主要保住 P2S、牺牲 CD，说明问题不是继续增加时间条件就能解决。

### CVM-002 `predict_alpha` 扫描

全部使用同一个 CVM-002 checkpoint、单次推理、无 TTA/融合：

| alpha | local2 CD | local2 P2S | local2 总分 | 相对 1.00 |
|---:|---:|---:|---:|---:|
| 0.94 | 56.94 | 88.19 | 72.57 | -0.20 |
| 0.97 | 56.99 | 88.39 | 72.69 | -0.08 |
| 1.00 | - | - | 72.77 | 基线 |
| 1.03 | 56.97 | 88.66 | 72.81 | +0.04 |
| **1.05** | **56.93** | **88.71** | **72.82** | **+0.05** |
| 1.06 | 56.91 | 88.74 | 72.82 | +0.05 |

`1.05` 取 1.03/1.06 平台区间的中点。2026-07-13 的 A 榜提交得到 **76.03**，相对 `alpha=1.0` 的 75.51 提升 **0.52**；CD 从 64.15 提升到 64.57，P2S 从 86.88 提升到 87.49。该结果确认了步长校准的收益，但也再次说明 local2 的小差距不能直接估计线上增益幅度。

## 已完成但没有线上分数

| 方法 | 变化 | local2 | 状态 | 决策 |
|---|---|---:|---|---|
| CVM-003 | full residual + deep supervision | 72.28 | 未生成提交包 | 不优先 |
| CVM-004 | stage/full blend=0.25 | 72.22 | 未生成提交包 | 不优先 |
| CVM-005 | stage/full blend=0.50 | 72.48 | 未生成提交包 | 次选 |
| **CVM-006** | **stage/full blend=0.75** | **72.72** | `SKIP_SUBMIT=1` | 最值得补线上验证 |
| CVM-007 | multi-scale velocity + full residual | 72.05 | `SKIP_SUBMIT=1` | 否定单纯多尺度 velocity |
| CVM-008 | multi-scale velocity + stage target | 72.60 | 已生成 zip，无线上记录 | 低于 CVM-002 |
| CVM-011 | CVM-002 + 更强训练噪声 | 71.97 | 未生成提交包 | 简单增大噪声失败 |
| MS-A110 | distance alpha=1.10 | 72.19 | 已生成 zip，无线上记录 | 可能过度去噪 |
| MS-002 | multi-scale distance + edge endpoint | 72.12 | 已生成 zip，无线上记录 | 收益不足 |
| MS-003 | 更大 edge loss | 71.88 | 已生成 zip，无线上记录 | 负向 |
| PD-001～004 | pointwise/residual/smooth distance gate | 71.03～71.40 | 已评测 | 整条路线暂时否定 |

## 未完成实验

- **CVM-009**：`full_residual + deep supervision + time/stage conditioning`。
  - 2026-07-07 启动时，新的 Jittor cache 尝试从 GitHub 下载 NCCL。
  - GitHub 返回 `HTTP 429 Too Many Requests`，训练在创建 checkpoint 前退出。
  - 因此这不是模型失败，condition 机制尚未得到实验验证。
- **CVM-010**：`CVM-002 + time/stage conditioning`。
  - 配置和实现已准备。
  - 因 CVM-009 失败后流水线直接退出，从未启动。

## 被否定或降级的方向

- 全局 attention、Point Transformer、朴素/修复版 3D RoPE：最高只回到基线附近。
- EMA + cosine LR：验证/模型选择不匹配，明显退化。
- module=5、dim=512：16GB 显存下 OOM。
- 两趟/三趟推理、普通 TTA、索引融合：本地收益不能迁移，部分线上下降。
- 全局 CD/P2S 辅助损失和过强 edge loss：同时损害 CD 与 P2S。
- 更强噪声区间：CVM-011 local2 下降到 71.97。

## 推荐后续顺序

1. 实现法向/表面感知 endpoint loss，直接对齐 P2S，同时约束 CD 覆盖。
2. 实现坐标图 + 特征图双图解码器，减少薄面跨表面误连。
3. 若需要补历史路线，再复训 CVM-006；不再优先投入 LDC/time-stage adapter。
4. 提交预算允许时，在 `alpha=1.03～1.06` 内做极少量线上校准；A105 已是默认最佳。
