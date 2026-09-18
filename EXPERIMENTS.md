> 历史实验记录：以下内容对应原仓库早期探索，保留原值与结论以便追溯。当前最终方案与入口见 [README](README.md)，请勿将不同验证集合的数值直接比较。

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
| **CVM-002** | **72.77** | **75.51** | **64.15** | **86.88** | stage target 与 deep supervision 的组合最佳 |

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

1. 先补 CVM-006 线上验证，确认 local2 `72.72` 是否能迁移。
2. 做 CVM-002/MS-A106 的预测位移融合，只扫 3 个权重。
3. 修复 NCCL cache 后直接训练 CVM-010；它是对当前最佳方案最干净的结构消融。
4. 若 conditioning 无效，再实现坐标图 + 特征图双图解码器。
