# 最终方法

最终方案位于 `solution/`，整体图见 [README](README.md)。旧版方法说明已保存在 [docs/history/METHOD.md](docs/history/METHOD.md)。

## 成员组织

两个 3DMambaIPF 使用相同四阶段迭代架构，通过数据构造和 checkpoint 形成预测差异；IterativePFN 提供结构互补。StraightPCF VM 只提供参考位移。网络结构来源、作者和许可证见 [第三方说明](docs/THIRD_PARTY.md)。

## 两级融合

1. 每个主体对重叠 patch 分别预测，按归一化中心距离以温度 0.10 进行 softmax 拼接，按输入索引写回。
2. 对两个 Mamba 与一个 IPFN 的完整输出逐坐标取中位数。
3. 从中位数减去 `0.01 * (VM - Input)`，保持点数和顺序。

融合只使用含噪输入和成员输出，不需要测试干净点云。中位数对应各坐标绝对偏差和的最小化；负残差的系数为既定验证选择，不将其解释为普适理论保证。

## 实现与训练

Mamba selective scan 采用 `jt.code` CUDA 前向和显式反向，每 16 个扫描序列位置保存状态、反向分段重算。IPFN 使用阶段最近邻位移监督与基于采样点的度量代理，权重前 2 epoch 从 0 升到 0.2，按停止梯度的损失比缩放。

公开训练文件及已知重训边界见 [TRAINING.md](docs/TRAINING.md)。本次整理没有新测收敛曲线、消融指标或真实降噪可视化。
