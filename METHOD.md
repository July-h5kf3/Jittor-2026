# 最终方法：CVM-002

## 总览

CVM-002 以 StraightPCF 为主干，将点云去噪表示为带噪点沿速度场移动到干净表面的过程。最终模型依次训练 CVM 和 SPCF 两个阶段：

1. CVM：4 个耦合 velocity modules 学习分阶段速度。
2. SPCF：冻结/继承速度网络并训练 multi-scale distance head，估计剩余移动步长。

## 编码与解码

- 编码器：三层 EdgeConv，特征维度 256，kNN=32。
- 解码器：在特征空间重新建立 kNN 图的 graph decoder。
- FiLM：向特征注入噪声相关条件。

图卷解码使邻近点的速度预测保持局部一致，是从早期基线提升到 73+ 官方分的关键结构改动。

## Stage velocity 与深监督

传统 full-residual target 让每个 velocity module 都预测完整剩余位移。CVM-002 使用 `stage_velocity`，让第 `m` 个模块只预测该阶段应完成的局部移动，并通过 `cvm_deep_sup` 对中间 waypoint 增加监督。

对应配置：

```yaml
num_modules: 4
decoder_type: graph
film: true
cvm_dir_target: stage_velocity
cvm_deep_sup: true
```

CVM-001 只使用 stage velocity，线上为 74.90；加入 deep supervision 后，CVM-002 提升至 75.51，说明两者需要组合使用。

## Distance head

SPCF 阶段使用 multi-scale distance encoder：

```yaml
stage: spcf
distance_multiscale: true
predict_alpha: 1.0
predict_passes: 1
predict_tta: 0
predict_fusion: false
```

最终推理坚持单次、无 TTA、无普通融合，因为此前的推理增强多次出现 local/online 背离。

## 训练分布

- 训练噪声标准差：`0.008–0.014`。
- outlier：2%，幅度系数 3.0。
- 随机缩放：`0.8–1.2`。
- 随机 SO(3) 旋转。
- patch size：1000。

训练 CVM-002 需要 `75.01` 模型的 CVM checkpoint 作为初始化，然后将新的 CVM 最佳权重用于 SPCF 阶段。
