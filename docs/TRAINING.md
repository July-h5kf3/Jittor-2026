# 训练入口与配置来源

本页汇总最终方案的训练入口与成员配置，配置依据为项目提交材料、权重元数据和实现代码。发布的四成员权重可直接用于[推理复现](REPRODUCE.md)。

| 成员 | 入口 | 配置/预算 |
|---|---|---|
| Mamba mix75 | `solution/code/mamba/jittor_core/train.py` | Adam；5e-5 → 2e-6；48K step；≤1024 顶点；75% 尺度抖动 |
| Mamba v8192 | 同上 | Adam；5e-5 → 2e-6；60K step；≤8192 顶点；关闭尺度抖动 |
| IPFN 1024 | `solution/code/ipfn/train_jittor.py` | lr=3e-6；10 epoch；metric_weight=0.2；前 2 epoch 预热 |
| VM fixed_v8192 | `solution/code/vm/train_vm.py` | Adam；5e-5 → 5e-7；10 epoch；500 step 预热；seed=20260819 |

IPFN 元数据：[epoch10.json](../solution/weights/ipfn_1024/epoch10.json)。VM 元数据：[fixed_v8192_training_config.json](../solution/weights/vm_fixed/fixed_v8192_training_config.json)。训练 patch 为 1000 点，梯度裁剪 1.0。四阶段 IPFN 的前三目标噪声尺度分别为 σ/4、σ/16、σ/64，末阶段为干净目标；度量项使用采样点距离代理，不等于官方网格 P2S。

## 训练准备

训练需先准备赛事训练 mesh、对应 datalist、CUDA 环境及模型初始化配置。进入 `solution/` 后可以查看各入口参数：

```bash
PYTHONPATH=code/mamba/jittor_core python code/mamba/jittor_core/train.py --help
PYTHONPATH=code/ipfn python code/ipfn/train_jittor.py --help
PYTHONPATH=code/vm python code/vm/train_vm.py --help
```

- Mamba 的 `train.py` 使用 `plr_training_loss`，实现阶段最近邻位移监督和可选 DCD；数据构造、训练步数与初始化方式应配套设置。
- IPFN 的 `--init-weights` 指定训练初始化权重，`--resume-weights` / `--resume-state` 用于恢复训练。Releases 中的 epoch 10 文件是最终推理权重，不能直接将其当成原训练起点。
- VM 训练入口从 `--data-root` 和 `--train-list` 读取 mesh 与样本划分；参数应与发布的 `fixed_v8192_training_config.json` 配套。

首页快速开始对应“下载最终权重并执行推理”。独立重新训练时，应同时记录初始权重、训练数据划分、完整命令与随机种子。`solution/a_board/` 中保留的历史 pipeline 与说明用于开发记录追溯，当前四成员配置以本页为索引。
