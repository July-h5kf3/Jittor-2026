# 精简 Jittor StraightPCF 模型子树

本目录只保留 `train_rot.py` 和 `generate_rot.py` 所需的 Jittor 模型模块：

- `src/model/straightpcf.py`
- `src/model/feature.py`
- `src/model/vm.py`
- `src/model/spec.py`
- `src/data/asset.py`

原研究仓库的通用 OmegaConf runner、SciPy/trimesh augmentation、tqdm/wandb system 和其他未使用模块均未提交。

- `../train_rot.py` 使用 Jittor、NumPy 和标准库直接实现 mesh 读取、surface sampling、噪声、旋转、patch 构建、VM → CVM → SPCF checkpoint chaining 和可恢复训练。
- `../generate_rot.py` 直接实现冻结 mean-var calibration、pass-2 和 CV-adaptive fusion。

正式 ROT 路径不依赖 PyTorch、Triton、SciPy、trimesh、OmegaConf 或第三方点云 runtime。来源与许可证见 `../LICENSES.md`。
