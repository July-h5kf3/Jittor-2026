# contest2_NKAI_031 最终检查表

## 根结构

- [ ] 文件名为 `contest2_NKAI_031.zip`。
- [ ] ZIP 根目录直接包含 `code/`、`requirements.txt`、`environment.yaml`、`提交说明文档.pdf`。
- [ ] 不额外包一层 `contest2_NKAI_031/` 目录。
- [ ] PDF 可打开，包含 NKAI、排名 031、79.23、三位联系人、环境、完整命令、超参数和已知问题。

## 内容合规

- [ ] 包内只有源码、配置、公开训练 key metadata 和文档。
- [ ] 无 `.pth/.pt/.ckpt/.pkl/.npy/.npz/.obj` 权重、数据或预测。
- [ ] 无结果 ZIP、日志、缓存、`.pyc` 或 `__pycache__`。
- [ ] 无账号、Cookie、token、密码、SSH key、私有绝对路径。
- [ ] 无历史 PyTorch checkpoint reader 或 migration exporter。
- [ ] 全部 Python 无 PyTorch/Triton/`mamba_ssm`/`causal_conv1d` runtime import。
- [ ] 正式路径无 SciPy/trimesh/OmegaConf/tqdm runtime import。

## 从原始数据复现

- [ ] `configs/reproduce_full.json` 可通过 `main.py pipeline --dry-run` 展开 39 个 stage。
- [ ] 所有 3DMambaIPF checkpoint 从 `base-random` 生成。
- [ ] StraightPCF 从 VM → CVM → SPCF 随机初始化训练。
- [ ] 不依赖包外 Full/ROT/expert prediction 或祖先 checkpoint。
- [ ] test key 自动发现为 200，route 子列表计数为 DCD84/table92/sofa30/airplane35/tail40。
- [ ] Full 和 PLR-003 route 公式未修改。
- [ ] 结果 ZIP 为 200 个 finite `float32 (50000,3)`。

## 目标环境

- [ ] Ubuntu 22.04。
- [ ] Python 3.10。
- [ ] CUDA Toolkit 12.4 / nvcc 可用。
- [ ] RTX 4090 / sm_89。
- [ ] Jittor 1.3.11.0，`jt.compiler.has_cuda=True`。
- [ ] `python main.py doctor --deep` 通过。
- [ ] selective scan forward/backward、训练一步、checkpoint save/load、ROT smoke、50K inference 和 prediction packaging 通过。
- [ ] Conda、pip、Jittor cache 与测试产物位于 `/root/autodl-tmp`。

## 本地静态审计

```bash
python -m compileall -q code
python code/tools/audit_source_archive.py --root code
```

打包脚本：

```bash
cd D:/Class/jittor/submission_packages
python build_submission_pdf.py \
  --input contest2_NKAI_031/code/docs/提交说明文档.md \
  --output contest2_NKAI_031/提交说明文档.pdf
python build_contest2_package.py
```

## 最终 ZIP

- [ ] CRC、成员唯一性、路径安全和 deterministic timestamp 通过。
- [ ] ZIP 大小和 SHA256 已写入外部 final audit。
- [ ] 精确 ZIP 在目标 4090 环境解压后重新通过 compile/source/CUDA smoke。
- [ ] 最终材料只由参赛者本人手动上传，自动化不登录或提交赛事网盘。
