# 推理复现指南

当前入口为 `solution/`，不是根目录早期 `run.py`。目标环境：Ubuntu 22.04、Python 3.10、NVIDIA GPU、CUDA Toolkit 12.4、Jittor 1.3.11.0。项目记录的目标设备为 RTX 4090 24GB；本次开源整理未重新测定最低显存或运行时间。

## 安装与模型

在系统配置好 CUDA 与 C++ 编译工具后：

```bash
cd solution
conda create -n nkai-jittor python=3.10 -y
conda activate nkai-jittor
pip install -r requirements.txt
python tools/download_weights.py
```

`environment.yaml` 保留原检查环境，包含部分扩展依赖；快速推理优先使用 `requirements.txt`。IPFN 推理使用 SciPy 的 cKDTree，因此当前三项运行依赖为 Jittor、NumPy、SciPy。

也可在 Releases 手动下载 ZIP，然后运行 `python tools/download_weights.py --archive /path/to/nkai-final-weights.zip`。该命令同时检查压缩包和四个 checkpoint 的 SHA-256。只加载可信来源的 checkpoint；`.pkl` 属于可反序列化的模型文件。

## 官方测试输入

```text
<INPUT_ROOT>/shapenet/00000000/btest_000001/noisy.npy
...
<INPUT_ROOT>/shapenet/00000000/btest_000200/noisy.npy
```

每个输入为有限值 `float32 (50000, 3)`。数据由赛事渠道取得，仓库不分发数据或测试真值。`datalist/test.txt` 是原仓库历史数据标识，不应直接替代最终测试清单。

```bash
export INPUT_ROOT=/path/to/official_test
python tools/prepare_keys.py --input-root "$INPUT_ROOT" --expected-count 200
python tools/check_release.py --input-root "$INPUT_ROOT"
WORLD_MAMBA=1 WORLD_IPFN=1 WORLD_VM=1 bash scripts/run_reproduce.sh
```

清单从真实存在的 `noisy.npy` 路径生成；检查失败时不启动 GPU 推理。最终运行链依次执行 Mamba 48K、Mamba 60K、IPFN、VM、融合和打包。结果在 `solution/result.zip`，校验在 `solution/result_audit.json`，日志在 `solution/logs/`。现有 `result.zip` 不会被打包器覆盖；重跑时请先另存旧结果。

## 分成员与多卡

```bash
bash scripts/infer_mamba.sh 48k 1
bash scripts/infer_mamba.sh 60k 1
bash scripts/infer_ipfn.sh 1024 1
bash scripts/infer_vm.sh 1
bash scripts/fuse.sh
```

最后一个参数为进程/设备分片数。多卡时可以设置 `WORLD_MAMBA`、`WORLD_IPFN`、`WORLD_VM`。原脚本每个 rank 默认绑定同编号 GPU；使用多卡脚本时不要同时设置一个包含多 GPU 的 `CUDA_VISIBLE_DEVICES` 列表，以免多个进程误用同一逻辑设备。单卡默认使用设备 0，可以显式设置单个设备 ID。

公开权重仅包含 IPFN 1024；原脚本的 8192 选项属于历史接口，本版本未发布该成员权重。

## 自定义数据

最终打包器明确限定赛事的 `btest_*` 命名、200 例和每例 50,000 点。自定义点云可以参考分成员入口，但需要自行调整点数、清单、期望数量与打包规则，不能将 quickstart 当成任意数据集的通用评测器。

## 验证边界

本次发布检查涵盖 Python 语法、Shell 语法、真实目录生成清单的约束、合成点云的中位数与负残差算术、点顺序及 ZIP 字节一致性、发布权重 SHA-256。合成输入不是效果对比。GPU 算子数值、完整推理、从头训练和官方成绩复现仍需在目标 GPU 环境运行；本次没有新测这些结果。
