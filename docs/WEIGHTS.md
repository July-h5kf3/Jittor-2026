# 发布权重

下载：[v1.0.0 Release](https://github.com/July-h5kf3/Jittor-2026/releases/tag/v1.0.0)。压缩包约 144 MB，解压目录为 `weights/`。

推荐在 `solution/` 中运行 `python tools/download_weights.py`；也可下载 ZIP 后传入 `--archive`。权重字节与团队最终项目文件一致，不做重新转换。

| 路径 | 字节数 | SHA-256 |
|---|---:|---|
| `weights/ipfn_1024/denoisenet-jittor-b-epoch10.npz` | 12960494 | `72ae0220dcb46d61ae010e9a03c46938f4fae78752f2de6a820f2ee18fd6a333` |
| `weights/mamba_mix75_s48000/step-048000-jittor.pkl` | 65231587 | `e8b1214ce44e3685d3110dabd25189216203d89b5893b9622879bf7153985a46` |
| `weights/mamba_v8192_s060000/step-060000-jittor.pkl` | 65231656 | `a76d60b8fe73268b5b4ba714711d22c67e2491394124c24e0123b2f326704ea4` |
| `weights/vm_fixed/fixed_v8192_epoch10.npz` | 951942 | `2f62b02039f7bc27bea352c29e6008fc553d67527fee9cd9a0bc11e750341467` |

压缩包 SHA-256：`ae2481a4abd47bda439c503d9692bf24582bd6c03b87430dea756e2d0d9bb22f`。

权重不进入 Git 历史。配置 JSON 与校验 manifest 随源码提供。训练入口、配置与来源引用分别见 [TRAINING.md](TRAINING.md) 与 [THIRD_PARTY.md](THIRD_PARTY.md)。
