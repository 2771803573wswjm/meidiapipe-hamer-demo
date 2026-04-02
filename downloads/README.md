这个目录默认用于存放运行所需的大资源文件，但不建议直接提交到 GitHub。

推荐结构：

```text
downloads/
├── _DATA/
│   ├── hamer_ckpts/
│   │   ├── model_config.yaml
│   │   └── checkpoints/
│   │       └── hamer.ckpt
│   └── data/
│       ├── mano_mean_params.npz
│       └── mano/
│           └── MANO_RIGHT.pkl
└── vedio/
    ├── vedio1.mp4
    └── vedio2.mp4
```

如果你本机已经有这些资源，可以执行：

```bash
python prepare_assets.py \
  --source_downloads "/path/to/existing/downloads"
```
