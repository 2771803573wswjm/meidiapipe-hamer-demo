# Hamer + MediaPipe Video Demo

这个目录是 `MediaPipe + HaMeR` 视频端到端路线的独立包装版本，目标是：

- 输入视频
- 输出 3D 手部渲染视频
- 同时输出每一帧的手框、手性、MANO 参数、弱透视相机参数

## 1. 仓库里包含什么

仓库内建议保留这些内容：

- `video_demo.py`
- `run_video_demo.py`
- `hand_crop.py`
- `hamer/`
- `wildhands/common/data_utils.py`
- `mediapipe_model/hand_landmarker.task`
- `prepare_assets.py`

## 2. 仓库里不建议直接提交什么

下面这些不建议直接放进 GitHub 仓库：

- `downloads/_DATA/hamer_ckpts/checkpoints/hamer.ckpt`
  - 体积太大，本地大约 `2.6G`
- `downloads/_DATA/data/mano/...`
  - 这类 MANO 资源通常也不建议直接公开提交
- 任何你自己的测试视频和输出结果

所以更推荐的方案是：

- GitHub 仓库只放代码和轻量模型
- 用 `prepare_assets.py` 把用户已有资源复制到正确位置
- 或者你后面把大权重单独放在 release / 网盘 / 私有资源路径，再让用户下载后执行同一个准备命令

## 3. 目录约定

运行前，项目根目录下建议准备成下面这样：

```text
Hamer+mediapipe/
├── downloads/
│   ├── _DATA/
│   │   ├── hamer_ckpts/
│   │   │   ├── model_config.yaml
│   │   │   └── checkpoints/
│   │   │       └── hamer.ckpt
│   │   └── data/
│   │       ├── mano_mean_params.npz
│   │       └── mano/
│   │           └── MANO_RIGHT.pkl
│   └── vedio/
│       ├── vedio1.mp4
│       └── vedio2.mp4
├── mediapipe_model/
│   └── hand_landmarker.task
└── ...
```

## 4. 资源准备

如果你本机已经有原项目的 `downloads`，最简单：

```bash
cd "/home/wjmlinus/hand_project/3dhand-demo/Hamer+mediapipe"
python prepare_assets.py \
  --source_downloads "/home/wjmlinus/hand_project/paper reproduction/hands/downloads"
```

只复制核心 HaMeR 资源：

```bash
python prepare_assets.py \
  --source_downloads "/your/source/downloads" \
  --copy_videos false
```

这个脚本会把需要的目录和文件复制到当前项目自己的 `downloads/` 下。

## 5. 输出内容

每个视频都会生成：

- `*_render.mp4`
  - 叠加了 3D 手部渲染的视频
- `*_bbox.mp4`
  - 手框和左右手标签调试视频
- `*_frames.jsonl`
  - 每帧一条 JSON 记录
- `*_frame_params/000000.json`
  - 每帧一个独立 JSON 文件
- `*_summary.json`
  - 整段视频的汇总信息

## 6. 每帧 JSON 里的关键字段

- `candidates`
  - 当前帧手框、左右手标签、`track_id`
- `render.hands[*].pred_cam_crop`
  - HaMeR 的弱透视相机参数，已经按最终左右手符号修正
- `render.hands[*].pred_cam_crop_raw`
  - 模型原始弱透视相机参数
- `render.hands[*].cam_t_full`
  - 投回整图后的相机平移
- `render.hands[*].mano.global_orient_rotmat`
  - MANO 全局朝向旋转矩阵
- `render.hands[*].mano.hand_pose_rotmat`
  - MANO 15 个手部关节旋转矩阵
- `render.hands[*].mano.betas`
  - MANO 形状参数

## 7. 推荐命令行参数

最常用的参数建议固定成这几个：

- `--video_path` 或 `--video_folder`
- `--out_folder`
- `--auto_focal_search`
- `--auto_focal_frames 10`
- `--auto_focal_stride 5`
- `--no_mediapipe_auto_handedness`

双视频批量跑：

```bash
cd "/home/wjmlinus/hand_project/3dhand-demo/Hamer+mediapipe"
source /home/wjmlinus/miniconda3/etc/profile.d/conda.sh
conda activate hands-dev

CUDA_VISIBLE_DEVICES=0 python run_video_demo.py \
  --video_folder "downloads/vedio" \
  --out_folder "out_video_demo" \
  --auto_focal_search \
  --auto_focal_frames 10 \
  --auto_focal_stride 5 \
  --no_mediapipe_auto_handedness
```

单视频运行：

```bash
CUDA_VISIBLE_DEVICES=0 python run_video_demo.py \
  --video_path "downloads/vedio/vedio1.mp4" \
  --out_folder "out_video_demo_single" \
  --auto_focal_search \
  --auto_focal_frames 10 \
  --auto_focal_stride 5 \
  --no_mediapipe_auto_handedness
```

如果你想显式指定资源路径，也可以直接传：

```bash
CUDA_VISIBLE_DEVICES=0 python run_video_demo.py \
  --video_path "downloads/vedio/vedio1.mp4" \
  --out_folder "out_video_demo_single" \
  --hamer_ckpt "downloads/_DATA/hamer_ckpts/checkpoints/hamer.ckpt" \
  --mediapipe_model "mediapipe_model/hand_landmarker.task"
```

## 8. 说明

- 视频版当前更建议 `--no_mediapipe_auto_handedness`
  - 因为视频序列通常更适合直接信 `MediaPipe` 的左右手，再配合时序平滑
- 左手的 MANO 参数仍然是 HaMeR 的规范化输出
  - 最终真实手性请结合 `is_right` 一起解释
