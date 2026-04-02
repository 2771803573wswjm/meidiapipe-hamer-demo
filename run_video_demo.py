from pathlib import Path
import os


ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = ROOT / "downloads" / "_DATA"
DEFAULT_MANO_DIR = DEFAULT_CACHE_DIR / "data"
DEFAULT_CKPT = DEFAULT_CACHE_DIR / "hamer_ckpts" / "checkpoints" / "hamer.ckpt"
DEFAULT_MODEL_CFG = DEFAULT_CACHE_DIR / "hamer_ckpts" / "model_config.yaml"
DEFAULT_MEDIAPIPE_MODEL = ROOT / "mediapipe_model" / "hand_landmarker.task"


def prepare_local_env():
    if "CACHE_DIR_HAMER" not in os.environ and DEFAULT_CACHE_DIR.exists():
        os.environ["CACHE_DIR_HAMER"] = str(DEFAULT_CACHE_DIR)
    if "HAMER_MANO_DIR" not in os.environ and DEFAULT_MANO_DIR.exists():
        os.environ["HAMER_MANO_DIR"] = str(DEFAULT_MANO_DIR)


def validate_local_assets():
    required = [
        DEFAULT_CKPT,
        DEFAULT_MODEL_CFG,
        DEFAULT_MANO_DIR / "mano_mean_params.npz",
        DEFAULT_MANO_DIR / "mano" / "MANO_RIGHT.pkl",
        DEFAULT_MEDIAPIPE_MODEL,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        message = [
            "Missing required local assets:",
            *[f"- {path}" for path in missing],
            "",
            "Prepare assets first, for example:",
            "python prepare_assets.py --source_downloads \"/path/to/existing/downloads\"",
        ]
        raise FileNotFoundError("\n".join(message))


def main():
    prepare_local_env()
    validate_local_assets()
    from video_demo import main as video_main

    video_main()


if __name__ == "__main__":
    main()
