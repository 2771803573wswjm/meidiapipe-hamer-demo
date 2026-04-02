from pathlib import Path
import argparse
import shutil


ROOT = Path(__file__).resolve().parent
TARGET_DOWNLOADS = ROOT / "downloads"


def copy_tree(src: Path, dst: Path):
    if not src.exists():
        raise FileNotFoundError(f"Missing source path: {src}")
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def str2bool(value: str) -> bool:
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def validate_downloads_tree(downloads_root: Path):
    required = [
        downloads_root / "_DATA" / "hamer_ckpts" / "checkpoints" / "hamer.ckpt",
        downloads_root / "_DATA" / "hamer_ckpts" / "model_config.yaml",
        downloads_root / "_DATA" / "data" / "mano_mean_params.npz",
        downloads_root / "_DATA" / "data" / "mano" / "MANO_RIGHT.pkl",
    ]
    missing = [str(path) for path in required if not path.exists()]
    return missing


def main():
    parser = argparse.ArgumentParser(description="Prepare local assets for the Hamer+MediaPipe demo")
    parser.add_argument("--source_downloads", type=str, required=True, help="Path to an existing downloads/ directory")
    parser.add_argument("--copy_videos", type=str2bool, default=True, help="Whether to also copy downloads/vedio")
    args = parser.parse_args()

    source_downloads = Path(args.source_downloads).expanduser().resolve()
    if not source_downloads.exists():
        raise FileNotFoundError(f"Source downloads directory does not exist: {source_downloads}")

    missing = validate_downloads_tree(source_downloads)
    if missing:
        raise FileNotFoundError(
            "Source downloads directory is missing required files:\n- " + "\n- ".join(missing)
        )

    TARGET_DOWNLOADS.mkdir(parents=True, exist_ok=True)
    copy_tree(source_downloads / "_DATA", TARGET_DOWNLOADS / "_DATA")

    video_dir = source_downloads / "vedio"
    if args.copy_videos and video_dir.exists():
        copy_tree(video_dir, TARGET_DOWNLOADS / "vedio")

    print(f"Prepared assets under: {TARGET_DOWNLOADS}")
    print(f"HaMeR checkpoint: {TARGET_DOWNLOADS / '_DATA' / 'hamer_ckpts' / 'checkpoints' / 'hamer.ckpt'}")
    print(f"MANO dir: {TARGET_DOWNLOADS / '_DATA' / 'data'}")
    if args.copy_videos and video_dir.exists():
        print(f"Videos copied to: {TARGET_DOWNLOADS / 'vedio'}")


if __name__ == "__main__":
    main()
