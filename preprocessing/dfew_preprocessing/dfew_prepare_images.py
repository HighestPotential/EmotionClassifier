import csv
import shutil
from pathlib import Path

LABEL_TO_EMOTION = {
    1: "happiness",
    2: "sadness",
    4: "anger",
    5: "surprise",
    6: "disgust",
    7: "fear",
}

FRAMES = [4, 8, 12, 16]

def read_items(csv_path: Path):
    items = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip_id = int(row["video_name"])
            label = int(row["label"])
            emotion = LABEL_TO_EMOTION.get(label)
            if emotion is not None:
                items.append((clip_id, emotion))
    return items

def ensure_structure(out_root: Path):
    for split in ["train", "test"]:
        for emo in LABEL_TO_EMOTION.values():
            (out_root / split / emo).mkdir(parents=True, exist_ok=True)

def copy_clip_frames(clip_root: Path, out_root: Path, split: str, clip_id: int, emotion: str):
    clip_dir = clip_root / f"{clip_id:05d}"
    if not clip_dir.is_dir():
        return 0
    copied = 0
    for fr in FRAMES:
        src = clip_dir / f"{fr}.jpg"
        if src.exists():
            dst = out_root / split / emotion / f"{clip_id:05d}_{fr}.jpg"
            shutil.copy2(src, dst)
            copied += 1
    return copied

def main():
    project_root = Path(__file__).resolve().parents[2]

    dfew_root = project_root / "datasets" / "DFEW"
    clip_root = dfew_root / "clip_224x224_16f"

    train_csv = dfew_root / "EmoLabel_DataSplit" / "train(single-labeled)" / "set_1.csv"
    test_csv  = dfew_root / "EmoLabel_DataSplit" / "test(single-labeled)" / "set_1.csv"

    out_root = Path(__file__).resolve().parent / "dfew_stage1"
    ensure_structure(out_root)

    train_items = read_items(train_csv)
    test_items = read_items(test_csv)

    train_copied = 0
    for clip_id, emo in train_items:
        train_copied += copy_clip_frames(clip_root, out_root, "train", clip_id, emo)

    test_copied = 0
    for clip_id, emo in test_items:
        test_copied += copy_clip_frames(clip_root, out_root, "test", clip_id, emo)

    print("FRAMES:", FRAMES)
    print("Output:", out_root)
    print("Train clips:", len(train_items), "| Frames copied:", train_copied)
    print("Test  clips:", len(test_items), "| Frames copied:", test_copied)

if __name__ == "__main__":
    main()