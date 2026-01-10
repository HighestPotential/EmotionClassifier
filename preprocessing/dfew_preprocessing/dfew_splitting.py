from pathlib import Path
import random
import shutil

EMOTIONS = ["happiness", "sadness", "anger", "surprise", "disgust", "fear"]

def list_images(folder: Path):
    return [p for p in folder.rglob("*") if p.is_file()]

def copy_all(src: Path, dst: Path):
    if not src.exists():
        return
    for p in list_images(src):
        rel = p.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)

def main():
    project_root = Path(__file__).resolve().parents[2]

    src_root = project_root / "preprocessed_dataset" / "DFEW"
    dst_root = project_root / "ready_to_use_datasets" / "DFEW"

    seed = 42
    eval_ratio = 0.10

    random.seed(seed)

    for split in ["train", "eval", "test"]:
        for emo in EMOTIONS:
            (dst_root / split / emo).mkdir(parents=True, exist_ok=True)

    for emo in EMOTIONS:
        src_emo = src_root / "train" / emo
        files = list_images(src_emo)
        random.shuffle(files)

        n_eval = int(len(files) * eval_ratio)
        eval_files = set(files[:n_eval])
        train_files = files[n_eval:]

        for p in train_files:
            shutil.copy2(p, dst_root / "train" / emo / p.name)

        for p in eval_files:
            shutil.copy2(p, dst_root / "eval" / emo / p.name)

    copy_all(src_root / "test", dst_root / "test")

    print("Source :", src_root)
    print("Output :", dst_root)
    print("Done. Eval ratio:", eval_ratio, "Seed:", seed)

if __name__ == "__main__":
    main()