from pathlib import Path
import cv2
import numpy as np
from pipeline import Pipeline
from skip_image import SkipImage  

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def collect_images(root: Path, max_files: int | None = None) -> list[Path]:
    """
    Recursively collects image paths from the directory tree, filtering by extension.

    Args:
        root (Path): The root directory to start searching from.
        max_files (int | None): Optional limit on the number of files to collect.

    Returns:
        list[Path]: A list of Path objects for all found images.
    """
    paths = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            paths.append(p)
            if max_files is not None and len(paths) >= max_files:
                break
    return paths

def filter_rate(pipe: Pipeline, input_dir: str, n: int = 2000):
    root = Path(input_dir)
    paths = collect_images(root, max_files=n)

    total = kept = filtered = failed = 0
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            failed += 1
            continue

        total += 1
        out = pipe.execute(img)  
        if out is None:
            filtered += 1
        else:
            kept += 1

    pct = (filtered / total * 100) if total else 0.0
    return {"total": total, "kept": kept, "filtered": filtered, "pct_filtered": pct, "read_failed": failed}

def find_dropped(pipe: Pipeline, input_dir: str, max_scan: int = 50000, max_keep: int = 10, log_every: int = 500):
    root = Path(input_dir)
    paths = collect_images(root, max_files=max_scan)

    dropped = []
    scanned = 0

    for p in paths:
        scanned += 1
        if log_every and scanned % log_every == 0:
            print(f"scanned {scanned}/{len(paths)} | dropped {len(dropped)}")

        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        try:
            tmp = img
            for step in pipe.steps:
                tmp = step.process(tmp)
        except SkipImage as e:
            dropped.append((p, str(e)))
            if len(dropped) >= max_keep:
                break

    print(f"DONE | scanned={scanned} | dropped_found={len(dropped)}")
    return dropped

def run_folder(pipe: Pipeline, input_dir: str, output_dir: str, keep_structure: bool = True, max_files: int | None = None, log_every: int = 500):
    """
    Reads images from an input folder, processes them through the pipeline, and saves them.

    Args:
        pipe (Pipeline): The processing pipeline to execute on each image.
        input_dir (str): Path to the source directory containing images.
        output_dir (str): Path to the destination directory for processed images.
        keep_structure (bool): If True, maintains the original subdirectory structure.
        max_files (int | None): Optional limit on the number of files to process.
        log_every (int): Frequency of logging progress to the console.

    Returns:
        None
    """
    in_root = Path(input_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    paths = collect_images(in_root, max_files=max_files)

    total = saved = filtered = failed = 0
    for idx, p in enumerate(paths, start=1):
        if log_every and idx % log_every == 0:
            print(f"scanned {idx}/{len(paths)} | saved {saved} | filtered {filtered} | failed {failed}")

        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            failed += 1
            continue

        total += 1
        out = pipe.execute(img)
        if out is None:
            filtered += 1
            continue

        out_path = (out_root / p.relative_to(in_root)) if keep_structure else (out_root / p.name)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        out = np.ascontiguousarray(out)
        if out.dtype != np.uint8:
            out = np.clip(out, 0, 255).astype(np.uint8)

        ok = cv2.imwrite(str(out_path), out)
        saved += int(ok)
        failed += int(not ok)

    print(f"Total read: {total} | Saved: {saved} | Filtered: {filtered} | Failed: {failed}")
