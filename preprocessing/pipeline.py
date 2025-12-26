from pathlib import Path
import cv2
import numpy as np
from skip_image import SkipImage

class Pipeline:
    def __init__(self, steps):
        self.steps = steps

    def execute(self, img):
        for step in self.steps:
            img = step.process(img) 
        return img

    def execute_batch(self, images: np.ndarray, labels: np.ndarray | None = None, log_every: int = 0):
        kept_imgs = []
        kept_labels = [] if labels is not None else None

        for i in range(len(images)):
            if log_every and i % log_every == 0:
                print(f"Processed {i}/{len(images)} | kept {len(kept_imgs)}")

            try:
                out = self.execute(images[i])
            except SkipImage:
                continue

            kept_imgs.append(out)
            if labels is not None:
                kept_labels.append(labels[i])

        kept_imgs = np.array(kept_imgs)
        if labels is None:
            return kept_imgs
        return kept_imgs, np.array(kept_labels)

    def run_folder(
        self,
        input_dir: str,
        output_dir: str,
        exts=(".png", ".jpg", ".jpeg", ".bmp", ".webp"),
        keep_structure: bool = True,
        log_every: int = 500
    ):
        in_root = Path(input_dir)
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        exts_set = {e.lower() for e in exts}
        paths = []
        for p in in_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts_set:
                paths.append(p)

        total = saved = filtered = failed = 0

        for idx, p in enumerate(paths, start=1):
            if log_every and idx % log_every == 0:
                print(f"Scanned {idx}/{len(paths)} | saved {saved} | filtered {filtered} | failed {failed}")

            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                failed += 1
                continue

            total += 1
            try:
                out = self.execute(img)
            except SkipImage:
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

    def find_dropped_in_folder(
        self,
        input_dir: str,
        max_scan: int = 20000,
        max_keep: int = 10,
        exts=(".png", ".jpg", ".jpeg", ".bmp", ".webp"),
        log_every: int = 500
    ):
        in_root = Path(input_dir)
        exts_set = {e.lower() for e in exts}

        paths = []
        for p in in_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts_set:
                paths.append(p)
                if len(paths) >= max_scan:
                    break

        dropped = []
        scanned = 0

        for p in paths:
            scanned += 1
            if log_every and scanned % log_every == 0:
                print(f"Scanned {scanned}/{len(paths)} | dropped {len(dropped)}")

            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            try:
                self.execute(img)
            except SkipImage as e:
                dropped.append((p, str(e)))
                if len(dropped) >= max_keep:
                    break

        print(f"DONE | scanned {scanned} | dropped found {len(dropped)}")
        return dropped
