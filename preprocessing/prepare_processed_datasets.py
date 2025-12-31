import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# Allowed labels (lowercase for case-insensitive matching)
ALLOWED_LABELS = {
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear"
}

def get_emotion_from_path(path: Path) -> str | None:
    """
    Extracts emotion label from the parent directory name.
    Returns None if the label is not in the allowed list.
    """
    # Assuming structure like .../Emotion/image.png
    # Check parent, then grandparent (in case of extra nesting)
    
    parts = path.parts
    # Iterate backwards through parts to find a valid emotion
    for part in reversed(parts[:-1]): # Exclude filename
        label = part.lower()
        if label in ALLOWED_LABELS:
            return label
            
    return None

def process_datasets(input_root_dir: str, output_root_dir: str):
    input_root = Path(input_root_dir)
    output_root = Path(output_root_dir)
    
    if not input_root.exists():
        print(f"Error: Input directory {input_root} does not exist.")
        return

    # 1. Iterate over each dataset folder in input_root
    # We assume direct children of input_root are dataset names (e.g., AffectNet, CKplus)
    for dataset_dir in input_root.iterdir():
        if not dataset_dir.is_dir():
            continue
            
        dataset_name = dataset_dir.name
        print(f"\nProcessing dataset: {dataset_name}")
        
        # 2. Collect all images and label them
        # Map: emotion -> list of source file paths
        images_by_emotion = {emotion: [] for emotion in ALLOWED_LABELS}
        
        # Gather all files recursively
        all_files = [p for p in dataset_dir.rglob("*") if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
        
        print(f"  Found {len(all_files)} potential image files.")
        
        for file_path in all_files:
            emotion = get_emotion_from_path(file_path)
            if emotion:
                images_by_emotion[emotion].append(file_path)
        
        total_valid = sum(len(imgs) for imgs in images_by_emotion.values())
        print(f"  Found {total_valid} valid images matching allowed labels.")
        
        if total_valid == 0:
            print("  Skipping dataset (no valid images found).")
            continue

        # 3. Split and Copy
        dataset_output_dir = output_root / dataset_name
        
        # Create output directories
        for split in ['train', 'eval', 'test']:
            for emotion in ALLOWED_LABELS:
                (dataset_output_dir / split / emotion).mkdir(parents=True, exist_ok=True)
                
        for emotion, files in images_by_emotion.items():
            if not files:
                continue
                
            # Random shuffle
            random.shuffle(files)
            
            n_total = len(files)
            n_train = int(n_total * 0.8)
            n_eval = int(n_total * 0.1)
            # Remaining goes to test (approx 10%)
            
            train_files = files[:n_train]
            eval_files = files[n_train:n_train+n_eval]
            test_files = files[n_train+n_eval:]
            
            print(f"    {emotion}: {n_total} files -> Train: {len(train_files)}, Eval: {len(eval_files)}, Test: {len(test_files)}")
            
            # Helper to copy files
            def copy_files(file_list, split_name):
                dest_dir = dataset_output_dir / split_name / emotion
                for src in file_list:
                    # Create a unique filename to avoid collisions if merging from multiple subfolders
                    # We utilize the original parent folder structure in filename if needed, 
                    # but simple unique naming is often safer.
                    # Let's keep original name, append a counter if collision (unlikely with this logic but good practice)
                    # Actually, simple copy is usually fine if sources are distinct files. 
                    # But if we flatten 'Product/test/anger/1.png' and 'Product/train/anger/1.png', we have collision.
                    
                    # Strategy: Use relative path parts joined by underscore
                    rel_path = src.relative_to(dataset_dir)
                    # sanitize path separators
                    new_name = str(rel_path).replace(os.sep, "_") 
                    dest = dest_dir / new_name
                    
                    shutil.copy2(src, dest)

            copy_files(train_files, 'train')
            copy_files(eval_files, 'eval')
            copy_files(test_files, 'test')

    print(f"\nProcessing complete. Output saved to: {output_root}")

if __name__ == "__main__":
    # You can configure these paths
    INPUT_DIR = r"D:\3thSemester\DLCVProject\EmotionClassifier\preprocessed_dataset"
    OUTPUT_DIR = r"D:\3thSemester\DLCVProject\EmotionClassifier\ready_to_use_datasets"
    
    process_datasets(INPUT_DIR, OUTPUT_DIR)
