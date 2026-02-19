import os
import shutil
import sys

def organize_dataset_recursive(source_root, label_file_path, output_path):
    """
    1. Scans source_root recursively to find ALL image files.
    2. Maps filenames to their real paths.
    3. Copies files based on label.lst (Handles duplicates/multi-face images).
    """
    
    # --- Step 1: Build the Index ---
    print(f"Scanning '{source_root}' for images...")
    file_index = {} 
    
    for root, dirs, files in os.walk(source_root):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                file_index[file] = os.path.join(root, file)
                
    print(f"Indexing complete. Found {len(file_index)} images in source directory.")
    
    # --- Step 2: Define Labels ---
    label_map = {
        0: "anger",
        1: "disgust",
        2: "fear",
        3: "happiness",
        4: "sadness",
        5: "surprise"
        # 6: "neutral" is skipped
    }

    # Ensure output directories exist
    for label_name in label_map.values():
        dir_path = os.path.join(output_path, label_name)
        os.makedirs(dir_path, exist_ok=True)

    if not os.path.exists(label_file_path):
        print(f"Error: Label file not found at {label_file_path}")
        return

    # --- Step 3: Process the List ---
    print(f"Reading {label_file_path}...")
    with open(label_file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    moved_count = 0
    missing_count = 0
    skipped_neutral_count = 0

    print("------------------------------------------------")
    
    # Use a set to track processed (filename, label) pairs to avoid redundant copies
    processed_pairs = set()

    for line in lines:
        parts = line.split()
        if len(parts) < 2: continue

        filename = parts[0]
        
        try:
            label_idx = int(float(parts[-1]))
        except ValueError:
            continue

        if label_idx == 6:
            skipped_neutral_count += 1
            continue

        if label_idx in label_map:
            # Check if valid image exists in our index
            if filename in file_index:
                src = file_index[filename]
                target_folder = label_map[label_idx]
                dst = os.path.join(output_path, target_folder, filename)

                # Avoid re-copying if we already did this exact pair
                if (filename, label_idx) in processed_pairs:
                    continue
                
                try:
                    # CHANGED: Use copy instead of move to handle multi-face images
                    shutil.copy(src, dst)
                    processed_pairs.add((filename, label_idx))
                    moved_count += 1
                except Exception as e:
                    print(f"Error copying {filename}: {e}")

            else:
                missing_count += 1
                if missing_count <= 5:
                    print(f"MISSING: {filename} (Not found in scan)")

    print("------------------------------------------------")
    print(f"Processing Complete.")
    print(f"Files Copied:     {moved_count}")
    print(f"Files Missing:    {missing_count}")
    print(f"Neutrals Skipped: {skipped_neutral_count}")

# --- Configuration ---
SOURCE_ROOT = r"/home/d/dumanskyy/work/EmotionClassifier/temp_file_new_datasets/ExpW/origin" 
LABEL_FILE = r"/home/d/dumanskyy/work/EmotionClassifier/temp_file_new_datasets/label.lst" 
OUTPUT_DIR = r"/home/d/dumanskyy/work/EmotionClassifier/temp_file_new_datasets/ExpWFormated"

if __name__ == "__main__":
    organize_dataset_recursive(SOURCE_ROOT, LABEL_FILE, OUTPUT_DIR)
