import os
import shutil

# ==========================================
# CONFIGURATION
# ==========================================
SOURCE_DIR = "/home/d/dumanskyy/work/EmotionClassifier/temp_file_new_datasets/NONAME"
OUTPUT_DIR = "/home/d/dumanskyy/work/EmotionClassifier/temp_file_new_datasets/NONAMEFormated"

# Mapping from Filename Prefix -> Target Folder
EMOTION_MAP = {
    'anger': "anger",
    'disgust': "disgust",
    'fear': "fear",
    'happy': "happiness",    
    'sad': "sadness",        
    'surprised': "surprise"  
}

SKIP_PREFIXES = ['contempt', 'neutral']
# ==========================================

def organize_by_filename():
    # 1. Create target directories
    print(f"--- Setting up folders in {OUTPUT_DIR} ---")
    target_folders = set(EMOTION_MAP.values())
    for folder in target_folders:
        path = os.path.join(OUTPUT_DIR, folder)
        os.makedirs(path, exist_ok=True)

    print(f"\n--- Processing images recursively from: {SOURCE_DIR} ---")

    count_moved = 0
    count_skipped = 0
    count_dirs_checked = 0

    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory '{SOURCE_DIR}' does not exist.")
        return

    # CHANGED: Use os.walk to go inside all subfolders (0, 1, 2, etc.)
    for root, dirs, files in os.walk(SOURCE_DIR):
        # Skip Mac system folders if present
        if "__MACOSX" in root:
            continue
            
        count_dirs_checked += 1
        
        for filename in files:
            # Filter for valid image files
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                continue

            # Standardize filename for checking
            name_base = os.path.splitext(filename)[0].lower()

            target_folder = None

            # Check matching prefixes
            for prefix, folder in EMOTION_MAP.items():
                if name_base.startswith(prefix):
                    target_folder = folder
                    break

            # Construct the full path to the CURRENT file inside its subfolder
            src_path = os.path.join(root, filename)

            if target_folder:
                # Construct destination
                dst_path = os.path.join(OUTPUT_DIR, target_folder, filename)
                
                # Check for duplicates (optional safety)
                if os.path.exists(dst_path):
                    # Rename if duplicate: happy.jpg -> happy_1.jpg
                    base, ext = os.path.splitext(filename)
                    dst_path = os.path.join(OUTPUT_DIR, target_folder, f"{base}_{count_moved}{ext}")

                shutil.copy2(src_path, dst_path)
                count_moved += 1
            else:
                is_skippable = any(name_base.startswith(prefix) for prefix in SKIP_PREFIXES)
                if is_skippable:
                    count_skipped += 1

    print("\n" + "="*30)
    print("PROCESSING COMPLETE")
    print("="*30)
    print(f"Subfolders Checked: {count_dirs_checked}")
    print(f"Images Organized: {count_moved}")
    print(f"Images Skipped: {count_skipped}")
    print(f"Output Location: {OUTPUT_DIR}")

if __name__ == "__main__":
    organize_by_filename()
