import os
import shutil

# ==========================================
# CONFIGURATION
# ==========================================
# 1. Path to your JAFFE images
SOURCE_DIR = "/home/d/dumanskyy/work/EmotionClassifier/temp_file_new_datasets/jaffe"

# 2. Where to create the organized folders
OUTPUT_DIR = "/home/d/dumanskyy/work/EmotionClassifier/temp_file_new_datasets/jaffeFormated"

# 3. Code Mapping based on indices 4-5 (Python index 3:5)
# Codes: NE, HA, SA, SU, AN, DI, FE
EMOTION_CODES = {
    'NE': None,        # Neutral -> SKIP
    'HA': "Happiness", # KEEP
    'SA': "Sadness",   # KEEP
    'SU': "Surprise",  # KEEP
    'AN': "Anger",     # KEEP
    'DI': "Disgust",   # KEEP
    'FE': "Fear"       # KEEP
}
# ==========================================

def organize_jaffe_labels():
    # Create target directories
    print(f"--- Setting up folders in {OUTPUT_DIR} ---")
    valid_emotions = set(val for val in EMOTION_CODES.values() if val is not None)
    
    for emotion in valid_emotions:
        os.makedirs(os.path.join(OUTPUT_DIR, emotion), exist_ok=True)

    print(f"Reading images from: {SOURCE_DIR}")
    
    count_moved = 0
    count_skipped = 0
    count_errors = 0

    # List all files
    try:
        files = os.listdir(SOURCE_DIR)
    except FileNotFoundError:
        print(f"Error: Source directory '{SOURCE_DIR}' not found.")
        return

    for filename in files:
        # JAFFE Standard Format: XX.YY.N.tiff (e.g., KA.AN.1.tiff or KA.AN1.15.tiff)
        # We need the 4th and 5th characters (indices 3 and 4)
        # Example: K A . A N ...
        # Index:   0 1 2 3 4
        
        if len(filename) < 5:
            continue

        # Extract the emotion code (Letters 4 & 5)
        # We use .upper() just in case filenames are inconsistent
        code = filename[3:5].upper()

        # Check mapping
        target_folder = EMOTION_CODES.get(code)

        if target_folder:
            src_path = os.path.join(SOURCE_DIR, filename)
            dst_path = os.path.join(OUTPUT_DIR, target_folder, filename)
            
            shutil.copy2(src_path, dst_path)
            count_moved += 1
        else:
            # Code is NE (Neutral) or unknown/invalid
            count_skipped += 1

    print("-" * 30)
    print("Processing Complete.")
    print(f"Images Organized: {count_moved}")
    print(f"Images Skipped (Neutral/Other): {count_skipped}")
    print(f"Output Location: {OUTPUT_DIR}")

if __name__ == "__main__":
    organize_jaffe_labels()
