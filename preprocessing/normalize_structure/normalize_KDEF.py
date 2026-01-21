import os
import shutil

# ==========================================
# CONFIGURATION
# ==========================================
# 1. Path to the main KDEF folder (containing AF01, AF02, etc.)
SOURCE_ROOT_DIR = "/home/d/dumanskyy/work/EmotionClassifier/temp_file_new_datasets/KDEF"

# 2. Where to create the organized dataset
OUTPUT_DIR = "/home/d/dumanskyy/work/EmotionClassifier/temp_file_new_datasets/KDEFFormated"

# 3. KDEF Mapping (Indices 4-5)
# AF=Fear, AN=Anger, DI=Disgust, HA=Happy, NE=Neutral, SA=Sad, SU=Surprise
EMOTION_CODES = {
    'AF': "Fear",      # KEEP
    'AN': "Anger",     # KEEP
    'DI': "Disgust",   # KEEP
    'HA': "Happiness", # KEEP
    'NE': None,        # Neutral -> SKIP
    'SA': "Sadness",   # KEEP
    'SU': "Surprise"   # KEEP
}
# ==========================================

def organize_kdef():
    print(f"--- Setting up folders in {OUTPUT_DIR} ---")
    valid_emotions = set(val for val in EMOTION_CODES.values() if val is not None)
    
    for emotion in valid_emotions:
        os.makedirs(os.path.join(OUTPUT_DIR, emotion), exist_ok=True)

    print(f"Scanning for images inside: {SOURCE_ROOT_DIR}")
    
    count_moved = 0
    count_skipped = 0
    
    # os.walk allows us to look inside AF01, AF02, AF20, etc. automatically
    for root, dirs, files in os.walk(SOURCE_ROOT_DIR):
        for filename in files:
            # Filter for images (ignore .DS_Store or text files)
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue

            # Standard KDEF Filename: AF01ANFL.JPG (8 chars + extension)
            # Indices:
            # A F 0 1 [A N] F L
            # 0 1 2 3  4 5  6 7
            
            # Safety check: Filename must be long enough
            if len(filename) < 6:
                continue

            # Extract letters 5 and 6 (Index 4 and 5)
            # We use .upper() to be safe
            emotion_code = filename[4:6].upper()

            target_folder = EMOTION_CODES.get(emotion_code)

            if target_folder:
                src_path = os.path.join(root, filename)
                dst_path = os.path.join(OUTPUT_DIR, target_folder, filename)
                
                shutil.copy2(src_path, dst_path)
                count_moved += 1
            else:
                # Neutral (NE) or unknown code
                count_skipped += 1

    print("-" * 30)
    print("Processing Complete.")
    print(f"Images Organized: {count_moved}")
    print(f"Images Skipped (Neutral/Other): {count_skipped}")
    print(f"Output Location: {OUTPUT_DIR}")

if __name__ == "__main__":
    organize_kdef()
