import os
import shutil
import csv

# ================= CONFIGURATION =================
# Standard FERPlus/ExpW mapping
EMOTION_MAP = {
    0: None,          # Neutral -> Skip
    1: "Happiness",   # Keep
    2: "Surprise",    # Keep
    3: "Sadness",     # Keep
    4: "Anger",       # Keep
    5: "Disgust",     # Keep
    6: "Fear",        # Keep
    7: None,          # Contempt -> Skip
    8: None,          # Unknown -> Skip
    9: None           # NF -> Skip
}
# =================================================

def organize_dataset(image_source_dir, csv_path, output_base_dir):
    valid_emotions = [name for name in EMOTION_MAP.values() if name is not None]
    
    print(f"Creating folders for: {valid_emotions}...")
    for emotion in valid_emotions:
        os.makedirs(os.path.join(output_base_dir, emotion), exist_ok=True)

    print(f"Reading annotations from: {csv_path}")
    
    count_moved = 0
    count_skipped = 0
    count_missing = 0
    errors_printed = 0

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        
        for row in reader:
            if len(row) < 12:
                continue

            # 1. Clean the filename: remove whitespace and quotes
            original_filename = os.path.basename(row[0].strip().replace('"', '').replace("'", ""))
            
            # Parse votes
            try:
                votes = [int(x) for x in row[2:12]]
            except ValueError:
                continue 

            # Find Winner
            max_votes = -1
            max_index = -1
            for idx, vote_count in enumerate(votes):
                if vote_count > max_votes:
                    max_votes = vote_count
                    max_index = idx
            
            target_folder = EMOTION_MAP.get(max_index)

            if target_folder:
                # 2. PATH FINDING LOGIC
                # First, try exact match (likely .png)
                src_path = os.path.join(image_source_dir, original_filename)
                
                # If not found, try swapping extension
                if not os.path.exists(src_path):
                    name_no_ext = os.path.splitext(original_filename)[0]
                    # Try JPG if PNG failed
                    if original_filename.lower().endswith('.png'):
                        src_path = os.path.join(image_source_dir, name_no_ext + ".jpg")
                    # Try PNG if JPG failed
                    elif original_filename.lower().endswith('.jpg'):
                        src_path = os.path.join(image_source_dir, name_no_ext + ".png")

                # 3. Copy if found
                if os.path.exists(src_path):
                    # We always save with the ORIGINAL extension found on disk
                    final_filename = os.path.basename(src_path)
                    dst_path = os.path.join(output_base_dir, target_folder, final_filename)
                    shutil.copy2(src_path, dst_path)
                    count_moved += 1
                else:
                    count_missing += 1
                    # Debug: Print first 3 missing files to verify they are truly absent
                    if errors_printed < 3:
                        print(f"MISSING: {original_filename} (Looked in: {image_source_dir})")
                        errors_printed += 1
            else:
                count_skipped += 1

    print("-" * 30)
    print("Processing Complete.")
    print(f"Images Organized: {count_moved}")
    print(f"Images Skipped (Neutral/NF/etc): {count_skipped}")
    print(f"Images Missing (in CSV but not in folder): {count_missing}")
    print(f"Output location: {output_base_dir}")

if __name__ == "__main__":
    SOURCE_IMAGES = "/home/d/dumanskyy/work/EmotionClassifier/temp_file_new_datasets/ExpwCleaned/train" 
    CSV_FILE = "/home/d/dumanskyy/work/EmotionClassifier/label_expw.csv"
    OUTPUT_FOLDER = "/home/d/dumanskyy/work/EmotionClassifier/temp_file_new_datasets/ExpwCleanedFormated"

    if os.path.exists(SOURCE_IMAGES) and os.path.exists(CSV_FILE):
        organize_dataset(SOURCE_IMAGES, CSV_FILE, OUTPUT_FOLDER)
    else:
        print("Error: Please check your SOURCE_IMAGES and CSV_FILE paths in the script.")
