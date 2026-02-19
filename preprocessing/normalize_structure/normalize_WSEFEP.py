import os
import shutil
import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================
# 1. Path to your Excel file (.xlsx or .xls)
EXCEL_FILE = r"/home/d/dumanskyy/work/EmotionClassifier/temp_file_new_datasets/WSEFEP_v101_hq/WSEFEP - norms & FACS.xlsx"

# 2. Path to the folder containing the images (can have subfolders)
SOURCE_DIR = "/home/d/dumanskyy/work/EmotionClassifier/temp_file_new_datasets/WSEFEP_v101_hq"

# 3. Where to create the organized dataset
OUTPUT_DIR = "/home/d/dumanskyy/work/EmotionClassifier/temp_file_new_datasets/WSEFEPFormated"

# 4. Mapping Excel 'Display' labels (Col D) to Folder Names
#    Based on your image: joy, anger, disgust, fear, sadness, surprise, neutral
EMOTION_MAP = {
    'joy': "Happiness",       # Maps "joy" -> "Happiness"
    'anger': "Anger",         # Keeps "Anger"
    'disgust': "Disgust",     # Keeps "Disgust"
    'fear': "Fear",           # Keeps "Fear"
    'sadness': "Sadness",     # Keeps "Sadness"
    'surprise': "Surprise",   # Keeps "Surprise"
    'neutral': None           # Neutral -> SKIP
}
# ==========================================

def index_source_files(source_dir):
    """
    Creates a dictionary {filename: full_path} to find images quickly
    without re-scanning directories for every Excel row.
    """
    print(f"Scanning source directory: {source_dir}...")
    file_index = {}
    for root, _, files in os.walk(source_dir):
        for file in files:
            # We store filenames in lowercase to avoid case-sensitivity issues
            file_index[file.lower()] = os.path.join(root, file)
    print(f"Found {len(file_index)} files.")
    return file_index

def organize_from_excel():
    # 1. Create target directories
    print(f"--- Setting up folders in {OUTPUT_DIR} ---")
    valid_folders = set(val for val in EMOTION_MAP.values() if val is not None)
    for folder in valid_folders:
        os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)

    # 2. Load Excel File
    print(f"--- Reading Excel: {EXCEL_FILE} ---")
    try:
        # Assuming header is on the first row (header=0)
        df = pd.read_excel(EXCEL_FILE, header=0)
    except FileNotFoundError:
        print(f"Error: Excel file not found at {EXCEL_FILE}")
        return
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # 3. Index the source files for fast lookup
    file_map = index_source_files(SOURCE_DIR)
    
    count_moved = 0
    count_skipped = 0
    count_missing = 0

    # 4. Process Rows
    # We expect columns: 'Picture ID' (Col B) and 'Display' (Col D)
    # Pandas usually reads them by name automatically.
    
    # Verify column names exist (strip whitespace just in case)
    df.columns = [c.strip() for c in df.columns]
    
    if 'Picture ID' not in df.columns or 'Display' not in df.columns:
        print("Error: Could not find 'Picture ID' or 'Display' columns in Excel.")
        print(f"Detected columns: {df.columns.tolist()}")
        return

    for index, row in df.iterrows():
        filename = str(row['Picture ID']).strip()
        emotion_label = str(row['Display']).strip().lower() # convert to lowercase to match our map

        # Get target folder
        target_folder = EMOTION_MAP.get(emotion_label)

        if target_folder:
            # Look for the file in our index
            # We search using lowercase to match the index we built
            src_path = file_map.get(filename.lower())

            if src_path:
                dst_path = os.path.join(OUTPUT_DIR, target_folder, filename)
                
                # Copy file
                shutil.copy2(src_path, dst_path)
                count_moved += 1
            else:
                # File listed in Excel but not found on disk
                count_missing += 1
        else:
            # Emotion is Neutral or not in our map
            count_skipped += 1

    print("-" * 30)
    print("Processing Complete.")
    print(f"Images Organized: {count_moved}")
    print(f"Images Skipped (Neutral): {count_skipped}")
    print(f"Images Missing (In Excel but not found): {count_missing}")
    print(f"Output Location: {OUTPUT_DIR}")

if __name__ == "__main__":
    organize_from_excel()
