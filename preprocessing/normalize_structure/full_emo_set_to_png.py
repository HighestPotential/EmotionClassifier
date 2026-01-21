import os
import multiprocessing
from PIL import Image
from tqdm import tqdm  # Install with: pip install tqdm

# Configuration
# CHANGED: Define your actual source path here
SOURCE_ROOT = r"D:\Datasets\EmoSet\Raw_PNG"
# CHANGED: Define where you want the new JPGs to go (keeps source safe)
TARGET_ROOT = r"D:\Datasets\EmoSet\Processed_JPG"

def convert_single_file(file_info):
    """
    Worker function to convert a single image.
    """
    source_path, relative_path = file_info
    
    try:
        # Construct target path maintaining folder structure
        target_path = os.path.join(TARGET_ROOT, relative_path)
        # Change extension to .jpg
        target_path = os.path.splitext(target_path)[0] + ".jpg"
        
        # Ensure the subfolder exists
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        # Skip if already exists (resume capability)
        if os.path.exists(target_path):
            return

        # Open and Convert
        with Image.open(source_path) as img:
            # CHANGED: Convert RGBA to RGB (JPEG requirement)
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # CHANGED: Save as JPEG with Quality 85 for ~90% size reduction
            img.save(target_path, "JPEG", quality=85, optimize=True)
            
    except Exception as e:
        print(f"Error converting {source_path}: {e}")

def main():
    print(f"Scanning directory: {SOURCE_ROOT}")
    
    # 1. Collect all PNG files
    files_to_process = []
    for root, dirs, files in os.walk(SOURCE_ROOT):
        for file in files:
            if file.lower().endswith('.png'):
                full_path = os.path.join(root, file)
                # Calculate relative path to replicate structure in target
                rel_path = os.path.relpath(full_path, SOURCE_ROOT)
                files_to_process.append((full_path, rel_path))
    
    total_files = len(files_to_process)
    print(f"Found {total_files} PNG images. Starting conversion...")

    # 2. Process in parallel
    # CHANGED: Uses all available CPU cores
    num_cpus = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_cpus) as pool:
        # tqdm creates the progress bar
        list(tqdm(pool.imap_unordered(convert_single_file, files_to_process), total=total_files))
        
    print("Conversion complete.")

if __name__ == '__main__':
    # Windows requires this protection for multiprocessing
    multiprocessing.freeze_support()
    main()
