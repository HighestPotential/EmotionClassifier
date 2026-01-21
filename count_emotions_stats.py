import os
from collections import defaultdict

ALLOWED_LABELS = {
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear"
}

LABEL_ALIASES = {
    "happy": "happiness",
    "angry": "anger",
    "sad": "sadness",
    "disgusted": "disgust",
    "fearful": "fear"
}

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

def count_images_in_folder(folder_path):
    count = 0
    try:
        for filename in os.listdir(folder_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                count += 1
    except OSError:
        pass
    return count

def scan_directories(base_folders):
    # Dictionary to store results: dataset_name -> {label: count}
    dataset_stats = defaultdict(lambda: defaultdict(int))

    for base_folder in base_folders:
        abs_base_folder = os.path.abspath(base_folder)
        if not os.path.exists(abs_base_folder):
            print(f"Warning: Folder not found: {abs_base_folder}")
            continue

        print(f"Scanning base folder: {abs_base_folder}...")

        # Get list of dataset folders in this base folder
        try:
            # We assume immediate subdirectories are the datasets (e.g. datasets/AffectNet)
            dataset_dirs = [d for d in os.listdir(abs_base_folder) if os.path.isdir(os.path.join(abs_base_folder, d))]
        except OSError as e:
            print(f"Error accessing {abs_base_folder}: {e}")
            continue

        for dataset_name in dataset_dirs:
            dataset_path = os.path.join(abs_base_folder, dataset_name)
            
            # Now walk through this specific dataset folder to find emotion subfolders
            for root, dirs, files in os.walk(dataset_path):
                folder_name = os.path.basename(root).lower()
                
                # Determine canonical label
                canonical_label = None
                if folder_name in ALLOWED_LABELS:
                    canonical_label = folder_name
                elif folder_name in LABEL_ALIASES:
                    canonical_label = LABEL_ALIASES[folder_name]
                
                if canonical_label:
                    count = 0
                    for file in files:
                        if os.path.splitext(file)[1].lower() in IMAGE_EXTENSIONS:
                            count += 1
                    
                    if count > 0:
                        dataset_stats[dataset_name][canonical_label] += count

    print("\n" + "="*50)
    print("EMOTION COUNTS PER DATASET")
    print("="*50)
    
    # Sort datasets for cleaner output
    sorted_datasets = sorted(dataset_stats.keys())
    
    for dataset in sorted_datasets:
        counts = dataset_stats[dataset]
        total_images = sum(counts.values())
        print(f"\nDataset: {dataset} (Total: {total_images})")
        
        # Sort labels alphabetically or by count? Alphabetically is standard
        for label in sorted(counts.keys()):
            print(f"  - {label}: {counts[label]}")

    print("\n" + "="*50)
    print("GRAND TOTALS")
    print("="*50)
    grand_totals = defaultdict(int)
    for ds in dataset_stats:
        for lbl, cnt in dataset_stats[ds].items():
            grand_totals[lbl] += cnt
            
    total_all_images = 0
    for label, count in grand_totals.items():
        print(f"{label}: {count}")
        total_all_images += count
        
    print("-" * 50)
    print(f"TOTAL IMAGES ACROSS ALL DATASETS: {total_all_images}")
    print("=" * 50)

if __name__ == "__main__":
    # Folders requested by user
    target_folders = ["new_datasets", "datasets"]
    
    # Add new_preprocessed_dataset just in case 'new_stasets' was referring to it, 
    # but sticking to the explicit request of 'new_dataset' (new_datasets) and 'dtasets' (datasets) for now.
    
    scan_directories(target_folders)
