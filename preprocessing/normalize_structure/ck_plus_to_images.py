import csv
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# Emotion label mapping
EMOTION_LABELS = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happiness",
    4: "sadness",
    5: "surprise",
    6: "neutral",
    7: "contempt"
}

def convert_ckplus_csv_to_images(
    csv_path: str,
    output_base_dir: str,
    img_width: int = 48,
    img_height: int = 48
):
    """
    Convert CK+ Extended CSV file to organized image folders.
    
    Args:
        csv_path: Path to the ckextended.csv file
        output_base_dir: Base directory for output (e.g., D:/path/to/datasets/CKplusImFile)
        img_width: Width of the image (default 48)
        img_height: Height of the image (default 48)
    """
    csv_path = Path(csv_path)
    output_base = Path(output_base_dir)
    
    # Create train folder
    train_dir = output_base / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    
    # Create emotion folders
    for emotion in EMOTION_LABELS.values():
        emotion_dir = train_dir / emotion
        emotion_dir.mkdir(exist_ok=True)
    
    # Counter for each emotion
    emotion_counters = {emotion: 0 for emotion in EMOTION_LABELS.values()}
    
    # Read and process CSV
    print(f"Reading CSV file: {csv_path}")
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        # Get total rows for progress bar
        rows = list(reader)
        total_rows = len(rows)
        
        print(f"Found {total_rows} images in CSV")
        print("Converting to images...")
        
        for row in tqdm(rows, desc="Processing images", unit="img"):
            try:
                # Get emotion label
                emotion_id = int(row['emotion'])
                emotion_name = EMOTION_LABELS[emotion_id]
                
                # Parse pixel values
                pixels = row['pixels'].split()
                pixels = np.array(pixels, dtype=np.uint8)
                
                # Reshape to image
                img = pixels.reshape(img_height, img_width)
                
                # Generate filename
                emotion_counters[emotion_name] += 1
                filename = f"ckplus_{emotion_name}_{emotion_counters[emotion_name]:04d}.png"
                
                # Save image
                output_path = train_dir / emotion_name / filename
                cv2.imwrite(str(output_path), img)
                
            except Exception as e:
                print(f"\nError processing row: {e}")
                continue
    
    # Print summary
    print("\n" + "="*60)
    print("Conversion Summary:")
    print("="*60)
    for emotion, count in emotion_counters.items():
        print(f"{emotion.capitalize():12s}: {count:5d} images")
    print("="*60)
    print(f"Total images: {sum(emotion_counters.values())}")
    print(f"\nImages saved to: {train_dir}")


if __name__ == "__main__":
    # Paths
    csv_file = r"D:\3thSemester\DLCVProject\EmotionClassifier\datasets\CKplus\ckextended.csv"
    output_dir = r"D:\3thSemester\DLCVProject\EmotionClassifier\datasets\CKplusIm"
    
    # Convert
    convert_ckplus_csv_to_images(csv_file, output_dir)
    
    print("\nConversion complete!")
