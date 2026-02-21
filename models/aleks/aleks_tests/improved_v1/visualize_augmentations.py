
import os
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np
from tqdm import tqdm

# --- Configuration ---
# Input: Path to a folder containing images (e.g., 'train/happiness')
INPUT_FOLDER = "/home/d/dumanskyy/work/EmotionClassifier/latest_3_0_ready_to_use_datasets/MMAFEDB/train/happiness" 

# Output: Where to save the augmented images
OUTPUT_FOLDER = "/home/d/dumanskyy/work/EmotionClassifier/models/aleks/improved_v1/augmented_images"

# Number of augmented versions to generate per input image
# (Not strictly used for the 20-image grid, but good to keep)
AUGMENTATIONS_PER_IMAGE = 40

img_size = 64

# --- Pipelines ---

# 1. Selected Pipeline (Improved V1)
# Benefits: Geometrically robust (Affine), Label smoothing (MixUp), Preserves features.
selected_pipeline = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03)
    ], p=1.0), # Force color jitter for visualization visibility
    transforms.RandomAffine(degrees=10, translate=(0.06, 0.06), scale=(0.95, 1.05), shear=3),
])

# 2. Rejected Pipeline (Destructive)
# Issues: Occlusion destroys eyes/mouth (Erasing), CutMix creates unrealistic faces.
rejected_prep = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(), # Need tensor for Erasing
    transforms.RandomErasing(p=1.0, scale=(0.1, 0.25), ratio=(0.3, 3.3), value=0), # Aggressive erasing
])

# 3. RandAugment (Extreme/Chaotic)
try:
    # Try importing modern RandAugment
    rand_aug_pipeline = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandAugment(num_ops=3, magnitude=20), # High magnitude to make it "visible"
    ])
except:
    # Fallback if RandAugment not found in this torch version
    rand_aug_pipeline = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomSolarize(threshold=128, p=1.0),
        transforms.RandomPosterize(bits=2, p=1.0),
        transforms.GaussianBlur(kernel_size=5),
    ])


# --- Helper Functions ---


def apply_mixup(img1, img2, alpha=0.15):
    """Blends two PIL images using MixUp logic."""
    # Convert to float arrays
    arr1 = np.array(img1, dtype=np.float32)
    arr2 = np.array(img2, dtype=np.float32)
    
    # Sample lambda from Beta distribution
    lam = np.random.beta(alpha, alpha)
    
    # FOR VISUALIZATION ONLY:
    # Ensure the original image (img1) is the dominant one.
    # Otherwise, the "Augmented" image looks like a totally different person (img2).
    lam = max(lam, 1.0 - lam) 
    
    # Mix
    mixed_arr = lam * arr1 + (1 - lam) * arr2
    
    # Convert back to uint8 image
    return Image.fromarray(mixed_arr.astype(np.uint8))

def apply_cutmix(img1, img2):
    """Pastes a patch from img2 onto img1 (CutMix logic)."""
    # Simple CutMix implementation for visualization
    w, h = img1.size
    
    # Random box
    lam = np.random.beta(1.0, 1.0)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    
    img1_copy = img1.copy()
    patch = img2.crop((bbx1, bby1, bbx2, bby2))
    img1_copy.paste(patch, (bbx1, bby1))
    return img1_copy


def make_grid(images, rows, cols, cell_size=64, padding=5, bg_color="#e1e8ed"):
    """Stitches PIL images into a grid with specific background color."""
    w = cols * cell_size + (cols + 1) * padding
    h = rows * cell_size + (rows + 1) * padding
    grid = Image.new('RGB', (w, h), color=bg_color)
    
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        x = c * (cell_size + padding) + padding
        y = r * (cell_size + padding) + padding
        grid.paste(img.resize((cell_size, cell_size)), (x, y))
    
    return grid

def add_title(img, text, bg_color="#000000"):
    """Adds a bar with text above the image."""
    w, h = img.size
    font_size = 20
    bar_height = 30
    
    # Title bar background
    new_img = Image.new("RGB", (w, h + bar_height), bg_color)
    new_img.paste(img, (0, bar_height))
    
    draw = ImageDraw.Draw(new_img)
    # Default font
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 5), text, fill=(255, 255, 255), font=font)
    return new_img

def process_augmentations():
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder not found: {INPUT_FOLDER}")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Get all images
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [f for f in os.listdir(INPUT_FOLDER) if os.path.splitext(f)[1].lower() in valid_exts]
    
    if len(image_files) < 10:
        print("Not enough images for distinct selection! Need at least 10.")
        return
        
    # We only need to generate ONE master grid for the presentation really, 
    # but let's generate a few variations (e.g., 5) so user can pick the best.
    NUM_GRIDS_TO_GENERATE = 10
    
    print(f"Generating {NUM_GRIDS_TO_GENERATE} variations of the Presentation Grid...")
    
    for grid_idx in range(NUM_GRIDS_TO_GENERATE):
        
        # --- 1. REJECTED BLOCK (Before & After) ---
        # Pick 5 random images
        batch_rej = random.sample(image_files, 5)
        
        orig_rej_imgs = []
        aug_rej_imgs = []
        
        for img_name in batch_rej:
            # Original
            path = os.path.join(INPUT_FOLDER, img_name)
            pil_img = Image.open(path).convert("RGB")
            orig_rej_imgs.append(pil_img)
            
            # Augmented (Bad)
            r = random.random()
            if r < 0.25: # Erasing
                erased_tensor = rejected_prep(pil_img)
                aug_rej_imgs.append(transforms.ToPILImage()(erased_tensor))
            elif r < 0.50: # CutMix
                # Need a 2nd image
                rand_img_name = random.choice(image_files)
                img2 = Image.open(os.path.join(INPUT_FOLDER, rand_img_name)).convert("RGB")
                aug_rej_imgs.append(apply_cutmix(pil_img.resize((img_size, img_size)), img2.resize((img_size, img_size))))
            elif r < 0.75: # BAD MixUp (High Alpha = 0.8)
                rand_img_name = random.choice(image_files)
                img2 = Image.open(os.path.join(INPUT_FOLDER, rand_img_name)).convert("RGB")
                ready_img1 = selected_pipeline(pil_img)
                ready_img2 = transforms.Resize((img_size, img_size))(img2)
                # Alpha 0.8 -> More mixing, more ghosting.
                aug_rej_imgs.append(apply_mixup(ready_img1, ready_img2, alpha=0.8))                
            else: # RandAug
                aug_rej_imgs.append(rand_aug_pipeline(pil_img))

        # Stitch Rejected Block (Row 1: Orig, Row 2: Aug)
        rej_images = orig_rej_imgs + aug_rej_imgs
        rej_grid = make_grid(rej_images, rows=2, cols=5, cell_size=128, bg_color="#e1e8ed")
        rej_grid = add_title(rej_grid, "REJECTED (Top: Original, Bottom: Augmented)", bg_color="#000000")


        # --- 2. ALLOWED BLOCK (Before & After) ---
        # Pick 5 NEW random images
        remaining_files = list(set(image_files) - set(batch_rej))
        if len(remaining_files) < 5: remaining_files = image_files
        batch_all = random.sample(remaining_files, 5)
        
        orig_all_imgs = []
        aug_all_imgs = []
        
        for img_name in batch_all:
             # Original
            path = os.path.join(INPUT_FOLDER, img_name)
            pil_img = Image.open(path).convert("RGB")
            orig_all_imgs.append(pil_img)
            
            # Augmented (Good)
            if random.random() < 0.5: # MixUp
                rand_img_name = random.choice(image_files)
                img2 = Image.open(os.path.join(INPUT_FOLDER, rand_img_name)).convert("RGB")
                ready_img1 = selected_pipeline(pil_img)
                ready_img2 = transforms.Resize((img_size, img_size))(img2)
                aug_all_imgs.append(apply_mixup(ready_img1, ready_img2, alpha=0.2))
            else: # Affine
                aug_all_imgs.append(selected_pipeline(pil_img))
        
        # Stitch Allowed Block (Row 1: Orig, Row 2: Aug)
        all_images = orig_all_imgs + aug_all_imgs
        all_grid = make_grid(all_images, rows=2, cols=5, cell_size=128, bg_color="#e1e8ed")
        all_grid = add_title(all_grid, "ALLOWED (Top: Original, Bottom: Augmented)", bg_color="#000000")
        
        # --- COMBINE BLOCKS ---
        final_w = max(rej_grid.width, all_grid.width)
        final_h = rej_grid.height + all_grid.height + 20 # Spacing
        
        final_img = Image.new("RGB", (final_w, final_h), "#e1e8ed")
        final_img.paste(rej_grid, (0, 0))
        final_img.paste(all_grid, (0, rej_grid.height + 20))
        
        # Save
        s_path = os.path.join(OUTPUT_FOLDER, f"presentation_grid_v{grid_idx}.jpg")
        final_img.save(s_path)
        
    print(f"Done! Saved {NUM_GRIDS_TO_GENERATE} presentation grids to {os.path.abspath(OUTPUT_FOLDER)}")


if __name__ == "__main__":
    process_augmentations()
