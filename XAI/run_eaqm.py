import os
import sys
from pathlib import Path
import random
from typing import Tuple, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from captum.attr import IntegratedGradients
import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.aleks.aleks_resnet18_se import ResNet18SE

MODEL_PATH = r"C:\Users\drnes\OneDrive\Desktop\PC\Aleks\Aleks Uni\Computer Vision\Final_Project\EmotionClassifier\EmotionClassifier-aleks\models\aleks\weights\best_resnet18_se_SGD_cbfl.pth"
DATASET_ROOT = r"C:\Users\drnes\OneDrive\Desktop\PC\Aleks\Aleks Uni\Computer Vision\Final_Project\EmotionClassifier\EmotionClassifier-aleks\ready_to_use_datasets"
OUTPUT_DIR = "xai_fer_results"

CLASSES = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
IMAGE_SIZE = 64
SEED = 42

IG_STEPS = 300
CANDIDATES_PER_CLASS = 300
FINAL_PER_CLASS = 2
CONFIDENCE_THRESHOLD = 0.80

class FERAttributionAnalyzer:
    
    EMOTION_FACIAL_REGIONS = {
        'anger': {'eyebrows': 0.35, 'eyes': 0.25, 'mouth': 0.30, 'nose': 0.10},
        'disgust': {'nose': 0.40, 'mouth': 0.35, 'eyebrows': 0.15, 'eyes': 0.10},
        'fear': {'eyes': 0.40, 'eyebrows': 0.30, 'mouth': 0.20, 'nose': 0.10},
        'happiness': {'mouth': 0.45, 'eyes': 0.35, 'eyebrows': 0.10, 'nose': 0.10},
        'sadness': {'mouth': 0.35, 'eyebrows': 0.30, 'eyes': 0.25, 'nose': 0.10},
        'surprise': {'eyes': 0.35, 'eyebrows': 0.30, 'mouth': 0.30, 'nose': 0.05}
    }
    
    @staticmethod
    def get_facial_regions(image_size: int = 64) -> Dict[str, Tuple[int, int, int, int]]:

        h, w = image_size, image_size
        return {
            'eyebrows': (int(0.20*h), int(0.35*h), int(0.15*w), int(0.85*w)),
            'eyes': (int(0.30*h), int(0.45*h), int(0.15*w), int(0.85*w)),
            'nose': (int(0.40*h), int(0.60*h), int(0.35*w), int(0.65*w)),
            'mouth': (int(0.60*h), int(0.80*h), int(0.25*w), int(0.75*w))
        }
    
    @staticmethod
    def compute_region_importance(attr_map: np.ndarray, regions: Dict) -> Dict[str, float]:

        region_scores = {}
        total = attr_map.sum()
        
        if total == 0:
            return {k: 0.0 for k in regions.keys()}
        
        for region_name, (y1, y2, x1, x2) in regions.items():
            region_sum = attr_map[y1:y2, x1:x2].sum()
            region_scores[region_name] = float(region_sum / total)
        
        return region_scores
    
    @staticmethod
    def compute_quality_score(attr_map: np.ndarray, emotion: str, regions: Dict) -> float:

        region_importance = FERAttributionAnalyzer.compute_region_importance(attr_map, regions)
        expected_regions = FERAttributionAnalyzer.EMOTION_FACIAL_REGIONS[emotion]
        
        alignment_score = 0.0
        for region, expected_weight in expected_regions.items():
            actual_weight = region_importance.get(region, 0.0)
            alignment_score += min(actual_weight, expected_weight)
        alignment_score *= 100
        
        attr_norm = attr_map / (attr_map.max() + 1e-8)
        significant_pixels = (attr_norm > 0.3).sum()
        total_pixels = attr_norm.size
        sparsity_score = (1 - significant_pixels / total_pixels) * 100
        
        quality = 0.7 * alignment_score + 0.3 * sparsity_score
        return quality


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(device: torch.device) -> nn.Module:
    model = ResNet18SE(num_classes=len(CLASSES))
    in_f = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, len(CLASSES)))
    
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f" Model loaded")
    return model


def get_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def collect_candidate_images() -> List[Dict]:
    candidates = []
    
    for dataset in os.listdir(DATASET_ROOT):
        dataset_path = os.path.join(DATASET_ROOT, dataset)
        if not os.path.isdir(dataset_path):
            continue
        
        for split in ['eval', 'test']:
            split_path = os.path.join(dataset_path, split)
            if not os.path.isdir(split_path):
                continue
            
            for emotion in CLASSES:
                emotion_path = os.path.join(split_path, emotion)
                if not os.path.isdir(emotion_path):
                    continue
                
                image_files = [f for f in os.listdir(emotion_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                
                n_samples = min(len(image_files), CANDIDATES_PER_CLASS)
                sampled = random.sample(image_files, n_samples)
                
                for img_file in sampled:
                    candidates.append({
                        'path': os.path.join(emotion_path, img_file),
                        'emotion': emotion,
                        'filename': img_file,
                        'dataset': dataset,
                        'split': split
                    })
    
    return candidates


def is_valid_image(img: Image.Image) -> bool:
    arr = np.array(img.convert('L'), dtype=np.float32) / 255.0
    std, mean = arr.std(), arr.mean()
    return 0.05 < std < 0.40 and 0.15 < mean < 0.85


def compute_integrated_gradients(model: nn.Module, input_tensor: torch.Tensor, target_class: int, device: torch.device) -> np.ndarray:

    ig = IntegratedGradients(model)
    
    baselines = [
        torch.zeros_like(input_tensor),  
        torch.zeros_like(input_tensor),  
    ]
    
    blurred = input_tensor.clone()
    for c in range(3):
        img_np = input_tensor[0, c].cpu().numpy()
        img_np = cv2.GaussianBlur(img_np, (15, 15), 5.0)
        blurred[0, c] = torch.from_numpy(img_np)
    baselines.append(blurred)
    
    attributions = []
    for baseline in baselines:
        attr = ig.attribute(
            input_tensor,
            baselines=baseline.to(device),
            target=target_class,
            n_steps=IG_STEPS,
            internal_batch_size=50
        )
        attributions.append(attr)
    
    avg_attr = torch.stack(attributions).mean(dim=0)
    attr_magnitude = avg_attr.abs().mean(dim=1).squeeze().cpu().numpy()
    
    return attr_magnitude


def postprocess_attribution(attr_map: np.ndarray) -> np.ndarray:

    attr_norm = attr_map / (attr_map.max() + 1e-8)
    
    attr_uint8 = (attr_norm * 255).astype(np.uint8)
    attr_smooth = cv2.bilateralFilter(attr_uint8, d=5, sigmaColor=75, sigmaSpace=75)
    attr_smooth = attr_smooth.astype(np.float32) / 255.0
    
    h, w = attr_smooth.shape
    grid_size = 8
    cell_h, cell_w = h // grid_size, w // grid_size
    
    attr_clean = np.zeros_like(attr_smooth)
    for i in range(grid_size):
        for j in range(grid_size):
            y1, y2 = i * cell_h, (i + 1) * cell_h if i < grid_size - 1 else h
            x1, x2 = j * cell_w, (j + 1) * cell_w if j < grid_size - 1 else w
            cell_mean = attr_smooth[y1:y2, x1:x2].mean()
            attr_clean[y1:y2, x1:x2] = cell_mean
    
    threshold = np.percentile(attr_clean, 95)
    attr_clean[attr_clean < threshold] = 0.0
    
    if attr_clean.max() > 0:
        attr_clean = attr_clean / attr_clean.max()
        binary = (attr_clean > 0.1).astype(np.uint8)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        attr_clean = attr_clean * binary
    
    return attr_clean


def create_visualization(original_img: np.ndarray, attr_map: np.ndarray, emotion: str, confidence: float, quality: float, region_scores: Dict[str, float], filename: str) -> str:

    fig = plt.figure(figsize=(16, 5), dpi=150)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.8], wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(original_img)
    ax1.set_title(f'Original\n{emotion.capitalize()}', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[1])
    im = ax2.imshow(attr_map, cmap='jet', vmin=0, vmax=1)
    ax2.set_title(f'Attribution\nConf: {confidence:.1%}', fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    ax3 = fig.add_subplot(gs[2])
    ax3.imshow(original_img)
    ax3.imshow(attr_map, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    ax3.set_title(f'Overlay\nQuality: {quality:.1f}', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[3])
    regions = list(region_scores.keys())
    scores = [region_scores[r] * 100 for r in regions]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    bars = ax4.barh(regions, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('Attribution %', fontsize=10, fontweight='bold')
    ax4.set_title('Region\nImportance', fontsize=12, fontweight='bold')
    ax4.set_xlim(0, max(scores) * 1.2 if scores else 100)
    
    for bar, score in zip(bars, scores):
        width = bar.get_width()
        ax4.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{score:.1f}%', ha='left', va='center', fontsize=9)
    
    ax4.grid(axis='x', alpha=0.3, linestyle='--')
    
    fig.suptitle('Facial Emotion Recognition - XAI Analysis', 
                fontsize=14, fontweight='bold', y=0.98)
    
    safe_filename = filename.replace(' ', '_').replace('/', '_').replace('\\', '_')
    save_path = os.path.join(OUTPUT_DIR, f'{emotion}_{safe_filename}')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return save_path


def analyze_image(model: nn.Module, image_dict: Dict, transform: transforms.Compose, device: torch.device, regions: Dict) -> Dict:

    try:
        img = Image.open(image_dict['path']).convert('RGB')
    except:
        return None
    
    if not is_valid_image(img):
        return None
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()
    
    true_idx = CLASSES.index(image_dict['emotion'])
    if pred_idx != true_idx or confidence < CONFIDENCE_THRESHOLD:
        return None
    
    img_tensor.requires_grad = True
    attr_map = compute_integrated_gradients(model, img_tensor, true_idx, device)
    attr_clean = postprocess_attribution(attr_map)
    
    quality = FERAttributionAnalyzer.compute_quality_score(attr_clean, image_dict['emotion'], regions)
    region_scores = FERAttributionAnalyzer.compute_region_importance(attr_clean, regions)
    
    return {
        **image_dict,
        'attr_map': attr_clean,
        'confidence': confidence,
        'quality': quality,
        'region_scores': region_scores
    }


def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"FER XAI System - Simplified Version")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    model = load_model(device)
    transform = get_transform()
    regions = FERAttributionAnalyzer.get_facial_regions(IMAGE_SIZE)
    
    print(f"\n Collecting candidates...")
    candidates = collect_candidate_images()
    print(f"   Found {len(candidates)} images")
    
    print(f"\n Analyzing...")
    analyzed = {emotion: [] for emotion in CLASSES}
    
    for i, candidate in enumerate(candidates):
        if (i + 1) % 50 == 0:
            print(f"   Progress: {i+1}/{len(candidates)}")
        
        result = analyze_image(model, candidate, transform, device, regions)
        if result:
            analyzed[result['emotion']].append(result)
        
        if torch.cuda.is_available() and (i + 1) % 100 == 0:
            torch.cuda.empty_cache()
    
    print(f"\n Results:")
    for emotion in CLASSES:
        print(f" {emotion.capitalize():12s}: {len(analyzed[emotion]):3d} valid")
    
    print(f"\n Selecting best...")
    final_selection = []
    
    for emotion in CLASSES:
        samples = analyzed[emotion]
        if not samples:
            print(f" Warning: No samples for {emotion}")
            continue
        
        samples_sorted = sorted(samples, key=lambda x: x['quality'], reverse=True)
        n_select = min(FINAL_PER_CLASS, len(samples_sorted))
        selected = samples_sorted[:n_select]
        final_selection.extend(selected)
        
        print(f"   {emotion.capitalize():12s}: {n_select} selected (q: {selected[0]['quality']:.1f})")
    
    print(f"\n Generating visualizations...")
    
    for i, sample in enumerate(final_selection, 1):
        img = Image.open(sample['path']).convert('RGB')
        img_np = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32) / 255.0
        
        create_visualization(
            img_np, sample['attr_map'], sample['emotion'],
            sample['confidence'], sample['quality'],
            sample['region_scores'], sample['filename']
        )
        
        print(f" [{i}/{len(final_selection)}] {sample['emotion']:10s} | " f"conf: {sample['confidence']:.1%} | quality: {sample['quality']:5.1f}")
    
    print(f"\n{'='*60}")
    print(f" Complete! {len(final_selection)} visualizations")
    print(f" Saved to: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
