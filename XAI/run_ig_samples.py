import os
import sys
from pathlib import Path
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from captum.attr import IntegratedGradients, NoiseTunnel
from captum.attr import visualization as viz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.aleks.aleks_resnet18_se import ResNet18SE


MODEL_PATH = r"C:\Users\drnes\OneDrive\Desktop\PC\Aleks\Aleks Uni\Computer Vision\Final_Project\EmotionClassifier\EmotionClassifier-aleks\models\aleks\weights\best_resnet18_se_SGD_cbfl.pth"
DATASET_ROOT = r"C:\Users\drnes\OneDrive\Desktop\PC\Aleks\Aleks Uni\Computer Vision\Final_Project\EmotionClassifier\EmotionClassifier-aleks\ready_to_use_datasets"
OUTPUT_DIR = "ig_results_samples"

CLASSES = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
SPLITS = ["eval"]

IMAGE_SIZE = 64
CANDIDATES_PER_CLASS = 150
CONFIDENCE_THRESHOLD = 0.85
TOP_K = 12

IG_STEPS = 150
NT_SAMPLES = 50
NT_STDEVS = 0.15
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_candidate_images(root, max_per_class):
    samples = []
    for dataset in os.listdir(root):
        for split in SPLITS:
            split_path = os.path.join(root, dataset, split)
            if not os.path.isdir(split_path):
                continue
            for cls in CLASSES:
                cls_path = os.path.join(split_path, cls)
                if not os.path.isdir(cls_path):
                    continue
                imgs = [f for f in os.listdir(cls_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
                chosen = random.sample(imgs, min(len(imgs), max_per_class))
                for img in chosen:
                    samples.append({
                        "path": os.path.join(cls_path, img),
                        "label": cls,
                        "filename": img,
                        "split": split
                    })
    return samples


def get_preprocess():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def load_model(device):
    model = ResNet18SE(num_classes=len(CLASSES))
    in_f = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, len(CLASSES)))
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def evaluate_prediction(model, x, true_label_idx):
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()
    return pred_idx == true_label_idx, confidence, pred_idx


def compute_attribution_quality(attr):
    attr_np = attr.detach().cpu().numpy().squeeze()
    if attr_np.max() == 0:
        return 0.0
    attr_norm = attr_np / attr_np.max()
    focus = np.sum(attr_norm > np.percentile(attr_norm, 80)) / attr_norm.size
    variance = np.var(attr_norm)
    return (1 - focus) * variance * 100


def compute_ig(model, x, target):
    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)
    baseline = torch.zeros_like(x)
    attr = nt.attribute(
        x,
        baselines=baseline,
        target=target,
        n_steps=IG_STEPS,
        nt_type="smoothgrad_sq",
        nt_samples=NT_SAMPLES,
        stdevs=NT_STDEVS,
        internal_batch_size=NT_SAMPLES
    )
    return attr.abs().sum(dim=1)


def visualize_and_save(x, attr, label, fname, confidence):
    img = x.detach().squeeze().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img * 0.5) + 0.5
    img = np.clip(img, 0, 1)

    attr = attr.detach().squeeze().cpu().numpy()
    attr = np.repeat(attr[:, :, np.newaxis], 3, axis=2)

    fig, _ = viz.visualize_image_attr_multiple(
        attr,
        img,
        methods=["original_image", "blended_heat_map", "heat_map"],
        signs=["absolute_value", "positive", "positive"],
        cmap="inferno",
        show_colorbar=True,
        outlier_perc=1,
        alpha_overlay=0.7,
        titles=[
            f"Original ({label}, conf: {confidence:.1%})",
            "Integrated Gradients (Overlay)",
            "Integrated Gradients (Heatmap)"
        ],
        use_pyplot=False
    )

    safe = fname.replace(" ", "_").replace("/", "_")
    fig.savefig(os.path.join(OUTPUT_DIR, f"{label}_{safe}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    preprocess = get_preprocess()

    print("Collecting candidate images...")
    candidates = get_candidate_images(DATASET_ROOT, CANDIDATES_PER_CLASS)

    scored = []
    print("Evaluating candidates...")
    for s in candidates:
        try:
            img = Image.open(s["path"]).convert("RGB")
        except:
            continue

        x = preprocess(img).unsqueeze(0).to(device)
        true_idx = CLASSES.index(s["label"])
        correct, conf, _ = evaluate_prediction(model, x, true_idx)

        if correct and conf >= CONFIDENCE_THRESHOLD:
            x.requires_grad_(True)
            attr = compute_ig(model, x, true_idx)
            quality = compute_attribution_quality(attr)
            scored.append({**s, "confidence": conf, "quality": quality})

    scored = sorted(
        scored,
        key=lambda x: (0.3 * x["confidence"] + 0.7 * x["quality"]),
        reverse=True
    )[:TOP_K]

    print(f"\nGenerating IG for top {len(scored)} samples...\n")

    for i, s in enumerate(scored, 1):
        img = Image.open(s["path"]).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(device)
        x.requires_grad_(True)

        target = CLASSES.index(s["label"])
        attr = compute_ig(model, x, target)
        visualize_and_save(x, attr, s["label"], s["filename"], s["confidence"])

        print(f"[{i}/{len(scored)}] {s['label']} | {s['confidence']:.1%}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nDone. Results saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
