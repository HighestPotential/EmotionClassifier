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
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.aleks.aleks_resnet18_se import ResNet18SE


MODEL_PATH = r"C:\Users\drnes\OneDrive\Desktop\PC\Aleks\Aleks Uni\Computer Vision\Final_Project\EmotionClassifier\EmotionClassifier-aleks\models\aleks\weights\best_resnet18_se_SGD_cbfl.pth"
DATASET_ROOT = r"C:\Users\drnes\OneDrive\Desktop\PC\Aleks\Aleks Uni\Computer Vision\Final_Project\EmotionClassifier\EmotionClassifier-aleks\ready_to_use_datasets"
OUTPUT_DIR = "ig_results_all"

CLASSES = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]

IMAGE_SIZE = 64
IG_STEPS = 150
NT_SAMPLES = 50
NT_STDEVS = 0.15
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_all_images(root):
    samples = []
    for dataset in os.listdir(root):
        dataset_path = os.path.join(root, dataset)
        if not os.path.isdir(dataset_path):
            continue

        for split in ["train", "eval", "test"]:
            split_path = os.path.join(dataset_path, split)
            if not os.path.isdir(split_path):
                continue

            for cls in CLASSES:
                cls_path = os.path.join(split_path, cls)
                if not os.path.isdir(cls_path):
                    continue

                for img in os.listdir(cls_path):
                    if img.lower().endswith((".jpg", ".jpeg", ".png")):
                        samples.append({
                            "path": os.path.join(cls_path, img),
                            "label": cls,
                            "split": split,
                            "dataset": dataset,
                            "filename": img
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
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_f, len(CLASSES))
    )

    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


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


def visualize_and_save(x, attr, label, fname, confidence, split, dataset):
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

    safe_name = fname.replace(" ", "_").replace("/", "_")
    out_name = f"{dataset}_{split}_{label}_{safe_name}"
    fig.savefig(os.path.join(OUTPUT_DIR, out_name), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    preprocess = get_preprocess()

    samples = get_all_images(DATASET_ROOT)
    print(f"Total images found: {len(samples)}")

    for s in tqdm(samples, desc="Computing Integrated Gradients", unit="img"):
        try:
            img = Image.open(s["path"]).convert("RGB")
        except:
            continue

        x = preprocess(img).unsqueeze(0).to(device)
        x.requires_grad_(True)

        target = CLASSES.index(s["label"])

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            confidence = probs[0, target].item()

        attr = compute_ig(model, x, target)

        visualize_and_save(
            x,
            attr,
            s["label"],
            s["filename"],
            confidence,
            s["split"],
            s["dataset"]
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
