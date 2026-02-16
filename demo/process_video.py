"""
Offline video processor — reads a video file, detects faces, classifies
emotions with optional XAI heatmap overlays (GradCAM / Integrated Gradients),
and writes the annotated result to a new video file.

Usage examples
--------------
    python process_video.py input.mp4 --xai gradcam
    python process_video.py input.mp4 --xai ig
    python process_video.py input.mp4 --xai none
    python process_video.py input.mp4 --xai gradcam --output out.mp4
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import platform
import urllib.request
from collections import deque
import argparse
from aleks_resnet18_se import ResNet18SE

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

MODEL_CHECKPOINT_PATH_RESNET18 = os.path.join(CURRENT_DIR, "best_resnet18_se_SGD_cbfl.pth")

IMG_SIZE = 64
CLASSES = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EMOJI_MAP = {
    'anger': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'happiness': '😄',
    'sadness': '😢',
    'surprise': '😲'
}


# ---------------------------------------------------------------------------
#  XAI helpers  (same as demo.py)
# ---------------------------------------------------------------------------

class GradCAMHelper:
    """Lightweight GradCAM that hooks into a single target layer."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _module, _inp, out):
        self.activations = out.detach()

    def _bwd(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x):
        """Return (heatmap_0_1, class_idx, probs_tensor)."""
        self.model.zero_grad()
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)

        class_idx = logits.argmax(dim=1).item()
        score = logits[:, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)[0].cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam, class_idx, probs


class IntegratedGradientsHelper:
    """Thin wrapper around captum IntegratedGradients for offline use."""

    def __init__(self, model, n_steps=50):
        from captum.attr import IntegratedGradients
        self.ig = IntegratedGradients(model)
        self.model = model
        self.n_steps = n_steps

    def __call__(self, x):
        """Return (heatmap_0_1, class_idx, probs_tensor)."""
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
            class_idx = logits.argmax(dim=1).item()

        x_ig = x.clone().requires_grad_(True)
        baseline = torch.zeros_like(x_ig)
        attr = self.ig.attribute(
            x_ig,
            baselines=baseline,
            target=class_idx,
            n_steps=self.n_steps,
            internal_batch_size=self.n_steps,
        )
        cam = attr.abs().sum(dim=1)[0].detach().cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam, class_idx, probs


def overlay_heatmap(frame, cam, x1, y1, x2, y2, alpha=0.5):
    """Alpha-blend a JET heatmap onto *frame* in-place over the face ROI."""
    h = y2 - y1
    w = x2 - x1
    if h <= 0 or w <= 0:
        return frame
    cam_resized = cv2.resize(cam, (w, h))
    cam_uint8 = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    roi = frame[y1:y2, x1:x2]
    blended = np.uint8((1 - alpha) * roi + alpha * heatmap)
    frame[y1:y2, x1:x2] = blended
    return frame


# ---------------------------------------------------------------------------
#  Video processor
# ---------------------------------------------------------------------------

class VideoProcessor:
    def __init__(self, checkpoint_path, xai_method="none"):
        self.device = DEVICE
        self.xai_method = xai_method
        print(f"Running on device: {self.device}")
        print(f"XAI method: {self.xai_method}")

        # Haar Cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        self.model = self._setup_model(checkpoint_path)

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # XAI helper
        self.xai = None
        if self.xai_method == "gradcam":
            target_layer = self.model.layer4[-1]
            self.xai = GradCAMHelper(self.model, target_layer)
        elif self.xai_method == "ig":
            self.xai = IntegratedGradientsHelper(self.model, n_steps=50)

        # Font is loaded in process() once we know the video resolution
        self.font = None
        self.text_offset = 30
        self.box_thickness = 2

    # ---- model ----

    def _setup_model(self, checkpoint_path):
        model = ResNet18SE(num_classes=len(CLASSES))

        if hasattr(model, "fc") and isinstance(model.fc, torch.nn.Linear):
            in_f = model.fc.in_features
            model.fc = torch.nn.Sequential(
                torch.nn.Dropout(p=0.3),
                torch.nn.Linear(in_f, len(CLASSES))
            )

        if not checkpoint_path:
            print("Warning: No checkpoint path. Random weights.")
        else:
            print(f"Loading weights from: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(state_dict)

        model.to(self.device)
        model.eval()
        return model

    # ---- font ----

    def _load_emoji_font(self, font_size):
        system = platform.system()
        font_path = None

        if system == "Windows":
            font_path = "C:\\Windows\\Fonts\\seguiemj.ttf"
        elif system == "Darwin":
            font_path = "/System/Library/Fonts/Apple Color Emoji.ttc"

        if font_path and os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, font_size)
            except Exception:
                pass

        local_font_path = os.path.join(CURRENT_DIR, "NotoColorEmoji.ttf")
        if not os.path.exists(local_font_path):
            url = "https://github.com/googlefonts/noto-emoji/raw/main/fonts/NotoColorEmoji.ttf"
            try:
                urllib.request.urlretrieve(url, local_font_path)
            except Exception:
                return ImageFont.load_default()

        try:
            return ImageFont.truetype(local_font_path, font_size)
        except Exception:
            return ImageFont.load_default()

    # ---- drawing ----

    def draw_complex_text(self, frame, emoji, text, x, y, text_color):
        cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)
        draw = ImageDraw.Draw(pil_im)

        try:
            draw.text((x, y), emoji, font=self.font, embedded_color=True)
        except TypeError:
            draw.text((x, y), emoji, font=self.font, fill=(255, 255, 255))

        emoji_width = self.font.getlength(emoji)
        draw.text((x + emoji_width + 5, y), text, font=self.font, fill=text_color)

        return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

    # ---- XAI ----

    def _compute_xai(self, input_tensor):
        if self.xai is None:
            return None, None, None
        return self.xai(input_tensor)

    # ---- main loop ----

    def process(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        scale = min(width, height) / 720.0
        font_size = max(16, int(24 * scale))
        self.font = self._load_emoji_font(font_size)
        self.text_offset = max(20, int(30 * scale))
        self.box_thickness = max(2, int(2 * scale))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Input : {input_path}  ({width}x{height} @ {fps:.1f} fps, {total_frames} frames)")
        print(f"Output: {output_path}")

        # How many frames between each processing step (≈ 5× per second)
        process_every = max(1, int(round(fps / 5.0)))

        # Tracking state  (same as demo.py)
        active_tracks = {}
        next_track_id = 0
        cached_results = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % process_every == 0:
                cached_results = []

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Scale minimum face size to ~10% of the shorter frame dimension
                min_face = max(60, int(min(width, height) * 0.10))
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 15, minSize=(min_face, min_face))

                current_frame_data = []
                now = frame_idx / fps  # virtual timestamp

                for (x, y, w, h) in faces:
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(width, x + w), min(height, y + h)

                    face_roi = frame[y1:y2, x1:x2]
                    if face_roi.size == 0:
                        continue

                    cx, cy = x + w // 2, y + h // 2
                    face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(face_rgb)
                    input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

                    # XAI branch
                    heatmap = None
                    if self.xai is not None:
                        heatmap, _, xai_probs = self._compute_xai(input_tensor)
                        probs = xai_probs
                    else:
                        with torch.no_grad():
                            outputs = self.model(input_tensor)
                            probs = F.softmax(outputs, dim=1)

                    current_frame_data.append({
                        'coords': (x1, y1, x2, y2),
                        'centroid': (cx, cy),
                        'probs': probs.detach(),
                        'heatmap': heatmap,
                    })

                # --- track matching (identical to demo.py) ---
                used_track_ids = set()

                for data in current_frame_data:
                    cx, cy = data['centroid']
                    best_match_id = None
                    min_dist = float('inf')
                    MAX_DIST = 150.0

                    for t_id, track in active_tracks.items():
                        if t_id in used_track_ids:
                            continue
                        tx, ty = track['centroid']
                        dist = np.sqrt((cx - tx) ** 2 + (cy - ty) ** 2)
                        if dist < min_dist and dist < MAX_DIST:
                            min_dist = dist
                            best_match_id = t_id

                    if best_match_id is not None:
                        active_tracks[best_match_id]['history'].append(data['probs'])
                        active_tracks[best_match_id]['centroid'] = (cx, cy)
                        active_tracks[best_match_id]['last_seen'] = now
                        used_track_ids.add(best_match_id)
                        final_probs = torch.mean(
                            torch.stack(list(active_tracks[best_match_id]['history'])), dim=0
                        )
                    else:
                        new_id = next_track_id
                        next_track_id += 1
                        history = deque(maxlen=3)
                        history.append(data['probs'])
                        active_tracks[new_id] = {
                            'history': history,
                            'centroid': (cx, cy),
                            'last_seen': now,
                        }
                        used_track_ids.add(new_id)
                        final_probs = data['probs']

                    confidence, predicted = torch.max(final_probs, 1)
                    emotion_idx = predicted.item()
                    emotion = CLASSES[emotion_idx]
                    conf_score = confidence.item()

                    emoji = EMOJI_MAP.get(emotion, '')
                    text_str = f"{emotion} ({conf_score * 100:.0f}%)"

                    cached_results.append({
                        'coords': data['coords'],
                        'emoji': emoji,
                        'text': text_str,
                        'color': (0, 255, 0),
                        'heatmap': data['heatmap'],
                    })

                # Cleanup old tracks (>1 s in video time)
                active_tracks = {
                    t_id: track
                    for t_id, track in active_tracks.items()
                    if now - track['last_seen'] < 1.0
                }

            # ---------- draw ----------
            if cached_results:
                for res in cached_results:
                    x1, y1, x2, y2 = res['coords']

                    if res['heatmap'] is not None:
                        frame = overlay_heatmap(frame, res['heatmap'], x1, y1, x2, y2, alpha=0.5)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), res['color'], self.box_thickness)

                    frame = self.draw_complex_text(
                        frame, res['emoji'], res['text'], x1, y1 - self.text_offset, res['color']
                    )

            writer.write(frame)

            frame_idx += 1
            if frame_idx % 100 == 0:
                pct = frame_idx / total_frames * 100 if total_frames else 0
                print(f"  processed {frame_idx}/{total_frames} frames ({pct:.1f}%)")

        cap.release()
        writer.release()
        print(f"Done! {frame_idx} frames written to {output_path}")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a video file — classify emotions with XAI overlay and save the result."
    )
    parser.add_argument("input", type=str, help="Path to the input video file.")
    parser.add_argument("--output", type=str, default=None,
                        help="Path for the output video (default: XAI_<input>.<ext>).")
    parser.add_argument("--checkpoint", type=str, default=MODEL_CHECKPOINT_PATH_RESNET18,
                        help="Path to the model checkpoint.")
    parser.add_argument("--xai", type=str, choices=["gradcam", "ig", "none"], default="gradcam",
                        help="XAI method: gradcam (default), ig, or none.")

    args = parser.parse_args()

    # Build default output path if not given
    if args.output is None:
        dir_name = os.path.dirname(args.input)
        base_name = os.path.basename(args.input)
        name, ext = os.path.splitext(base_name)
        args.output = os.path.join(dir_name, f"XAI_{name}{ext}")

    processor = VideoProcessor(checkpoint_path=args.checkpoint, xai_method=args.xai)
    processor.process(args.input, args.output)
