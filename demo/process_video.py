"""
Offline video processor — reads a video file, detects faces, classifies
emotions with optional XAI heatmap overlays (GradCAM / Integrated Gradients),
and writes the annotated result to a new video file.

Face detection back-ends
------------------------
- **retinaface** (default) — deep-learning detector, more accurate.
- **haarcascade** — classic OpenCV Haar-cascade, faster but less robust.

Usage examples
--------------
    python process_video.py input.mp4 --xai gradcam
    python process_video.py input.mp4 --xai gradcam --face-detector haarcascade
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

class GifReader:
    """Wrapper around PIL that mimics cv2.VideoCapture interface."""
    def __init__(self, path):
        self.img = Image.open(path)
        self.total_frames = getattr(self.img, 'n_frames', 1)
        # Duration is per-frame delay in ms. Default to 100ms (10fps).
        duration = self.img.info.get('duration', 100.0)
        self.fps = 1000.0 / max(1.0, duration) if duration else 10.0
        self.current_frame = 0
        self.width, self.height = self.img.size
    
    def isOpened(self):
        return True
    
    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FPS: return self.fps
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH: return self.width
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT: return self.height
        if prop_id == cv2.CAP_PROP_FRAME_COUNT: return self.total_frames
        return 0
    
    def read(self):
        if self.current_frame >= self.total_frames:
            return False, None
        try:
            self.img.seek(self.current_frame)
            rgb_im = self.img.convert('RGB')
            # Convert PIL RGB -> CV2 BGR
            frame_bgr = cv2.cvtColor(np.array(rgb_im), cv2.COLOR_RGB2BGR)
            self.current_frame += 1
            return True, frame_bgr
        except EOFError:
            return False, None
    
    def release(self):
        self.img.close()

class VideoProcessor:
    def __init__(self, checkpoint_path, xai_method="none", face_detector="retinaface"):
        self.device = DEVICE
        self.xai_method = xai_method
        self.face_detector = face_detector
        print(f"Running on device: {self.device}")
        print(f"XAI method: {self.xai_method}")
        print(f"Face detector: {self.face_detector}")

        # Face detection back-end
        if self.face_detector == "haarcascade":
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        else:
            from batch_face import RetinaFace as _RetinaFace
            self._retina_detector = _RetinaFace(
                gpu_id=0 if self.device == 'cuda' else -1
            )

        self.model = self._setup_model(checkpoint_path)

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # XAI helper
        self.xai = None
        if self.xai_method == "gradcam":
            target_layer = self.model.layer3[-1]
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

    def draw_complex_text(self, frame, emoji, text, x, y, text_color, face_w):
        # On Windows/Mac (where emojis are enabled), use PIL for drawing
        if emoji:
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

        # On Linux/Other (ASCII only), use cv2.putText for reliable scaling
        # User requested size relative to face width
        # Heuristic: face_w=200px -> scale=1.0
        font_scale = max(0.5, face_w / 200.0)
        thickness = max(1, int(font_scale * 2))
        
        cv2.putText(
            frame,
            text,
            (x, y),  # cv2 draws from bottom-left; we might need to adjust y if it's top-left
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA
        )
        return frame

    # ---- XAI ----

    def _compute_xai(self, input_tensor):
        if self.xai is None:
            return None, None, None
        return self.xai(input_tensor)

    # ---- face detection ----

    def _detect_faces(self, frame, frame_w, frame_h):
        """Return a list of (x, y, w, h) bounding boxes."""
        if self.face_detector == "haarcascade":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            min_face = max(60, int(min(frame_w, frame_h) * 0.10))
            detections = self.face_cascade.detectMultiScale(
                gray, 1.1, 15, minSize=(min_face, min_face)
            )
            return list(detections)

        # RetinaFace (batch_face API)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = self._retina_detector(rgb, threshold=0.5)
        faces = []
        if detections:
            for box, _landmarks, _score in detections:
                x1, y1, x2, y2 = map(int, box)
                faces.append((x1, y1, x2 - x1, y2 - y1))
        return faces

    # ---- main loop ----

    def process(self, input_path, output_path):
        is_gif_input = input_path.lower().endswith('.gif')
        is_gif_output = output_path.lower().endswith('.gif')

        if is_gif_input:
            cap = GifReader(input_path)
            print(f"Using GifReader for input: {input_path}")
        else:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        scale = min(width, height) / 720.0
        # For PIL (Windows/Mac) scaling
        font_size = max(40, int(80 * scale))
        self.font = self._load_emoji_font(font_size)
        
        # Box thickness scaled to resolution
        self.box_thickness = max(2, int(4 * scale))

        writer = None
        gif_frames = []

        if is_gif_output:
            print(f"Output mode: GIF (frames will be buffered in memory)")
        else:
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
        
        # Check OS for emoji support
        use_emoji = platform.system() in ["Windows", "Darwin"]

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % process_every == 0:
                cached_results = []

                faces = self._detect_faces(frame, width, height)

                current_frame_data = []
                now = frame_idx / fps  # virtual timestamp

                for (x1, y1, w, h) in faces:
                    x2, y2 = x1 + w, y1 + h
                    # clamp
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)

                    face_roi = frame[y1:y2, x1:x2]
                    if face_roi.size == 0:
                        continue

                    cx, cy = x1 + w // 2, y1 + h // 2
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

                # --- track matching ---
                used_track_ids = set()

                for data in current_frame_data:
                    cx, cy = data['centroid']
                    best_match_id = None
                    min_dist = float('inf')
                    MAX_DIST = 150.0 * scale  # scale distance threshold too

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

                    emoji = EMOJI_MAP.get(emotion, '') if use_emoji else ''
                    text_str = f"{emotion} ({conf_score * 100:.0f}%)"

                    cached_results.append({
                        'coords': data['coords'],
                        'emoji': emoji,
                        'text': text_str,
                        'color': (0, 255, 0),
                        'heatmap': data['heatmap'],
                    })

                # Cleanup old tracks (>1 s)
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
                        frame,
                        res['emoji'],
                        res['text'],
                        x1,
                        y1 - max(30, int((x2 - x1) * 0.2)), # dynamic offset based on face width (20%)
                        res['color'],
                        face_w=(x2 - x1)
                    )

            if is_gif_output:
                # Convert BGR -> RGB for PIL
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gif_frames.append(Image.fromarray(rgb_frame))
            else:
                writer.write(frame)

            frame_idx += 1
            if frame_idx % 100 == 0:
                pct = frame_idx / total_frames * 100 if total_frames else 0
                print(f"  processed {frame_idx}/{total_frames} frames ({pct:.1f}%)")

        cap.release()
        if writer:
            writer.release()
        
        if is_gif_output and gif_frames:
            print(f"Saving GIF with {len(gif_frames)} frames...")
            # duration is in ms
            duration_ms = int(1000.0 / fps) if fps > 0 else 100
            gif_frames[0].save(
                output_path,
                save_all=True,
                append_images=gif_frames[1:],
                duration=duration_ms,
                loop=0
            )

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
    parser.add_argument("--face-detector", type=str,
                        choices=["retinaface", "haarcascade"], default="retinaface",
                        help="Face detection back-end: retinaface (default) or haarcascade.")

    args = parser.parse_args()

    # Build default output path if not given
    if args.output is None:
        dir_name = os.path.dirname(args.input)
        base_name = os.path.basename(args.input)
        name, ext = os.path.splitext(base_name)
        args.output = os.path.join(dir_name, f"XAI_{name}{ext}")

    processor = VideoProcessor(
        checkpoint_path=args.checkpoint,
        xai_method=args.xai,
        face_detector=args.face_detector,
    )
    processor.process(args.input, args.output)
