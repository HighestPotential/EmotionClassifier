import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import sys
import time
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



class EmotionDemo:
    def __init__(self, checkpoint_path=None):
        self.device = DEVICE
        self.checkpoint_path = checkpoint_path
        print(f"Running on device: {self.device}")
        
        # Haar Cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        self.model = self._setup_model()
        
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Load Font
        self.font = self._load_emoji_font()

    def _load_emoji_font(self):
        font_size = 24
        system = platform.system()
        font_path = None
        
        print(f"Detected OS: {system}")

        if system == "Windows":
            # Segoe UI Emoji is the standard Windows emoji font
            font_path = "C:\\Windows\\Fonts\\seguiemj.ttf"

        elif system == "Darwin":
            font_path = "/System/Library/Fonts/Apple Color Emoji.ttc"
        
        # Try loading system font
        if font_path and os.path.exists(font_path):
             try:
                 print(f"Loading system emoji font from: {font_path}")
                 return ImageFont.truetype(font_path, font_size)
             except Exception as e:
                 print(f"Failed to load system font: {e}")

        # Fallback: Check for local NotoColorEmoji or download it
        local_font_name = "NotoColorEmoji.ttf"
        local_font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), local_font_name)
        
        if not os.path.exists(local_font_path):
            print(f"System emoji font not found. Downloading {local_font_name}...")
            url = "https://github.com/googlefonts/noto-emoji/raw/main/fonts/NotoColorEmoji.ttf"
            try:
                urllib.request.urlretrieve(url, local_font_path)
                print("Download complete.")
            except Exception as e:
                print(f"Failed to download font: {e}")
                print("Warning: Emojis might not render correctly.")
                return ImageFont.load_default()

        try:
            print(f"Loading local emoji font from: {local_font_path}")
            return ImageFont.truetype(local_font_path, font_size)
        except Exception as e:
            print(f"Failed to load local font: {e}")
            return ImageFont.load_default()

    def _setup_model(self):
        # Always ResNet18
        model = ResNet18SE(num_classes=len(CLASSES))
        
        if hasattr(model, "fc") and isinstance(model.fc, torch.nn.Linear):
            in_f = model.fc.in_features
            model.fc = torch.nn.Sequential(
                torch.nn.Dropout(p=0.3),
                torch.nn.Linear(in_f, len(CLASSES))
            )

        if not self.checkpoint_path:
            print("Warning: No checkpoint path provided for ResNet18. Initializing with random weights.")
        else:
            print(f"Loading weights from: {self.checkpoint_path}")
            try:
                state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            except Exception as e:
                print(f"Error loading weights: {e}")
                raise e
            
            model.load_state_dict(state_dict)

        model.to(self.device)
        model.eval()
        return model

    def draw_complex_text(self, frame, emoji, text, x, y, text_color):
        """
        Draws the emoji in its natural color (using embedded_color=True) 
        and the text in the specified text_color.
        """
        cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)
        draw = ImageDraw.Draw(pil_im)

        # 1. Draw Emoji (Native Color)
        try:
            draw.text((x, y), emoji, font=self.font, embedded_color=True)
        except TypeError:
            # Fallback for older Pillow versions (<10.1.0)
            draw.text((x, y), emoji, font=self.font, fill=(255,255,255))
        
        # 2. Calculate offset to draw text next to emoji
        # getlength returns the width of the string in pixels
        emoji_width = self.font.getlength(emoji)
        
        # 3. Draw Text (Custom Color, e.g., Green)
        draw.text((x + emoji_width + 5, y), text, font=self.font, fill=text_color)

        return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Webcam not found.")
            return

        print("Starting Demo... Press 'q' to quit.")

        prev_time = 0
        cached_results = []
        
        # Tracking state
        # structure: { id: {'history': deque(maxlen=3), 'centroid': (cx, cy), 'last_seen': time} }
        active_tracks = {} 
        next_track_id = 0
        
        # Target FPS for processing (3 times per second)
        PROCESS_INTERVAL = 1.0 / 3.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            if now - prev_time >= PROCESS_INTERVAL:
                prev_time = now
                cached_results = [] 
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #makes the model do detection faster in grayscale
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

                # Current frame detections
                current_frame_data = []

                for (x, y, w, h) in faces:
                    x1, y1 = x, y
                    x2, y2 = x + w, y + h
                    
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    
                    face_roi = frame[y1:y2, x1:x2]
                    if face_roi.size == 0: continue
                    
                    cx, cy = x + w // 2, y + h // 2

                    face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(face_rgb)
                    
                    input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        outputs = self.model(input_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        # Store tensor directly for averaging later
                    
                    current_frame_data.append({
                        'coords': (x1, y1, x2, y2),
                        'centroid': (cx, cy),
                        'probs': probs
                    })

                # Match to existing tracks
                used_track_ids = set()
                
                for data in current_frame_data:
                    cx, cy = data['centroid']
                    best_match_id = None
                    min_dist = float('inf')
                    
                    # Search radius
                    MAX_DIST = 150.0 

                    for t_id, track in active_tracks.items():
                        if t_id in used_track_ids:
                            continue
                        
                        tx, ty = track['centroid']
                        dist = np.sqrt((cx - tx)**2 + (cy - ty)**2)
                        
                        if dist < min_dist and dist < MAX_DIST:
                            min_dist = dist
                            best_match_id = t_id
                    
                    if best_match_id is not None:
                        # Update existing track
                        active_tracks[best_match_id]['history'].append(data['probs'])
                        active_tracks[best_match_id]['centroid'] = (cx, cy)
                        active_tracks[best_match_id]['last_seen'] = now
                        used_track_ids.add(best_match_id)
                        final_probs = torch.mean(torch.stack(list(active_tracks[best_match_id]['history'])), dim=0)
                    else:
                        # New track
                        new_id = next_track_id
                        next_track_id += 1
                        history = deque(maxlen=3)
                        history.append(data['probs'])
                        
                        active_tracks[new_id] = {
                            'history': history,
                            'centroid': (cx, cy),
                            'last_seen': now
                        }
                        used_track_ids.add(new_id)
                        final_probs = data['probs']
                    
                    # Prepare result for display
                    confidence, predicted = torch.max(final_probs, 1)
                    emotion_idx = predicted.item()
                    emotion = CLASSES[emotion_idx]
                    conf_score = confidence.item()

                    emoji = EMOJI_MAP.get(emotion, '')
                    text_str = f"{emotion} ({conf_score*100:.0f}%)"
                    
                    cached_results.append({
                        'coords': data['coords'],
                        'emoji': emoji,
                        'text': text_str,
                        'color': (0, 255, 0)
                    })

                # Cleanup old tracks
                active_tracks = {
                    t_id: track 
                    for t_id, track in active_tracks.items() 
                    if now - track['last_seen'] < 1.0
                }

            # Draw cached results
            if cached_results:
                for res in cached_results:
                    x1, y1, x2, y2 = res['coords']
                    
                    # Draw Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), res['color'], 2)
                    
                    # Draw Emoji + Text
                    frame = self.draw_complex_text(
                        frame, 
                        res['emoji'], 
                        res['text'], 
                        x1, 
                        y1 - 30, 
                        res['color']
                    )

            cv2.imshow('Emotion Demo', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion Classification Demo")
    parser.add_argument("--checkpoint", type=str, default=MODEL_CHECKPOINT_PATH_RESNET18, help="Path to the model checkpoint file. Defaults to bundled ResNet18 checkpoint.")
    
    args = parser.parse_args()
    
    # Simple instantiation without model_type
    if args.checkpoint:
        demo = EmotionDemo(checkpoint_path=args.checkpoint)
    else:
        # Fallback to default path constant
        demo = EmotionDemo(checkpoint_path=MODEL_CHECKPOINT_PATH_RESNET18)

    demo.run()
