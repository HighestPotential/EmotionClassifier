import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import sys
import importlib.util
import time

import argparse
from aleks_resnet18_se import ResNet18SE

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

MODEL_SOURCE_PATH = os.path.join(PROJECT_ROOT, "models", "dmytro", "CCT-7withSAM", "custom_cct.py")
MODEL_CHECKPOINT_PATH_CCT = os.path.join(CURRENT_DIR, "model_checkpoint_epoch_185.pth")
MODEL_CHECKPOINT_PATH_RESNET18 = os.path.join(CURRENT_DIR, "best_resnet18_se.pth")

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

def load_model_dynamically(path):
    print(f"Importing model architecture from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find model file at {path}")

    src_dir = os.path.dirname(path)          
    package_root = os.path.dirname(src_dir)  

    if package_root not in sys.path:
        sys.path.insert(0, package_root)

    package_name = os.path.basename(src_dir) 
    module_name = f"{package_name}.cct"

    spec = importlib.util.spec_from_file_location(module_name, path)
    cct_module = importlib.util.module_from_spec(spec)
    
    sys.modules[module_name] = cct_module
    spec.loader.exec_module(cct_module)
    
    return cct_module.CCT

class EmotionDemo:
    def __init__(self, model_type='cct', checkpoint_path=None):
        self.device = DEVICE
        self.model_type = model_type.lower()
        self.checkpoint_path = checkpoint_path
        print(f"Running on device: {self.device}")
        print(f"Model Type: {self.model_type}")

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
        try:
            # Segoe UI Emoji is the standard Windows emoji font
            self.font = ImageFont.truetype("C:\\Windows\\Fonts\\seguiemj.ttf", 40)
        except OSError:
            print("Warning: Segoe UI Emoji font not found. Emojis might not render.")
            self.font = ImageFont.load_default()

    def _setup_model(self):
        if self.model_type == 'resnet18':
            model = ResNet18SE(num_classes=len(CLASSES), reduction=16)
            
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

        else: # Default to CCT
            CCT_Class = load_model_dynamically(MODEL_SOURCE_PATH)
            model = CCT_Class(
                img_size=IMG_SIZE,
                num_classes=len(CLASSES),
                positional_embedding='learnable'
            )
            
            path = self.checkpoint_path if self.checkpoint_path else MODEL_CHECKPOINT_PATH_CCT
            print(f"Loading weights from: {path}")
            
            if os.path.exists(path):
                state_dict = torch.load(path, map_location=self.device)
                model.load_state_dict(state_dict)
            else:
                 print(f"Warning: Checkpoint not found at {path}. Using random weights.")


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

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            if now - prev_time >= 0.5:
                prev_time = now
                cached_results = [] 
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    x1, y1 = x, y
                    x2, y2 = x + w, y + h
                    
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    
                    face_roi = frame[y1:y2, x1:x2]
                    if face_roi.size == 0: continue

                    face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(face_rgb)
                    
                    input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        outputs = self.model(input_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        confidence, predicted = torch.max(probs, 1)
                        
                        emotion_idx = predicted.item()
                        emotion = CLASSES[emotion_idx]
                        conf_score = confidence.item()

                    emoji = EMOJI_MAP.get(emotion, '')
                    text_str = f"{emotion} ({conf_score*100:.0f}%)"
                    
                    cached_results.append({
                        'coords': (x1, y1, x2, y2),
                        'emoji': emoji,
                        'text': text_str,
                        'color': (0, 255, 0) # Text color (Green)
                    })

            # Draw cached results
            if cached_results:
                for res in cached_results:
                    x1, y1, x2, y2 = res['coords']
                    
                    # Draw Box (OpenCV)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), res['color'], 2)
                    
                    # Draw Emoji + Text (PIL)
                    # We pass the RGB tuple for green (0, 255, 0) for the text part
                    frame = self.draw_complex_text(
                        frame, 
                        res['emoji'], 
                        res['text'], 
                        x1, 
                        y1 - 50, 
                        (0, 255, 0)
                    )

            cv2.imshow('Emotion Demo', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion Classification Demo")
    parser.add_argument("--model", type=str, default="cct", choices=["cct", "resnet18"], help="Model type to use: 'cct' or 'resnet18'")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the model checkpoint file. Required for resnet18 if you want loaded weights.")
    
    args = parser.parse_args()
    if not args.checkpoint and args.model == "resnet18":
        demo = EmotionDemo(model_type=args.model, checkpoint_path=MODEL_CHECKPOINT_PATH_RESNET18)
    elif not args.checkpoint and args.model == "cct":
        demo = EmotionDemo(model_type=args.model, checkpoint_path=MODEL_CHECKPOINT_PATH_CCT)
    else:
        demo = EmotionDemo(model_type=args.model, checkpoint_path=args.checkpoint)

    demo.run()
