import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import time
import argparse

from aleks_resnet18_se import ResNet18SE

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "best_resnet18_se_SGD_cbfl.pth")

IMG_SIZE = 64
CLASSES = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class EmotionDemo:
    def __init__(self, checkpoint, camera_index):
        self.device = DEVICE

        cascade_path = os.path.join(CURRENT_DIR, "haarcascade_frontalface_default.xml")
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError("Haar cascade not found")

        self.model = ResNet18SE(num_classes=len(CLASSES), reduction=16)
        if isinstance(self.model.fc, torch.nn.Linear):
            in_f = self.model.fc.in_features
            self.model.fc = torch.nn.Sequential(
                torch.nn.Dropout(0.3),
                torch.nn.Linear(in_f, len(CLASSES))
            )

        state = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            raise RuntimeError("Camera not accessible")

        self.process_interval = 1.0 / 3.0
        self.last_process_time = 0.0
        self.last_second_time = time.time()

        self.prob_buffer = []
        self.last_bbox = None
        self.display_text = None

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            now = time.time()

            if now - self.last_process_time >= self.process_interval:
                self.last_process_time = now

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, 1.1, 5, minSize=(30, 30)
                )

                for (x, y, w, h) in faces:
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                    face = frame[y1:y2, x1:x2]
                    if face.size == 0:
                        continue

                    self.last_bbox = (x1, y1, x2, y2)

                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(face_rgb)
                    inp = self.transform(pil).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        probs = torch.softmax(self.model(inp), dim=1)

                    self.prob_buffer.append(probs)

            if now - self.last_second_time >= 1.0 and self.prob_buffer:
                mean_probs = torch.mean(torch.cat(self.prob_buffer, dim=0), dim=0)
                conf, idx = torch.max(mean_probs, 0)
                emotion = CLASSES[idx.item()]
                self.display_text = f"{emotion} ({conf.item()*100:.0f}%)"
                self.prob_buffer.clear()
                self.last_second_time = now

            if self.display_text and self.last_bbox:
                x1, y1, x2, y2 = self.last_bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    self.display_text,
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            cv2.imshow("Emotion Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=MODEL_PATH)
    parser.add_argument("--camera", type=int, default=1)
    args = parser.parse_args()

    demo = EmotionDemo(args.checkpoint, args.camera)
    demo.run()