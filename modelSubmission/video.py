import argparse
from dataclasses import dataclass

import cv2 as cv
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from aleks_resnet18_se import ResNet18SE
from retinaface import RetinaFace

@dataclass
class VideoContext:
    input: cv.VideoCapture
    writer: cv.VideoWriter
    detector: RetinaFace

@dataclass
class ModelContext:
    model: nn.Module
    device: torch.device
    classifier: nn.Softmax
    transform: transforms.Compose


def genVideoContext(input: str, output: str, codec: str) -> VideoContext:
    vid = cv.VideoCapture(input)
    if not vid.isOpened():
        print("Error opening video")
        exit(1)

    fps = vid.get(cv.CAP_PROP_FPS)
    width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
    fcc = cv.VideoWriter.fourcc(*codec)

    writer = cv.VideoWriter(output, fcc, fps, (width, height))

    detector = RetinaFace

    return VideoContext(vid, writer, detector)

def genModelContext():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    softmax = nn.Softmax(dim=1)
    
    model = ResNet18SE()

    modelState = torch.load("./ResNet18_trained.pth", map_location=device, weights_only=False)
    model.load_state_dict(modelState)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return ModelContext(model=model,
                 device=device,
                 classifier=softmax,
                 transform=transform)

def contextCelanup(ctx: VideoContext):
    ctx.input.release()
    ctx.writer.release()

def main(videoCtx: VideoContext, modelCtx: ModelContext):
    while True:
        ret, frame = videoCtx.input.read()
        if not ret:
            break
        
        faces = videoCtx.detector.detect_faces(frame)

        for face in faces:
            x1, y1, x2, y2 = faces[f"{face}"]["facial_area"]
            cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            crop = frame[y1:y2, x1:x2]
            crop = cv.resize(crop, (64, 64))

            inputImg:torch.Tensor = modelCtx.transform(crop)
            inputImg = inputImg.unsqueeze(0)

            pred: torch.Tensor = modelCtx.model(inputImg)
            probs: torch.Tensor = modelCtx.classifier(pred)
            prob = probs.max(dim=1).item()

            idx = pred.argmax(dim=1).item()
            emotion = modelCtx.model.emotionMap[idx]
            cv.putText(frame, f"{emotion}: {prob}", (x1, y1 - 10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        videoCtx.writer.write(frame)

    contextCelanup(videoCtx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "Video classifier", 
                                     description="Takes in a video path, classifies the Emotion and adds a saliency map"
                                     )
    parser.add_argument("filename", help="filepath of the video to process")
    parser.add_argument("-of", help="output path", default="./result.mp4", required=False)

    args = parser.parse_args()
    
    inFile = args.filename
    outFile = args.of
    codec = "mp4v"

    vid: VideoContext = genVideoContext(inFile, outFile, codec)
    mod: ModelContext = genModelContext()

    main(vid, mod)