import argparse
from dataclasses import dataclass
from typing import Callable, Union

import cv2 as cv
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from captum.attr import IntegratedGradients

from aleks_resnet18_se import ResNet18SE
from retinaface import RetinaFace

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

@dataclass
class ModelContext:
    model: ResNet18SE
    device: torch.device
    classifier: nn.Softmax
    transform: transforms.Compose

@dataclass
class VideoContext:
    input: cv.VideoCapture
    writer: cv.VideoWriter
    detector: RetinaFace

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

    return VideoContext(input=vid,
                        writer=writer,
                        detector=detector,)

def genModelContext():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    softmax = nn.Softmax(dim=1)
    
    model = ResNet18SE()

    modelState = torch.load("./ResNet18_trained.pth", map_location=device, weights_only=False)
    model.load_state_dict(modelState)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return ModelContext(model=model,
                 device=device,
                 classifier=softmax,
                 transform=transform
                )

def contextCleanup(ctx: VideoContext):
    ctx.input.release()
    ctx.writer.release()

def genGradCAM(cam: GradCAM, frame: torch.Tensor, label: int) -> np.ndarray:
    targetEmotion = [ClassifierOutputTarget(label)]

    result = cam(input_tensor=frame, targets=targetEmotion)

    img = frame.squeeze().cpu().numpy().transpose(1, 2, 0)
    img = (img * [0.5, 0.5, 0.5]) + [0.5, 0.5, 0.5]
    img = np.clip(img, 0, 1)

    overlay = show_cam_on_image(img=img, mask=result[0], use_rgb=False, image_weight=0.8)
    return overlay

def genIG(ctx: ModelContext, frame: torch.Tensor, label: int) -> np.ndarray:

    baseline = torch.zeros_like(frame)

    ig = IntegratedGradients(ctx.model)
    attr, _ = ig.attribute(inputs=frame, baselines=baseline, target=label, return_convergence_delta=True)
    
    attr = attr.squeeze(dim=0)
    frame = frame.squeeze(dim=0)

    


def main(videoCtx: VideoContext, modelCtx: ModelContext, saliency_fn: Callable, avgIters: int = 5):
    iteration = 0
    previousEmotion = ""
    accumulator = 0

    while True:
        iteration = iteration % avgIters
        ret, frame = videoCtx.input.read()
        if not ret:
            break
        
        faces = videoCtx.detector.detect_faces(frame)

        for face in faces:
            x1, y1, x2, y2 = faces[f"{face}"]["facial_area"]
            cv.rectangle(frame, (x1, y1), (x2, y2), GREEN, 5)

            originalCrop = frame[y1:y2, x1:x2]
            crop = cv.resize(originalCrop, (64, 64))

            inputImg: torch.Tensor = modelCtx.transform(crop)
            inputImg = inputImg.unsqueeze(0)

            output: torch.Tensor = modelCtx.model(inputImg)
            probs: torch.Tensor = modelCtx.classifier(output)
            confidence, idx = probs.max(dim=1)
            
            idx = idx.item()
            accumulator += confidence.item()
            emotion = modelCtx.model.emotionMap[idx]

            if not iteration:
                printScore = accumulator / avgIters
                accumulator = 0
            elif previousEmotion != emotion:
                printScore = confidence.item()
                accumulator = confidence.item()

            heatMap = saliency_fn(inputImg, idx)
            
            crop_h, crop_w = originalCrop.shape[:2]
            heatMap = cv.resize(heatMap, (crop_w, crop_h))
            frame[y1:y2, x1:x2] = heatMap

            cv.putText(frame, f"{emotion}: {(printScore * 100):.1f}", (10, 100), cv.FONT_HERSHEY_PLAIN, 10, RED, 8)

        videoCtx.writer.write(frame)
        
        iteration += 1
        previousEmotion = emotion

    contextCleanup(videoCtx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "Video classifier", 
                                     description="Takes in a video path, classifies the Emotion and adds a saliency map"
                                     )
    parser.add_argument("filename", help="filepath of the video to process")
    parser.add_argument("-of", help="output path", default="./result.mp4", required=False)
    parser.add_argument("-cam", help="Use GradCAM instead of Integrated gradients for XAI", action="store_true")

    args = parser.parse_args()
    
    inFile = args.filename
    outFile = args.of
    codec = "mp4v"

    vid: VideoContext = genVideoContext(inFile, outFile, codec)
    mod: ModelContext = genModelContext()

    saliency_fn = lambda x, y: genGradCAM(GradCAM(mod.model, [mod.model.layer3[-1]]), x, y) if args.cam else lambda x, y: genIG(mod, x, y)
   
    main(vid, mod, saliency_fn)
