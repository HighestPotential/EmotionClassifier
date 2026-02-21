# EmotionClassifier

A comprehensive repository for emotion classification using various deep learning architectures, including EfficientNetV2, Vision Mamba (Vim), ResNet18, and IR50. This project includes a complete pipeline from preprocessing and training to a functional video processing demo with Explainable AI (XAI) capabilities.

| GradCAM | Integrated Gradients |
|---------|---------------------|
| ![GradCAM Output](resources/XAI_reactions.gif) | ![Integrated Gradients Output](resources/XAI_IG_reactions.gif) |
## Demo

The demo demonstrates real-time emotion classification on video files or GIFs. It detects faces, classifies their prevailing emotion, and optionally overlays visual explanations (heatmaps) to show which parts of the face contributed most to the decision.



### How to Run

To run the demo, use the `process_video.py` script located in the `demo` directory.

```bash
python demo/process_video.py <input_video_path> [options]
```

**Examples:**

```bash
# Basic usage
python demo/process_video.py inputs/my_video.mp4

# Initial run with GradCAM XAI (default)
python demo/process_video.py inputs/my_video.mp4 --xai gradcam

# Use Integrated Gradients for XAI
python demo/process_video.py inputs/my_video.mp4 --xai ig

# Use Haar Cascade/RetinaFace detector
python demo/process_video.py inputs/my_video.mp4 --face-detector haarcascade

# Save to specific output
python demo/process_video.py inputs/my_video.mp4 --output results/output.mp4
```

### Process Video Script (`demo/process_video.py`)

The `process_video.py` script handles the entire inference pipeline:
1.  **Input Reading**: Supports both standard video formats (`.mp4`, `.avi`, etc.) and `.gif` files via a custom `GifReader`.
2.  **Face Detection**: capable of using either **RetinaFace** (default, higher accuracy) or **Haar Cascades** (faster, lower accuracy).
3.  **Preprocessing**: Crops faces, resizes them to 64x64, and normalizes them to match training conditions.
4.  **Inference**: Runs the loaded model (e.g., ResNet18SE) to predict one of 6 emotions: anger, disgust, fear, happiness, sadness, surprise.
5.  **XAI Overlay**: Generates heatmaps using **GradCAM** or **Integrated Gradients** to visualize model focus.
6.  **Output**: Annotates the video with bounding boxes, emotion labels, confidence scores, and emojis. Saves as video or GIF depending on the output extension.

## Models

 The repository contains implementations of several state-of-the-art models for emotion recognition, organized by contributor/experiment:

*   **`models/dmytro`**:
    *   **EfficientNetV2**: Multiple variations (v1, v2, v4, v5) experimenting with optimizers (SGD, AdamW), loss functions (LDAM, Class-Balanced Focal Loss), and resolution tuning.
    *   **Vision Mamba (Vim)**: Experiments with attention-free state space models for vision.
    *   **CCT (Compact Convolutional Transformer)**: Lightweight transformer-based models.
    *   **ConvNeXt**: Modern ConvNet architectures.
*   **`models/aleks`**:
    *   **ResNet18**: Baseline and improved versions (ResNet18-SE) incorporating Squeeze-and-Excitation blocks and specialized training strategies.

## Preprocessing

The preprocessing pipeline ensures high-quality input data by filtering and normalizing raw images before training.

### Pipeline Structure
The preprocessing is managed by a pipeline system (see `preprocessing/pipeline.py` and `preprocessing/dataset_runner.py`) that applies a sequence of steps to every image in a dataset folder.

### Filters and Steps
The standard preprocessing pipeline includes the following steps:

1.  **`FaceExistenceFilter`**: Ensures a face is actually present in the image with high confidence.
2.  **`FaceRotationFilter`**: Corrects or filters images based on yaw rotation (looking left/right).
3.  **`FaceOrientationFilter`**: Corrects in-plane rotation (roll correction).
4.  **`LightingFilter`**: Filters out images with poor lighting conditions (too dark/bright).
5.  **`RemoveBlurredFaces`**: Detects and removes blurry images.
6.  **`CroppingFace`**: Tight crops around the detected face.
7.  **`ResizingTo64`**: Resizes the cropped face to a standard 64x64 resolution.
8.  **`ToRGB`**: Converts images to 3-channel RGB.

### Deduplication
A **Duplication Pre-processing** step (refer to `preprocessing/image_duplicates.py`) is used to clean datasets. It employs **Difference Hashing (dHash)** to identify and remove near-duplicate images, ensuring that the model doesn't overfit to repeated data.

## Transfer Learning

The `transfer_learning` directory contains scripts for fine-tuning pre-trained models, specifically focusing on **IR50** (ResNet50-IR).

*   **Objective**: To leverage powerful pre-trained face recognition features (originally trained on CelebA) for emotion classification. The pre-trained weights were sourced from [Qualitative Content Selection (QCS)](https://github.com/birdwcp/QCS/tree/main).
*   **Evolution**:
    1.  **Original**: Fine-tuning the standard IR50 backbone.
    2.  **Layer Freezing**: Experimenting with freezing different blocks of the backbone (`train_ir50_layer_freezing.py`) to prevent overfitting and preserve low-level features.
    3.  **Light Head (Final Version)**: The final iteration (`train_ir50_light_head.py`) replaces the massive 12.8M parameter fully-connected head with a lightweight, attention-based head (~795K params).
*   **Key Script**: `transfer_learning/train_ir50_light_head.py`
*   **Techniques**:
    *   **Light Head Architecture**: Uses **Spatial Attention** (to focus on key facial regions like eyes/mouth) followed by Global Average Pooling, significantly reducing parameters while maintaining spatial awareness.
    *   **Class Balanced Focal Loss (CBFL)**: Handles class imbalance in emotion datasets.
    *   **Differential Learning Rates**: Fine-tunes the backbone gently (low LR) while training the new head aggressively (high LR).
    *   **Mixup**: Data augmentation technique aimed at improving generalization.
