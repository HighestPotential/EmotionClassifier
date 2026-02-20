import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import os
from PIL import Image

# Config
CLASSES = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
IMG_SIZE = 64
MEAN = [0.4681, 0.4447, 0.4560]
STD = [0.2327, 0.2227, 0.2224]
WEIGHTS = '/home/d/dumanskyy/work/EmotionClassifier/models/dmytro/EfficentNetV2/efficientnetv2_last.pth'
IMAGE_PATH = '/home/d/dumanskyy/work/EmotionClassifier/latest_3_0_ready_to_use_datasets/AffectNet/eval/anger/Test_Anger_image0000697.jpg' 

def create_model():
    model = timm.create_model(
        'tf_efficientnetv2_s', 
        pretrained=False, 
        num_classes=len(CLASSES), 
        drop_path_rate=0.2
    )
    # Stem Fix
    old_stem = model.conv_stem
    model.conv_stem = nn.Conv2d(
        in_channels=old_stem.in_channels,
        out_channels=old_stem.out_channels,
        kernel_size=old_stem.kernel_size,
        stride=(1, 1),
        padding=old_stem.padding,
        bias=False
    )
    return model

def main():
    print("--- Debugging Model ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load Model
    model = create_model().to(device)
    try:
        checkpoint = torch.load(WEIGHTS, map_location=device)
        # Try NON-EMA first to verify
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Strip simple prefixes
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        print("Weights loaded.")
        if missing:
            print(f"Missing Keys: {len(missing)}")
            print(f"Sample Missing: {missing[:5]}")
        if unexpected:
            print(f"Unexpected Keys: {len(unexpected)}")
            print(f"Sample Unexpected: {unexpected[:5]}")
            
        # Inspect Classifier
        if 'classifier.bias' in new_state_dict:
            entry = new_state_dict['classifier.bias']
            print(f"Checkpoint Classifier Bias Shape: {entry.shape}")
        elif 'head.fc.bias' in new_state_dict: # Vim?
             print(f"Checkpoint Head Bias Shape: {new_state_dict['head.fc.bias'].shape}")
        else:
            print("Could not find classifier bias in state_dict keys.")
            print(f"First 5 keys: {list(new_state_dict.keys())[:5]}")
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return

    model.eval()
    
    # 2. Check Weights Stats
    print(f"Stem Weight Mean: {model.conv_stem.weight.mean().item()}")
    
    # 3. Find an Image
    # We will search for ANY image in the anger folder to be sure
    anger_dir = '/home/d/dumanskyy/work/EmotionClassifier/latest_3_0_ready_to_use_datasets/AffectNet/test/anger'
    if not os.path.exists(anger_dir):
        print(f"Path not found: {anger_dir}")
        return
        
    img_name = os.listdir(anger_dir)[0]
    img_path = os.path.join(anger_dir, img_name)
    print(f"Testing on image: {img_path}")
    
    # 4. Process
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # 5. Predict
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        
    print("\nPredictions:")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls}: {probs[0][i].item():.4f} (Logit: {logits[0][i].item():.4f})")
        
    print(f"\nPredicted Class: {CLASSES[logits.argmax().item()]}")
    print(f"True Class: anger")

if __name__ == '__main__':
    main()
