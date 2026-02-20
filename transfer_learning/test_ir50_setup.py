import torch
import sys
import os

# Ensure we can import ir50.py
sys.path.append(os.getcwd())
from ir50 import Backbone, load_pretrained_weights

def test_model():
    print("Initializing Backbone with num_classes=6...")
    model = Backbone(num_layers=50, drop_ratio=0.5, mode='ir_se', num_classes=6)
    
    weights_path = 'pretrain/ir50.pth'
    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}")
        return

    print(f"Loading weights from {weights_path}...")
    try:
        checkpoint = torch.load(weights_path, map_location='cpu')
        load_pretrained_weights(model, checkpoint)
        print("Weights loaded successfully (ignoring size mismatch for classifier head).")
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return

    model.eval()
    dummy_input = torch.randn(2, 3, 112, 112) # Standard input size
    print(f"Running partial forward pass with input {dummy_input.shape}...")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    
    if output.shape == (2, 6):
        print("SUCCESS: Output shape matches (batch_size, 6).")
    else:
        print(f"FAILURE: Output shape mismatch. Expected (2, 6), got {output.shape}")

if __name__ == "__main__":
    test_model()
