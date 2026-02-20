import torch
import sys
import os

# Updated path relative to this script location
model_path = "pretrain/ir50.pth"

print(f"Current working directory: {os.getcwd()}")
print(f"Checking for model at: {model_path}")

if not os.path.exists(model_path):
    print(f"ERROR: File not found at {model_path}")
    sys.exit(1)

try:
    print(f"Loading {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    print(f"Successfully loaded. Total keys: {len(state_dict)}")
    
    keys = list(state_dict.keys())
    
    # Check for body/layer depth to confirm 4 stages
    body_keys = [k for k in keys if "body" in k]
    if body_keys:
        indices = set()
        for k in body_keys:
            parts = k.split('.')
            if len(parts) > 1 and parts[0] == 'body' and parts[1].isdigit():
                indices.add(int(parts[1]))
        
        found_stages = sorted(list(indices))
        print(f"Body stages found: {found_stages}")
        
        if 3 in indices: # 0, 1, 2, 3 -> 4 stages
            print("VERIFICATION SUCCESS: Found body.3 (Stage 4). This is a full 4-stage model.")
        else:
            print("VERIFICATION WARNING: Did not find body.3. This might be a truncated model.")
    else:
        print("No 'body' keys found.")

except Exception as e:
    print(f"Error loading checkpoint: {e}")
