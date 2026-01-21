import torch
import subprocess
import sys
import os

print(f"Python Executable: {sys.executable}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available in PyTorch: {torch.cuda.is_available()}")
try:
    print(f"CUDA version PyTorch expects: {torch.version.cuda}")
except:
    print("CUDA version PyTorch expects: Unknown")

# Check CUDA_HOME or related env vars
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not Set')}")

try:
    # Check if NVIDIA driver is visible to the OS
    res = subprocess.check_output(["nvidia-smi"]).decode()
    print("NVIDIA Driver: Detected")
    print(res[:500]) # Print first few lines to verify
except Exception as e:
    print(f"NVIDIA Driver Error: {e}")

import socket
print(f"Running on node: {socket.gethostname()}")
