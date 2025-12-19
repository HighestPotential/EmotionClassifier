import torch

def get_gpu_info():
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Device Count: {torch.cuda.device_count()}")
    else:
        print("CUDA is not available. Check your NVIDIA drivers and PyTorch version.")

get_gpu_info()
