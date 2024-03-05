import torch


def is_cudable():
    print(f"Torch CUDA version: {torch.__version__}")
    print(f"Device Name: {torch.cuda.get_device_name()}")
    return torch.cuda.is_available()
