import torch

def auto_device():
    device = "mps" 
    print("Device: ", device)
    return device