import torch
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"GPU Device: {torch.cuda.get_device_name(0)}")

if torch.cuda.is_available():
    print("Nvidia GPU detected!")

else:
    print("No Nvidia GPU in system!")
