import torch
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)
print("Built with CUDA?:", torch.backends.cudnn.is_available())
