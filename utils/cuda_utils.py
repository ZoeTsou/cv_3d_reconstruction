import torch

def get_available_gpu(threshold_mb=1000):
    for i in range(torch.cuda.device_count()):
        stats = torch.cuda.memory_reserved(i) / 1024 / 1024  # MB
        if stats < threshold_mb:
            return i
    return 0  # fallback

