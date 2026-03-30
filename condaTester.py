import torch
import skrl
import gymnasium
import pettingzoo
import psutil

print(f"--- Verification for RLRobocup environment ---")
print(f"PyTorch version: {torch.__version__}")
print(f"skrl version:    {skrl.__version__}")
print(f"Gymnasium:       {gymnasium.__version__}")
print(f"PettingZoo:      {pettingzoo.__version__}")
print(f"psutil:          {psutil.__version__}")

cuda_available = torch.cuda.is_available()
print(f"CUDA status:     {'ENABLED' if cuda_available else 'DISABLED'}")

if cuda_available:
    print(f"Device Name:     {torch.cuda.get_device_name(0)}")
    print(f"Memory Alloc:    {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
else:
    print("\nWARNING: No se detectó GPU CUDA. Revisa los drivers de NVIDIA.")