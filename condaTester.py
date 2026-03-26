import torch
import gymnasium as gym
import pettingzoo
import skrl
import numpy as np
import matplotlib.pyplot as plt

print(f"PyTorch Version: {torch.__version__} | GPU Available: {torch.cuda.is_available()}")
print(f"PettingZoo Version: {pettingzoo.__version__}")
print(f"SKRL Version: {skrl.__version__}")
print("All MAPPO dependencies successfully loaded!")