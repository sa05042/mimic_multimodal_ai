import torch
import random
import numpy as np
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(model, output_dir, filename="best_model.pth"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    torch.save(model.state_dict(), path)
    print(f"Checkpoint saved: {path}")