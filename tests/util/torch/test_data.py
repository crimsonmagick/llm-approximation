import json
from pathlib import Path
from tokenize import String

import torch
from torch import Tensor

DATA_DIR = Path(__file__).parent.parent.parent / "data"

def load_tensor(relative_path: String, device) -> Tensor:
    with open(DATA_DIR / relative_path, "r") as file:
        return torch.tensor(json.load(file), device=device)


def save_tensor(tensor: Tensor, relative_path: String):
    with open(DATA_DIR / relative_path, "w") as file:
        json.dump(tensor.tolist(), file)
