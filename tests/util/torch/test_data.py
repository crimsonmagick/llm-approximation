import json
from tokenize import String

import torch
from torch import Tensor


def load_tensor(relative_path: String, device) -> Tensor:
    with open('../data/' + relative_path, "r") as file:
        return torch.tensor(json.load(file), device=device)


def save_tensor(tensor: Tensor, relative_path: String):
    with open('../data/' + relative_path, "w") as file:
        json.dump(tensor.tolist(), file)
