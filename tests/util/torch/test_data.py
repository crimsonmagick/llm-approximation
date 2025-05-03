import json
from tokenize import String

import torch


def load_tensor(relative_path: String):
    with open('../data/' + relative_path, "r") as file:
        return torch.tensor(json.load(file))