import torch


def get_device() -> torch.device:
    # TODO allow setting the device from system or env args
    # return torch.get_default_device()
    return torch.device("cuda")