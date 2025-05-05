import torch


# TODO decouple from torch and cuda
def get_allocated_memory():
    torch.cuda.memory_allocated()