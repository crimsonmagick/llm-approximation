from enum import Enum

from .nvidia import nvidia_utilities

configuration = {}


class ComputePlatform(Enum):
    CPU = 'CPU'
    CUDA = 'CUDA'
    ROCM = 'ROCM'
    ONE_API = 'ONE_API'


if nvidia_utilities.is_cuda_available():
    nvidia_utilities.initialize_nvml()
    configuration['compute_platform'] = ComputePlatform.CUDA
else:
    configuration['compute_platform'] = ComputePlatform.CPU