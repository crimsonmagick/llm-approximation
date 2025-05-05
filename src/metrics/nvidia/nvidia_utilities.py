import atexit
import logging

logger = logging.getLogger(__name__)


def is_cuda_available() -> bool:
    try:
        import pynvml
    except ImportError:
        logger.debug("pynvml is not available", exec_info=True)
        return False
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        logger.debug("Unable to initialize nvml", exec_info=True)
        return False
    
    # TODO should we be looking for compute ability?
    device_count = pynvml.nvmlDeviceGetCount()
    pynvml.nvmlShutdown()
    return device_count > 0


def initialize_nvml():
    import pynvml
    pynvml.nvmlInit()
    atexit.register(pynvml.nvmlShutdown)
