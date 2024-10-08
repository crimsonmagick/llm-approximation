import logging
import time

import pynvml

logger = logging.getLogger(__name__)


class MetricsRecorder:
    def __init__(self):
        self.end_time = None
        self.end_energy = None
        self.start_energy = None
        self.start_time = None
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # in a simple consumer setup, GPU will be 0
    
    def start(self):
        self.start_energy = self.__get_total_energy()
        self.start_time = time.time()
        self.end_energy = None
        self.end_time = None
        return self
    
    def __get_total_energy(self):
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
    
    def end(self):
        self.end_energy = self.__get_total_energy()
        self.end_time = time.time()
        return self
    
    def get_metrics(self):
        if self.end_energy is None or self.end_time is None:
            return 0, 0
        energy_consumed_mj = self.end_energy - self.start_energy
        duration = self.end_time - self.start_time
        return energy_consumed_mj, duration * 1000
    
    def __del__(self):
        pynvml.nvmlShutdown()


def capture_evaluation(func):
    class CaptureEvaluation:
        def __init__(self, func):
            self.generated_token_count = 0
            self.execution_time_ms = 0
            self.evaluation_count = 0
            self.func = func
            self.energy_usage_mj = 0
            self.recorder = MetricsRecorder()
        
        def capture(self, *args, **kwargs):
            self.recorder.start()
            evaluation = self.func(*args, **kwargs)
            energy_usage_mj, execution_time_ms = self.recorder.end().get_metrics()
            self.generated_token_count += evaluation[1]
            self.energy_usage_mj += energy_usage_mj
            self.execution_time_ms = execution_time_ms
            average_time_per_token_ms = self.execution_time_ms / self.generated_token_count
            average_energy_per_token_mj = self.energy_usage_mj / self.generated_token_count
            logger.info(f"evaluation_count={self.evaluation_count}, execution_time={execution_time_ms / 1000:.2f} s, "
                        f"energy_usage={energy_usage_mj / 1000:.2f} j")
            logger.info(
                f"average_time_per_token={average_time_per_token_ms:.2f} ms, "
                f"average_energy_per_token_mj={average_energy_per_token_mj / 1000:.2f} j")
            self.evaluation_count += 1
            return evaluation
    
    capture = CaptureEvaluation(func)
    return lambda *args, **kwargs: capture.capture(*args, **kwargs)
