import time

import pynvml


class EnergyRecorder:
    def __init__(self):
        self.end_time = None
        self.end_energy = None
        self.start_energy = None
        self.start_time = None
        self.temperature = None
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # in a simple consumer setup, GPU will be 0
    
    def start(self):
        self.start_energy = self.__get_total_energy()
        self.start_time = time.perf_counter()
        self.end_energy = None
        self.end_time = None
        self.temperature = None
        return self
    
    def __get_total_energy(self):
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
    
    def __get_gpu_temperature(self):
        return pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
    
    def end(self):
        self.end_energy = self.__get_total_energy()
        self.end_time = time.perf_counter()
        self.temperature = self.__get_gpu_temperature()
        return self
    
    def get_metrics(self):
        if self.end_energy is None or self.end_time is None:
            return 0, 0
        energy_consumed_mj = self.end_energy - self.start_energy
        duration_ms = (self.end_time - self.start_time) * 1000
        return energy_consumed_mj, duration_ms * 1000, self.temperature
