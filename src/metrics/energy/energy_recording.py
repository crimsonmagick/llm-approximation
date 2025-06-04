import pynvml
import torch
from torch.cuda import Event


class EnergyRecorder:
    def __init__(self):
        self.start_event = Event(enable_timing=True)
        self.end_event = Event(enable_timing=True)
        self.end_energy = None
        self.start_energy = None
        self.elapsed_time = None
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # in a simple consumer setup, GPU will be 0

    def start(self):
        torch.cuda.synchronize()
        self.start_event.record()
        self.start_energy = self.__get_total_energy()
        self.end_energy = None
        return self

    def __get_total_energy(self):
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)

    def end(self):
        self.end_event.record()
        torch.cuda.synchronize()
        self.end_energy = self.__get_total_energy()
        self.elapsed_time = self.start_event.elapsed_time(self.end_event)
        return self

    def get_metrics(self):
        if self.end_energy is None or self.elapsed_time is None:
            return 0, 0
        energy_consumed_mj = self.end_energy - self.start_energy
        return energy_consumed_mj, self.elapsed_time
