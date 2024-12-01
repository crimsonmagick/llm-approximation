import logging
import time

import torch

import pynvml
from torch import tensor

logger = logging.getLogger(__name__)


class MetricsManager:
    def __init__(self):
        self._perplexity = None
        self._execution_time_ms = None
        self._average_energy_per_token_mj = None
        self._average_time_per_token_ms = None
        self._total_energy = None
        self._allocated_memory = None
        self._layer_idx = None
        self._head_idxs = None
        self._saved_metrics = dict()
        self.header = (
            'label',
            'layer_idx',
            'head_idxs',
            'perplexity',
            'average_energy_per_token_mj',
            'average_time_per_token_ms',
            'allocated_memory'
        )
    
    def clear(self):
        self._perplexity = None
        self._execution_time_ms = None
        self._average_energy_per_token_mj = None
        self._average_time_per_token_ms = None
        self._total_energy = None
        self._allocated_memory = None
        self._layer_idx = None
        self._head_idxs = None
    
    def perplexity(self, perplexity):
        self._perplexity = perplexity
        return self
    
    def execution_time_ms(self, execution_time_ms):
        self._execution_time_ms = int(execution_time_ms)
        return self
    
    def average_energy_per_token_mj(self, average_energy_per_token_mj):
        self._average_energy_per_token_mj = average_energy_per_token_mj
        return self
    
    def average_time_per_token_ms(self, average_time_per_token_ms):
        self._average_time_per_token_ms = average_time_per_token_ms
        return self
    
    def total_energy(self, total_energy):
        self._total_energy = total_energy
        return self
    
    def allocated_memory(self, allocated_memory):
        self._allocated_memory = allocated_memory
        return self
    
    def layer_idx(self, layer_idx):
        self._layer_idx = layer_idx
        return self
    
    def head_idxs(self, head_idxs):
        self._head_idxs = head_idxs
        return self
    
    def save_metrics(self, label):
        self._saved_metrics[label] = (
            label, self._layer_idx, self._head_idxs, self._perplexity, self._average_energy_per_token_mj,
            self._average_time_per_token_ms, self._allocated_memory)
        return self
    
    def get_metrics(self):
        return [self.header] + list(self._saved_metrics.values())


singleton = MetricsManager()


def metrics_manager():
    return singleton


class EnergyRecorder:
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
        def __init__(self, instance):
            self.token_count = 0
            self.execution_time_ms = 0
            self.batch_count = 0
            self.func = func
            self.instance = instance
            self.energy_usage_mj = 0
        
        def capture(self, *args, **kwargs):
            recorder = EnergyRecorder().start()
            evaluation = self.func(self.instance, *args, **kwargs)
            energy_usage_mj, execution_time_ms = recorder.end().get_metrics()
            self.token_count += evaluation[1]
            self.energy_usage_mj += energy_usage_mj
            self.execution_time_ms += execution_time_ms
            self.batch_count += 1
            average_time_per_token_ms = self.execution_time_ms / self.token_count
            average_energy_per_token_mj = self.energy_usage_mj / self.token_count
            logger.info(
                f"batch_count={self.batch_count}, execution_time={self.execution_time_ms / 1000:.2f} s, "
                f"energy_usage={self.energy_usage_mj / 1000:.2f} j")
            logger.info(
                f"average_time_per_token={average_time_per_token_ms:.2f} ms, "
                f"average_energy_per_token_mj={average_energy_per_token_mj / 1000:.2f} j")
            logger.debug(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
            logger.debug(f"Reserved Memory: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
            metrics_manager() \
                .execution_time_ms(self.execution_time_ms) \
                .total_energy(self.energy_usage_mj) \
                .average_time_per_token_ms(average_time_per_token_ms) \
                .average_energy_per_token_mj(average_energy_per_token_mj) \
                .allocated_memory(torch.cuda.memory_allocated())
            
            return evaluation
    
    def wrapper(self, *args, **kwargs):
        # Lazily create a CaptureEvaluation instance if it doesn't exist
        if not hasattr(self, '_capture_evaluation'):
            self._capture_evaluation = CaptureEvaluation(self)
        
        return self._capture_evaluation.capture(*args, **kwargs)
    
    return wrapper


def capture_loss(func):
    class CaptureLoss:
        def __init__(self, instance):
            self.token_count = 0
            self.aggregate_loss = 0
            self.instance = instance
        
        def capture(self, *args, **kwargs):
            token_sequences: tensor = args[0]['input_ids']
            for sequence in token_sequences:
                self.token_count += len(
                    sequence) - 1  # can't count first token, is not generated as a part of evaluation
            token_losses = func(self.instance, *args, **kwargs)
            self.aggregate_loss += token_losses.sum()
            perplexity = self.aggregate_loss / self.token_count
            logger.info(f'Perplexity: {perplexity}')
            metrics_manager().perplexity(perplexity.item())
            return token_losses
    
    def wrapper(self, *args, **kwargs):
        # Lazily create a CaptureLoss instance if it doesn't exist
        if not hasattr(self, '_capture_loss'):
            self._capture_loss = CaptureLoss(self)
        
        return self._capture_loss.capture(*args, **kwargs)
    
    return wrapper
