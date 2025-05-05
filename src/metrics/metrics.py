import logging

from .energy.energy_recording import EnergyRecorder
from .function import objective
from .memory import get_allocated_memory

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
        self._temperature = None
        self._saved_metrics = dict()
        self.header = (
            'label',
            'layer_idx',
            'head_idxs',
            'perplexity',
            'average_energy_per_token_mj',
            'average_time_per_token_ms',
            'allocated_memory',
            'temperature'
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
    
    def clear_saved(self):
        self._saved_metrics = dict()
    
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
    
    def temperature(self, temperature):
        self._temperature = temperature
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
            self._average_time_per_token_ms, self._allocated_memory, self._temperature)
        return self
    
    def get_metrics(self):
        return [self.header] + list(self._saved_metrics.values())


_singleton = MetricsManager()

def metrics_manager():
    return _singleton


def capture_evaluation(func):
    
    class CaptureEvaluation:
        def __init__(self, instance):
            self.aggregate_loss = 0
            self.token_count = 0
            self.loss_token_count = 0
            self.execution_time_ms = 0
            self.func = func
            self.instance = instance
            self.energy_usage_mj = 0
            self.temperature = 0
            self.energy_recorder = EnergyRecorder()
        
        def capture(self, *args, **kwargs):
            tokens = args[0]
            input_ids = tokens['input_ids']
            attention_mask = tokens['attention_mask']
            self.token_count = attention_mask.sum().item() # only count unmasked tokens
            self.loss_token_count = attention_mask[:, 1:].sum().item()  # can't count first token, is not generated as a part of the prediction
            self.energy_recorder.start()
            predicted = self.func(self.instance, *args, **kwargs)
            energy_usage_mj, execution_time_ms, temperature = self.energy_recorder.end().get_metrics()
            self.energy_usage_mj += energy_usage_mj
            self.execution_time_ms += execution_time_ms
            self.temperature = temperature
            if self.token_count > 0:
                average_time_per_token_ms = self.execution_time_ms / self.token_count
                average_energy_per_token_mj = self.energy_usage_mj / self.token_count
            else:
                average_time_per_token_ms = 0
                average_energy_per_token_mj = 0
            logger.debug(
                f"execution_time={self.execution_time_ms / 1000:.2f} s, "
                f"energy_usage={self.energy_usage_mj:.2f} mj")
            logger.debug(
                f"average_time_per_token={average_time_per_token_ms:.2f} ms, "
                f"average_energy_per_token_mj={average_energy_per_token_mj / 1000:.2f} mj")
            
            token_losses = objective.cross_entropy(input_ids, attention_mask, predicted.logits)
            perplexity = objective.aggregate_perplexity(token_losses, self.loss_token_count)
            
            metrics_manager() \
                .execution_time_ms(self.execution_time_ms) \
                .total_energy(self.energy_usage_mj) \
                .average_time_per_token_ms(average_time_per_token_ms) \
                .average_energy_per_token_mj(average_energy_per_token_mj) \
                .allocated_memory(get_allocated_memory()) \
                .temperature(temperature) \
                .perplexity(perplexity)
            return predicted
    
    def wrapper(self, *args, **kwargs):
        return CaptureEvaluation(self).capture(*args, **kwargs)
    
    return wrapper
