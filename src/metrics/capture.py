import logging

from .energy.energy_recording import EnergyRecorder
from .function import objective
from .memory import get_allocated_memory
from . import metrics_manager

logger = logging.getLogger(__name__)


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
            self.token_count = attention_mask.sum().item()  # only count unmasked tokens
            self.loss_token_count = attention_mask[:,
                                    1:].sum().item()  # can't count first token, is not generated as a part of the prediction
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
            
            metrics_manager \
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
