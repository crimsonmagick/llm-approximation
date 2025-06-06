import logging

from torch import nn

from src.metrics.energy.energy_recording import EnergyRecorder
from src.metrics.function import objective
from src.metrics.memory import get_allocated_memory
from src.metrics import metrics_manager
from src.metrics.metrics_manager import MetricsCapture

logger = logging.getLogger(__name__)


def instrument(model, label, layer_idx, head_idxs, suite):
    if isinstance(model, nn.Module):
        model.forward = _capture_evaluation(model.forward, label, layer_idx, head_idxs, suite)
    else:
        raise TypeError("Only derivatives of nn.Module are supported")
    return model


def _capture_evaluation(func, label, layer_idx, head_idxs, suite):
    class CaptureEvaluation:
        def __init__(self):
            self.func = func
            self.label = label
            self.invocation_count = 0
        
        def capture(self, *args, **kwargs):
            input_ids = kwargs['input_ids']
            attention_mask = kwargs['attention_mask']
            token_count = attention_mask.sum().item()  # only count unmasked tokens
            # can't count first token, is not generated as a part of the prediction
            loss_token_count = attention_mask[:, 1:].sum().item()
            energy_recorder = EnergyRecorder()
            energy_recorder.start()
            predicted = self.func(*args, **kwargs)
            energy_usage_mj, execution_time_ms, temperature = energy_recorder.end().get_metrics()
            if token_count > 0:
                average_time_per_token_ms = execution_time_ms / token_count
                average_energy_per_token_mj = energy_usage_mj / token_count
            else:
                average_time_per_token_ms = 0
                average_energy_per_token_mj = 0
            logger.info(
                f"execution_time={execution_time_ms:.2f} ms, "
                f"energy_usage={energy_usage_mj:.2f} mj")
            logger.info(
                f"average_time_per_token={average_time_per_token_ms:.2f} ms, "
                f"average_energy_per_token_mj={average_energy_per_token_mj :.2f} mj")
            
            token_losses = objective.cross_entropy(input_ids, attention_mask, predicted.logits)
            perplexity = objective.aggregate_perplexity(token_losses, loss_token_count)
            allocated_memory = get_allocated_memory()
            
            captured_metrics = MetricsCapture(
                allocated_memory=allocated_memory,
                label=f'{label}-{self.invocation_count}',
                perplexity=perplexity,
                average_energy_per_token_mj=average_energy_per_token_mj,
                average_time_per_token_ms=average_time_per_token_ms,
                temperature=temperature,
                layer_idx=layer_idx,
                head_idxs=head_idxs
            )
            metrics_manager.accept(captured_metrics, suite=suite)
            
            self.invocation_count += 1
            return predicted
    
    return CaptureEvaluation().capture
