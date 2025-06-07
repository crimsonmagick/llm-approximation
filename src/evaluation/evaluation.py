import gc
import logging

import torch

from src.metrics import metrics_manager
from src.metrics.energy.energy_recording import EnergyRecorder
from src.metrics.function import objective
from src.metrics.metrics_manager import EnergyCapture, PerplexityCapture
from src.models.model_resolution import resolve_model, LLMType

logger = logging.getLogger(__name__)


class Evaluation:

    def __init__(self, *, model_path: str, scenario_name: str,
                 supports_attn_pruning: bool, device, repetitions,
                 llm_type: LLMType, label, warmup_repetitions=None, pruning_strategy=None):
        self.scenario_name = scenario_name
        self.device = device
        self.repetitions = repetitions
        self.label = label
        self.llm_type = llm_type
        self.model_path = model_path
        self.supports_attn_pruning = supports_attn_pruning
        self.model = self._get_model()
        self.warmup_repetitions = warmup_repetitions
        self.pruning_strategy = pruning_strategy
        self.pruning_metadata = None

    def evaluate(self, tokens_by_batch):
        logger.info(
            f"{self.scenario_name}-{self.label}: Evaluating repetitions={self.repetitions}")
        input_ids = tokens_by_batch[0]['input_ids']
        attention_mask = tokens_by_batch[0]['attention_mask']
        with torch.no_grad():
            for run_idx in range(self.repetitions):
                prediction = self.model(input_ids=input_ids, attention_mask=attention_mask)
        self._clear_memory()
        return [prediction]

    def _get_model(self):
        model = resolve_model(self.llm_type, self.model_path,
                              self.supports_attn_pruning)
        if self.pruning_strategy:
            self.pruning_strategy(model)
        return model

    @staticmethod
    def _clear_memory():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class EnergyEvaluation(Evaluation):

    def evaluate(self, tokens_by_batch):
        token_count = 0
        for tokens in tokens_by_batch:
            attention_mask = tokens['attention_mask']
            token_count += attention_mask.sum().item()  # only count unmasked tokens
        token_count *= self.repetitions
        energy_recorder = EnergyRecorder()
        input_ids = tokens_by_batch[0]['input_ids']
        attention_mask = tokens_by_batch[0]['attention_mask']
        self._clear_memory()
        with torch.no_grad():
            if self.warmup_repetitions:
                logger.info(
                    f"{self.scenario_name}-{self.label}: Evaluating warmup repetitions, warmup_repetitions={self.warmup_repetitions}")
                for run_idx in range(self.warmup_repetitions):
                    prediction = self.model(input_ids=input_ids, attention_mask=attention_mask)

            logger.info(
                f"{self.scenario_name}-{self.label}: Evaluating repetitions={self.repetitions}")
            energy_recorder.start()
            for run_idx in range(self.repetitions):
                prediction = self.model(input_ids=input_ids, attention_mask=attention_mask)
            energy_usage_mj, execution_time_ms = energy_recorder.end().get_metrics()

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

        captured_metrics = EnergyCapture(
            label=self.label,
            average_energy_per_token_mj=average_energy_per_token_mj,
            average_time_per_token_ms=average_time_per_token_ms,
            layer_idx=self.layer_idx if hasattr(self, 'layer_idx') else None,
            head_idxs=self.head_idxs if hasattr(self, 'head_idxs') else None
        )
        metrics_manager.accept_energy(captured_metrics, scenario=self.scenario_name)

        return [prediction]


class PerplexityEvaluation(Evaluation):

    def evaluate(self, tokens_by_batch):
        loss_token_count = 0
        input_ids = []
        attention_masks = []
        for tokens in tokens_by_batch:
            input_ids.append(tokens['input_ids'])
            attention_masks.append(tokens['attention_mask'])
            loss_token_count += tokens['attention_mask'][:, 1:].sum().item()

        prediction = super().evaluate(tokens_by_batch)
        prediction_logits = [batch.logits for batch in prediction]

        token_losses = objective.cross_entropy(torch.cat(input_ids), torch.cat(attention_masks),
                                               torch.cat(prediction_logits))
        perplexity = objective.aggregate_perplexity(token_losses, loss_token_count)
        captured_metrics = PerplexityCapture(
            label=self.label,
            perplexity=perplexity,
            layer_idx=self.layer_idx if hasattr(self, 'layer_idx') else None,
            head_idxs=self.head_idxs if hasattr(self, 'head_idxs') else None
        )
        metrics_manager.accept_perplexity(captured_metrics, scenario=self.scenario_name)

        return prediction
