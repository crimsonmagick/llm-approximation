import gc
import logging

import torch

from src.metrics.energy.energy_recording import EnergyRecorder
from src.metrics.function import objective
from src.metrics.metrics_manager import EnergyCapture, PerplexityCapture, EnergyLogger, PerplexityLogger
from src.models.model_resolution import resolve_model, LLMType

logger = logging.getLogger(__name__)


class Evaluation:

    def __init__(self, *, model_path: str, scenario_name: str,
                 device, repetitions, llm_type: LLMType, label, warmup_repetitions=None,
                 pruning_strategy=None):
        self.scenario_name = scenario_name
        self.device = device
        self.repetitions = repetitions
        self.label = label
        self.llm_type = llm_type
        self.model_path = model_path
        self.warmup_repetitions = warmup_repetitions
        self.model = resolve_model(self.llm_type, self.model_path)
        self.pruning_strategy_name = type(pruning_strategy).__name__ if pruning_strategy else None
        self.pruning_metadata = pruning_strategy(self.model) if pruning_strategy else None

    def evaluate(self, batch):
        logger.info(
            f"{self.scenario_name}-{self.label}: Evaluating repetitions={self.repetitions}")
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        with torch.no_grad():
            for run_idx in range(self.repetitions):
                prediction = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return prediction

    @staticmethod
    def _clear_memory():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class EnergyEvaluation(Evaluation):

    def __init__(self, *, energy_logger: EnergyLogger, **kwargs):
        self.energy_logger = energy_logger
        super().__init__(**kwargs)

    def evaluate(self, batch):
        attention_mask = batch['attention_mask']
        token_count = attention_mask.sum().item()  # only count unmasked tokens
        token_count *= self.repetitions
        energy_recorder = EnergyRecorder()
        input_ids = batch['input_ids']
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

        captured_metric = EnergyCapture(
            label=self.label,
            average_energy_per_token_mj=average_energy_per_token_mj,
            average_time_per_token_ms=average_time_per_token_ms,
            pruning_strategy=self.pruning_strategy_name,
            pruning_metadata=self.pruning_metadata
        )
        self.energy_logger.log(captured_metric)

        return prediction


class PerplexityEvaluation(Evaluation):

    def __init__(self, *, perplexity_logger: PerplexityLogger, **kwargs):
        self.perplexity_logger = perplexity_logger
        super().__init__(**kwargs)

    def evaluate(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        loss_token_count = attention_mask[:, 1:].sum().item()

        prediction = super().evaluate(batch)

        token_losses = objective.cross_entropy(input_ids, attention_mask,
                                               prediction.logits)
        perplexity = objective.aggregate_perplexity(token_losses, loss_token_count)
        captured_metric = PerplexityCapture(
            label=self.label,
            pruning_strategy=self.pruning_strategy_name,
            pruning_metadata=self.pruning_metadata,
            perplexity=perplexity
        )
        self.perplexity_logger.log(captured_metric)

        return prediction
