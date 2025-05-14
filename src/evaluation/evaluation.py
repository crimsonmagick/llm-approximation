import gc
import logging
from typing import List

import torch
from datasets import load_dataset
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.utils import PaddingStrategy

from src.metrics import metrics_manager
from src.metrics.energy.energy_recording import EnergyRecorder
from src.metrics.function import objective
from src.metrics.metrics_manager import EnergyCapture, PerplexityMetricsCapture
from src.models.model_resolution import resolve_model, LLMType

logger = logging.getLogger(__name__)


class Evaluation:
    
    def __init__(self, *, model_path: str, dataset: tuple,
                 evaluation_row_count: int, scenario_name: str,
                 supports_attn_pruning: bool = False, device,
                 repetitions, batch_size, llm_type: LLMType, tokenizer, label):
        self.model_path = model_path
        self.test_data = load_dataset(*dataset)["test"].filter(
            lambda ex: ex["text"] and len(ex["text"].strip()) > 500
        )
        self.evaluation_size_rows = evaluation_row_count
        self.scenario_name = scenario_name
        self.supports_attn_pruning = supports_attn_pruning
        self.device = device
        self.repetitions = repetitions
        self.batch_size = batch_size
        self.llm_type = llm_type
        self.tokenizer = tokenizer
        self.label = label
        self.model = self._get_model()
    
    def evaluate(self, *, input_ids, attention_mask):
        num_batches = (self.evaluation_size_rows + self.batch_size - 1) // self.batch_size
        for run_idx in range(self.repetitions):
            self._clear_memory()
            for batch_index in range(num_batches):
                logger.info(
                    f"{self.scenario_name}-{self.label}: Evaluating run={run_idx}/{self.repetitions}, batch={batch_index + 1}/{num_batches}")
                start_idx = batch_index * self.batch_size
                end_idx = min(start_idx + self.batch_size, self.evaluation_size_rows)
                batch = self.test_data.select(range(start_idx, end_idx))
                prompts = [example["text"] for example in batch]
                tokens = self._tokenize(prompts)
                input_ids = tokens['input_ids']
                attention_mask = tokens['attention_mask']
                
                with torch.no_grad():
                    return self.model(input_ids=input_ids, attention_mask=attention_mask)
    
    def _tokenize(self, prompt):
        return self.tokenizer(prompt, return_tensors='pt', padding=PaddingStrategy.LONGEST, padding_side='left',
                              truncation=TruncationStrategy.LONGEST_FIRST, max_length=512).to(self.device)
    
    def _get_model(self):
        return resolve_model(self.llm_type, self.model_path,
                             self.supports_attn_pruning)
    
    @staticmethod
    def _clear_memory():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class PrunedEvaluation(Evaluation):
    
    def __init__(self, layer_idx: int, pruning_strategy, **kwargs):
        self.layer_idx = layer_idx
        self.pruning_strategy = pruning_strategy
        super.__init__(**kwargs)
    
    def _get_model(self):
        model = super()._get_model()
        return self.pruning_strategy(model)


class EnergyEvaluation(Evaluation):
    
    def evaluate(self, **kwargs):
        attention_mask = kwargs['attention_mask']
        token_count = attention_mask.sum().item()  # only count unmasked tokens
        energy_recorder = EnergyRecorder()
        energy_recorder.start()
        predicted = super().evaluate(**kwargs)
        energy_usage_mj, execution_time_ms, temperature = energy_recorder.end().get_energy_metrics()
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
            layer_idx=self.layer_idx,
            head_idxs=self.head_idxs
        )
        metrics_manager.accept_energy(captured_metrics, suite=self.scenario_name)
        
        return predicted


class PerplexityEvaluation(Evaluation):
    
    def evaluate(self, **kwargs):
        input_ids = kwargs['input_ids']
        attention_mask = kwargs['attention_mask']
        loss_token_count = attention_mask[:, 1:].sum().item()
        
        predicted = super().evaluate(**kwargs)
        
        token_losses = objective.cross_entropy(input_ids, attention_mask, predicted.logits)
        perplexity = objective.aggregate_perplexity(token_losses, loss_token_count)
        captured_metrics = PerplexityMetricsCapture(
            label=self.label,
            perplexity=perplexity,
            layer_idx=self.layer_idx,
            head_idxs=self.head_idxs
        )
        metrics_manager.accept_perplexity(captured_metrics, suite=self.scenario_name)
        
        return predicted
