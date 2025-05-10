import csv
import gc
import logging
import os

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.utils import PaddingStrategy

from src.metrics import metrics_manager
from src.models.model_resolution import get_model
from src.metrics.capture import instrument

logger = logging.getLogger(__name__)


# TODO might as well combine with EvaluationScenario. Consider limiting the scope of this class.
class EvaluationRunner:
    
    def __init__(self, model_path: str, dataset_path: tuple, evaluation_row_count: int = 10, *,
                 scenario_name: str = 'default', layer_range: tuple = None, supports_attn_pruning: bool = False,
                 device='cuda', num_heads, llm_type, results_path: str):
        self.model_path = model_path
        self.test_data = load_dataset(*dataset_path)["test"].filter(
            lambda ex: ex["text"] and len(ex["text"].strip()) > 500
        )
        self.evaluation_size_rows = evaluation_row_count
        self.scenario_name = scenario_name
        self.layer_range = layer_range
        self.supports_attn_pruning = supports_attn_pruning
        self.num_heads = num_heads
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.device = device
        self.model_type = llm_type
        self.results_path = results_path
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def evaluate(self, reverse_eval=False, batch_size=5, num_runs: int = 1, baseline_only=False):
        first_layer, final_layer = self.layer_range
        layers = range(final_layer, first_layer - 1, -1) if reverse_eval else range(
            first_layer, final_layer + 1)
        model = instrument(get_model(self.model_type, self.model_path, self.supports_attn_pruning),
                           'baseline', None, None, self.scenario_name)
        num_baseline_runs = num_runs if baseline_only else 2 * num_runs
        label = "warmup" if baseline_only else "baseline"
        self._evaluate_model(model, batch_size, num_baseline_runs, label)
        
        if self.supports_attn_pruning and not baseline_only:
            for layer_idx in layers:
                # prune all heads
                head_idxs = list(range(self.num_heads))
                label = f'pruned-{layer_idx}-all'
                del model
                model = instrument(get_model(self.model_type, self.model_path, self.supports_attn_pruning),
                                   label, layer_idx, head_idxs, self.scenario_name)
                model.prune_heads({layer_idx: head_idxs})
                
                logger.info(f"Evaluating all heads pruned for layer={layer_idx}")
                self._evaluate_model(model, batch_size, num_runs, label)
                
                # prune every other head
                head_idxs = list(range(0, self.num_heads, 2))
                label = f'pruned-{layer_idx}-everyother'
                del model
                model = instrument(get_model(self.model_type, self.model_path, self.supports_attn_pruning),
                                   label, layer_idx, head_idxs, self.scenario_name)
                model.prune_heads({layer_idx: head_idxs})
                
                logger.info(f"Evaluating every other head pruned for layer={layer_idx}")
                self._evaluate_model(model, batch_size, num_runs, label)
        self._write_to_csv(self.results_path)
    
    def _evaluate_model(self, model, batch_size, number_of_runs, label):
        num_batches = (self.evaluation_size_rows + batch_size - 1) // batch_size
        for run_idx in range(number_of_runs):
            self._clear_memory()
            for batch_index in range(num_batches):
                logger.info(
                    f"{self.scenario_name}-{label}: Evaluating run={run_idx}/{number_of_runs}, batch={batch_index + 1}/{num_batches}")
                start_idx = batch_index * batch_size
                end_idx = min(start_idx + batch_size, self.evaluation_size_rows)
                batch = self.test_data.select(range(start_idx, end_idx))
                prompts = [example["text"] for example in batch]
                tokens = self._tokenize(prompts)
                input_ids = tokens['input_ids']
                attention_mask = tokens['attention_mask']
                
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
    
    def _tokenize(self, prompt):
        return self.tokenizer(prompt, return_tensors='pt', padding=PaddingStrategy.LONGEST, padding_side='left',
                              truncation=TruncationStrategy.LONGEST_FIRST, max_length=512).to(self.device)
    
    def _write_to_csv(self, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as file:
            writer = csv.writer(file)
            writer.writerows(metrics_manager.get_metrics(self.scenario_name))
    
    @staticmethod
    def _clear_memory():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
