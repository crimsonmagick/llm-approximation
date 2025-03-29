import string
import time

import torch
from datasets import load_dataset
from large_language_model_service import get_model
from src.metrics.metrics import metrics_manager


class HeadPruningTester:

  def __init__(self, dataset_path: tuple, batch_size: int,
      evaluation_row_count: int):
    self.transformer = None
    self.batch_size = batch_size
    self.test_data = load_dataset(*dataset_path)["test"].filter(
        lambda ex: ex["text"] and ex["text"].strip() != ""
    )
    self.evaluation_size_rows = evaluation_row_count
    self.supports_pruning = False
    self.pruned_layer_idx = None
    self.pruned_head_idxs = None

  def _batch_evaluate(self, test_case):
    num_batches = (
                      self.evaluation_size_rows + self.batch_size - 1) // self.batch_size
    for batch_index in range(num_batches):
      start_idx = batch_index * self.batch_size
      end_idx = min(start_idx + self.batch_size, self.evaluation_size_rows)
      batch = self.test_data.select(range(start_idx, end_idx))
      prompts = [example["text"] for example in batch]
      print(f"PROMPT={prompts}")
      tokens = self.transformer.tokenize(prompts)
      start = time.time()
      generated, _ = self.transformer.evaluate(tokens)
      print(f"Time for evaluation: {time.time() - start}\n")
      detok = self.transformer.detokenize(generated)
      print(f"generated={detok}")

      self.transformer.per_token_losses(tokens)
      metrics_manager().save_metrics(test_case + f'batch{batch_index}')

  def prune_heads(self, layer_idx, head_idxs):
    self.pruned_head_idxs = head_idxs
    self.pruned_layer_idx = layer_idx
    self.transformer.model.prune_heads({layer_idx: head_idxs})
    metrics_manager().layer_idx(layer_idx).head_idxs(head_idxs)
    return self

  def run_test(self, test_case):
    self._batch_evaluate(test_case)
    return self

  def num_layers(self):
    return self.transformer.model.config.num_hidden_layers

  def num_attention_heads(self):
    return self.transformer.model.config.num_attention_heads

  def num_key_value_groups(self):
    config = self.transformer.model.config
    return config.num_attention_heads // config.num_key_value_heads

  def transformer_under_test(self, model_type, model_path: string,
      supports_pruning: bool):
    del self.transformer
    self.supports_pruning = supports_pruning
    self.pruned_layer_idx = None
    self.pruned_head_idxs = None
    self.transformer = get_model(model_type, model_path, supports_pruning)
    metrics_manager().clear()
    return self

