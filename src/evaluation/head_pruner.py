import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.utils import PaddingStrategy

from model_resolution import get_model
from src.metrics.capture import instrument


class HeadPruningTester:
    
    def __init__(self, model_path: str, dataset_path: tuple, batch_size: int,
                 evaluation_row_count: int):
        self.model_path = model_path
        self.transformer = None
        self.batch_size = batch_size
        self.test_data = load_dataset(*dataset_path)["test"].filter(
            lambda ex: ex["text"] and ex["text"].strip() != ""
        )
        self.evaluation_size_rows = evaluation_row_count
        self.supports_pruning = False
        self.pruned_layer_idx = None
        self.pruned_head_idxs = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.device = 'cuda'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def batch_evaluate(self, test_case):
        num_batches = (self.evaluation_size_rows + self.batch_size - 1) // self.batch_size
        for batch_index in range(num_batches):
            start_idx = batch_index * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.evaluation_size_rows)
            batch = self.test_data.select(range(start_idx, end_idx))
            prompts = [example["text"] for example in batch]
            tokens = self.tokenize(prompts)
            input_ids = tokens['input_ids']
            attention_mask = tokens['attention_mask']
            
            # print(f"prompts={prompts}")
            start = time.time()
            with torch.no_grad():
                self.transformer(input_ids=input_ids, attention_mask=attention_mask)
            print(f"test_case={test_case}, batch_index={batch_index}, evaluation_time={time.time() - start}\n")
    
    def num_layers(self):
        return self.transformer.model.config.num_hidden_layers
    
    def num_attention_heads(self):
        return self.transformer.config.num_attention_heads
    
    def num_key_value_groups(self):
        config = self.transformer.model.config
        return config.num_attention_heads // config.num_key_value_heads
    
    def transformer_under_test(self, model_type, supports_pruning: bool, label: str,
                               layer_idx: int = None, head_idxs: list = None):
        del self.transformer
        self.supports_pruning = supports_pruning
        self.pruned_layer_idx = layer_idx
        self.pruned_head_idxs = head_idxs
        self.transformer = instrument(get_model(model_type, self.model_path, supports_pruning),
                                      label, layer_idx, head_idxs)
        if layer_idx is not None and head_idxs is not None and len(head_idxs) > 0:
            self.transformer.prune_heads({layer_idx: head_idxs})
        return self
    
    def tokenize(self, prompt):
        return self.tokenizer(prompt, return_tensors='pt', padding=PaddingStrategy.LONGEST, padding_side='left',
                              truncation=TruncationStrategy.LONGEST_FIRST, max_length=512).to(self.device)
