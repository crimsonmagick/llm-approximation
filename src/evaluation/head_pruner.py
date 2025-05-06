import time

from datasets import load_dataset
from large_language_model_service import get_model
from src.metrics.capture import instrument


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
    
    def batch_evaluate(self, test_case):
        num_batches = (self.evaluation_size_rows + self.batch_size - 1) // self.batch_size
        for batch_index in range(num_batches):
            start_idx = batch_index * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.evaluation_size_rows)
            batch = self.test_data.select(range(start_idx, end_idx))
            prompts = [example["text"] for example in batch]
            tokens = self.transformer.tokenize(prompts)
            
            # print(f"prompts={prompts}")
            start = time.time()
            self.transformer.predict(tokens)
            print(f"test_case={test_case}, batch_index={batch_index}, evaluation_timne={time.time() - start}\n")
    
    def num_layers(self):
        return self.transformer.model.config.num_hidden_layers
    
    def num_attention_heads(self):
        return self.transformer.model.config.num_attention_heads
    
    def num_key_value_groups(self):
        config = self.transformer.model.config
        return config.num_attention_heads // config.num_key_value_heads
    
    def transformer_under_test(self, model_type, model_path: str,
                               supports_pruning: bool, label: str, layer_idx: int = None,
                               head_idxs: list = None):
        del self.transformer
        self.supports_pruning = supports_pruning
        self.pruned_layer_idx = layer_idx
        self.pruned_head_idxs = head_idxs
        self.transformer = instrument(get_model(model_type, model_path, supports_pruning),
                                      label, layer_idx, head_idxs)
        if layer_idx is not None and head_idxs is not None and len(head_idxs) > 0:
            self.transformer.model.prune_heads({layer_idx: head_idxs})
        return self
