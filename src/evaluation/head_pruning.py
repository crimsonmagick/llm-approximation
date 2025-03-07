import argparse
import csv
import gc
import logging
import os
import string
from typing import Final

import torch
from datasets import load_dataset

from large_language_model_service import get_model
from llm_type import LLMType
from src.metrics.metrics import metrics_manager

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

torch.manual_seed(633)


class HeadPruningTester:
    
    def __init__(self, dataset_path: tuple, batch_size: int, evaluation_row_count: int):
        self.transformer = None
        self.batch_size = batch_size
        self.test_data = load_dataset(*dataset_path)["test"].filter(
            lambda ex: ex["text"] and ex["text"].strip() != ""
        )
        self.evaluation_size_rows = evaluation_row_count
        self.supports_pruning = False
        self.pruned_layer_idx = None
        self.pruned_head_idxs = None
    
    def _batch_evaluate(self):
        num_batches = (self.evaluation_size_rows + self.batch_size - 1) // self.batch_size
        for batch_index in range(num_batches):
            start_idx = batch_index * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.evaluation_size_rows)
            batch = self.test_data.select(range(start_idx, end_idx))
            prompts = [example["text"] for example in batch]
            tokens = self.transformer.tokenize(prompts)
            self.transformer.evaluate(tokens)
            self.transformer.per_token_losses(tokens)
    
    def prune_heads(self, layer_idx, head_idxs):
        self.pruned_head_idxs = head_idxs
        self.pruned_layer_idx = layer_idx
        self.transformer.model.prune_heads({layer_idx: head_idxs})
        metrics_manager().layer_idx(layer_idx).head_idxs(head_idxs)
        return self
    
    def run_test(self, test_case):
        self._batch_evaluate()
        metrics_manager().save_metrics(test_case)
        return self
    
    def num_layers(self):
        return self.transformer.model.config.num_hidden_layers
    
    def num_attention_heads(self):
        return self.transformer.model.config.num_attention_heads
    
    def num_key_value_groups(self):
        config = self.transformer.model.config
        return config.num_attention_heads // config.num_key_value_heads
    
    def transformer_under_test(self, model_type, model_path: string, supports_pruning: bool):
        del self.transformer
        self.supports_pruning = supports_pruning
        self.pruned_layer_idx = None
        self.pruned_head_idxs = None
        self.transformer = get_model(model_type, model_path, supports_pruning)
        metrics_manager().clear()
        return self


def run_tests(batch_size: int, evaluation_row_count: int, reverse_eval=False,
              model_path='meta-llama/Meta-Llama-3-8B', layer_range=None):
    transformer_type: Final[LLMType] = LLMType.LLAMA_3
    dataset: Final[tuple] = ("Salesforce/wikitext", 'wikitext-2-v1')
    tester = HeadPruningTester(dataset, batch_size, evaluation_row_count)
    tester.transformer_under_test(transformer_type, model_path, True) \
        .run_test('baseline')
    
    num_heads = tester.num_attention_heads()
    if layer_range is not None:
        first_layer, final_layer = layer_range
    else:
        first_layer = 0
        final_layer = tester.num_layers() - 1
    
    if (final_layer <= first_layer
            or final_layer - first_layer > tester.num_layers()
            or final_layer < 0 or first_layer < 0):
        error_message = f"Invalid layer range specified: {layer_range}, model layer_range={tester.num_layers()}"
        raise Exception(error_message)
    
    layers = range(final_layer, first_layer - 1, -1) if reverse_eval else range(first_layer, final_layer + 1)
    for layer in layers:
        # prune all heads, then ever other head
        logger.info(f"Evaluating all heads pruned for layer={layer}")
        clear_memory()
        tester = HeadPruningTester(dataset, batch_size, evaluation_row_count)
        tester.transformer_under_test(transformer_type, model_path,
                                      True) \
            .prune_heads(layer, list(range(num_heads))) \
            .run_test(f'pruned-{layer}-all')
        
        logger.info(f"Evaluating every other head pruned for layer={layer}")
        clear_memory()
        tester.transformer_under_test(transformer_type, model_path, True) \
            .prune_heads(layer, list(range(0, num_heads, 2))) \
            .run_test(f'pruned-{layer}-every-other')


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def write_to_csv(output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(metrics_manager().get_metrics())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Performs a battery of performance tests against Transformer models.")
    parser.add_argument(
        '--output-path',
        type=str,
        default='./transformer_metrics.csv',
        help='Optional output path. Defaults to the "transformer_metrics.csv".')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Optional batch size for dataset evaluation. Defaults to 1.'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='meta-llama/Meta-Llama-3-8B',
        help='Path param for the model under test.  Defaults to meta-llama/Meta-Llama-3-8B'
    )
    parser.add_argument(
        '--eval-rows',
        type=int,
        default=1,
        help='Number of rows of the dataset to evaluate. Defaults to 1.'
    )
    parser.add_argument(
        '--reverse-eval',
        type=bool,
        default=False,
        help='Optional toggle for testing layers in reverse order. Defaults to False.'
    )
    parser.add_argument(
        '--layer-range',
        type=str,
        help='Range of layers to evaluate'
    )
    args = parser.parse_args()
    if args.layer_range is not None:
        arg_range = args.layer_range.split('-')
        layer_range = (int(arg_range[0]), int(arg_range[1]))
    else:
        layer_range = None
    
    run_tests(batch_size=args.batch_size, evaluation_row_count=args.eval_rows,
              reverse_eval=args.reverse_eval, model_path=args.model_path, layer_range=layer_range)
    write_to_csv(args.output_path)
