import csv
import gc
import logging
import os
import torch
from typing import Final

from src.evaluation.evaluation_runner import HeadPruningTester
from src.models.model_resolution import LLMType
from src.metrics import metrics_manager

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

torch.manual_seed(633)


def run_tests(batch_size: int, evaluation_row_count: int,
              reverse_eval=False, model_path='meta-llama/Meta-Llama-3-8B',
              layer_range=None, runs_per_layer=1):
    logger.info(f'runs_per_layer={runs_per_layer}')
    
    transformer_type: Final[LLMType] = LLMType.LLAMA_3
    dataset: Final[tuple] = ("Salesforce/wikitext", 'wikitext-2-v1')
    tester = HeadPruningTester(model_path, dataset, batch_size, evaluation_row_count)
    
    tester.transformer_under_test(transformer_type, True, "baseline", suite='forward')
    
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
    
    layers = range(final_layer, first_layer - 1, -1) if reverse_eval else range(
        first_layer, final_layer + 1)
    
    num_gpus = torch.cuda.device_count()
    logger.info(f"num_gpus={num_gpus}")
    
    # run baseline first
    print("test baseline")
    for run_idx in range(runs_per_layer):
        clear_memory()
        tester.batch_evaluate(f'baseline-{run_idx}')
    print("tested baseline")
    
    for layer in layers:
        
        label = f'pruned-{layer}-all'
        # prune all heads, then ever other head
        tester.transformer_under_test(transformer_type, True, label, 'forward',
                                      layer_idx=layer, head_idxs=list(range(num_heads)))
        
        for run_idx in range(runs_per_layer):
            logger.info(f"Evaluating all heads pruned for layer={layer}, run={run_idx}")
            clear_memory()
            tester.batch_evaluate(f'pruned-{layer}-all-{run_idx}')
        
        tester.transformer_under_test(transformer_type, True, label, 'reverse',
                                      layer_idx=layer, head_idxs=list(range(0, num_heads, 2)))
        for run_idx in range(runs_per_layer):
            logger.info(f"Evaluating every other head pruned for layer={layer}, run={run_idx}")
            clear_memory()
            tester.batch_evaluate(f'pruned-{layer}-every-other-{run_idx}')


def evaluate_baseline(batch_size: int, evaluation_row_count: int,
                      model_path='meta-llama/Meta-Llama-3-8B', runs_per_layer=1):
    transformer_type: Final[LLMType] = LLMType.LLAMA_3
    dataset: Final[tuple] = ("Salesforce/wikitext", 'wikitext-2-v1')
    tester = HeadPruningTester(model_path, dataset, batch_size, evaluation_row_count)
    for run_idx in range(runs_per_layer):
        logger.info(f"Testing baseline, run={run_idx}")
        tester.transformer_under_test(transformer_type, True, "baseline", 'forward') \
            .batch_evaluate(f'baseline-{run_idx}')


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def write_to_csv(output_path: str, suite: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(metrics_manager.get_metrics(suite))


