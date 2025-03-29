import argparse
import csv
import gc
import logging
import os
import torch
from typing import Final

from llm_type import LLMType
from src.evaluation.head_pruner import HeadPruningTester
from src.metrics.metrics import metrics_manager

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

torch.manual_seed(633)


def run_tests(batch_size: int, evaluation_row_count: int,
              reverse_eval=False, model_path='meta-llama/Meta-Llama-3-8B',
              layer_range=None, runs_per_layer=1):
    logger.info(f'runs_per_layer={runs_per_layer}')
    
    transformer_type: Final[LLMType] = LLMType.LLAMA_3
    dataset: Final[tuple] = ("Salesforce/wikitext", 'wikitext-2-v1')
    tester = HeadPruningTester(dataset, batch_size, evaluation_row_count)
    tester.transformer_under_test(transformer_type, model_path, True)
    
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
    for run in range(runs_per_layer):
        clear_memory()
        tester.run_test(f'baseline-{run}')
    
    for layer in layers:
        
        # prune all heads, then ever other head
        tester \
            .transformer_under_test(transformer_type, model_path, True) \
            .prune_heads(layer, list(range(num_heads)))
        
        for run in range(runs_per_layer):
            logger.info(f"Evaluating all heads pruned for layer={layer}, run={run}")
            clear_memory()
            tester.run_test(f'pruned-{layer}-all-{run}')
        
        tester \
            .transformer_under_test(transformer_type, model_path, True) \
            .prune_heads(layer, list(range(0, num_heads, 2)))
        for run in range(runs_per_layer):
            logger.info(f"Evaluating every other head pruned for layer={layer}, run={run}")
            clear_memory()
            tester.run_test(f'pruned-{layer}-every-other-{run}')


def test_baseline(batch_size: int, evaluation_row_count: int,
                  model_path='meta-llama/Meta-Llama-3-8B', runs_per_layer=1):
    transformer_type: Final[LLMType] = LLMType.LLAMA_3
    dataset: Final[tuple] = ("Salesforce/wikitext", 'wikitext-2-v1')
    tester = HeadPruningTester(dataset, batch_size, evaluation_row_count)
    for run_idx in range(runs_per_layer):
        logger.info(f"Testing baseline, run={run_idx}")
        tester.transformer_under_test(transformer_type, model_path, True) \
            .run_test(f'baseline-{run_idx}')


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
    parser = argparse.ArgumentParser(
        description="Performs a battery of performance tests against Transformer models.")
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
    parser.add_argument(
        '--baseline',
        type=bool,
        default=False,
        help='Optional toggle to just test baseline. Defaults to False.'
    )
    parser.add_argument(
        '--runs-per-layer',
        type=int,
        default=1,
        help='Number of repeated runs per pruned layer (and baseline). Defaults to 1.'
    )
    
    if not torch.cuda.is_available():
        raise Exception("Cuda is currently the only supported platform.")
    
    args = parser.parse_args()
    if args.layer_range is not None:
        arg_range = args.layer_range.split('-')
        layer_range = (int(arg_range[0]), int(arg_range[1]))
    else:
        layer_range = None
    
    if args.baseline:
        test_baseline(batch_size=args.batch_size,
                      evaluation_row_count=args.eval_rows,
                      model_path=args.model_path, runs_per_layer=args.runs_per_layer)
        write_to_csv(args.output_path + '-baseline.csv')
    else:
        run_tests(batch_size=args.batch_size, evaluation_row_count=args.eval_rows,
                  model_path=args.model_path, layer_range=layer_range,
                  runs_per_layer=args.runs_per_layer)
        write_to_csv(args.output_path + '-forward.csv')
        metrics_manager().clear_saved()
        run_tests(batch_size=args.batch_size, evaluation_row_count=args.eval_rows,
                  model_path=args.model_path, layer_range=layer_range,
                  reverse_eval=True,
                  runs_per_layer=args.runs_per_layer)
        write_to_csv(args.output_path + '-reverse.csv')
