import argparse

import torch

from src.evaluation.evaluation_harness import evaluate_baseline, write_to_csv, run_tests

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
        evaluate_baseline(batch_size=args.batch_size,
                          evaluation_row_count=args.eval_rows,
                          model_path=args.model_path, runs_per_layer=args.runs_per_layer)
        write_to_csv(args.output_path + '-baseline.csv', 'forward')
    else:
        run_tests(batch_size=args.batch_size, evaluation_row_count=args.eval_rows,
                  model_path=args.model_path, layer_range=layer_range,
                  runs_per_layer=args.runs_per_layer)
        write_to_csv(args.output_path + '-forward.csv', 'forward')
        run_tests(batch_size=args.batch_size, evaluation_row_count=args.eval_rows,
                  model_path=args.model_path, layer_range=layer_range,
                  reverse_eval=True,
                  runs_per_layer=args.runs_per_layer)
        write_to_csv(args.output_path + '-reverse.csv', 'reverse')
