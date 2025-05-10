import argparse

import torch

from src.evaluation.evaluation_scenario import EvaluationScenario
from src.models.model_resolution import LLMType

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
        EvaluationScenario(model_path=args.model_path, llm_type=LLMType.LLAMA_3,
                           evaluation_row_count=args.eval_rows, scenario_name='baselineonly',
                           supports_attn_pruning=True) \
            .runner(results_path=args.output_path + '-baseline.csv') \
            .evaluate(batch_size=args.batch_size, num_runs=args.runs_per_layer, baseline_only=True)
    else:
        EvaluationScenario(model_path=args.model_path, llm_type=LLMType.LLAMA_3,
                           evaluation_row_count=args.eval_rows, scenario_name='baselineonly',
                           supports_attn_pruning=True) \
            .runner(results_path=args.output_path + '-baseline.csv') \
            .evaluate(batch_size=args.batch_size, num_runs=args.runs_per_layer, baseline_only=True)
        # EvaluationScenario(model_path=args.model_path, llm_type=LLMType.LLAMA_3,
        #                    evaluation_row_count=args.eval_rows, scenario_name='forward',
        #                    supports_attn_pruning=True, layer_range=layer_range) \
        #     .runner(results_path=args.output_path + '-forward.csv') \
        #     .evaluate(batch_size=args.batch_size, num_runs=args.runs_per_layer)
        EvaluationScenario(model_path=args.model_path, llm_type=LLMType.LLAMA_3,
                           evaluation_row_count=args.eval_rows, scenario_name='reverse',
                           supports_attn_pruning=True, layer_range=layer_range) \
            .runner(results_path=args.output_path + '-reverse.csv') \
            .evaluate(batch_size=args.batch_size, num_runs=args.runs_per_layer)
