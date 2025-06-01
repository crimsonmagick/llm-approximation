import argparse

import torch

from src.evaluation.pruning import EveryOtherHead
from src.evaluation.scenario import EvaluationScenario
from src.models.model_resolution import LLMType

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Performs a battery of performance tests against Transformer models.")
    parser.add_argument(
        'scenario_name',
        type=str,
        help='Name of the evaluation scenario to be run.')
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
        '--layer-range',
        type=str,
        help='Range of layers to evaluate'
    )
    parser.add_argument(
        '--repetitions',
        type=int,
        default=1,
        help='Number of repeated runs per evaluation. Defaults to 1.'
    )
    parser.add_argument(
        '--warmup-repetitions',
        type=int,
        default=0,
        help='Number of warmup evaluation repetitions. Defaults to 0.'
    )

    if not torch.cuda.is_available():
        raise Exception("Cuda is currently the only supported platform.")

    args = parser.parse_args()
    if args.layer_range is not None:
        arg_range = args.layer_range.split('-')
        layer_range = (int(arg_range[0]), int(arg_range[1]))
    else:
        layer_range = None

    scenario = EvaluationScenario(model_path=args.model_path, llm_type=LLMType.LLAMA_3,
                                  evaluation_row_count=args.eval_rows, scenario_name=args.scenario_name,
                                  supports_attn_pruning=True, batch_size=args.batch_size)

    if args.warmup_repetitions:
        scenario.add_warmup_evaluation(repetitions=args.warmup_repetitions)
    for i in range(50):
        scenario.add_baseline_evaluation(capture_energy=True, repetitions=args.repetitions)

    # scenario \
    #     .add_baseline_evaluation(capture_perplexity=True) \
    #     .add_baseline_evaluation(capture_energy=True, repetitions=args.repetitions) \
    #     .add_pruned_evaluations(capture_perplexity=True, pruning_strategy=EveryOtherHead,
    #                                      evaluation_name="every_other_head_perplexity", layer_range=layer_range)
    for i in range(50):
        scenario.add_pruned_evaluations(capture_energy=True, pruning_strategy=EveryOtherHead,
                                        evaluation_name=f"every_other_head_energy_{i}", layer_range=layer_range,
                                        repetitions=args.repetitions)
    scenario.execute()
