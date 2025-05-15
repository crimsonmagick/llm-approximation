import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, Mock

from torch import nn
from torch.fx.experimental.unification.dispatch import namespace

from src.evaluation.evaluation import Evaluation, PrunedEvaluation
from src.evaluation.pruning import PruningStrategy
from src.models.model_resolution import LLMType, resolve_model


def pack(tokens_by_batch):
    batches = []
    for input_ids, attention_mask in tokens_by_batch:
        batches.append({"input_ids": input_ids,
                        "attention_mask": attention_mask})
    return batches


class EvaluationTests(unittest.TestCase):

    def test_base_evaluation(self):
        with ExitStack() as stack:
            mock_empty_cache = stack.enter_context(patch('torch.cuda.empty_cache'))
            mock_synchronize = stack.enter_context(patch('torch.cuda.synchronize'))

            mock_model = Mock(nn.Module)
            mock_resolve_model = stack.enter_context(patch('src.evaluation.evaluation.resolve_model'))
            mock_resolve_model.return_value = mock_model

            model_path = "test/path/model"
            scenario_name = "base-scenario"
            supports_attn_pruning = True
            device = "testdevice"
            repetitions = 5
            label = 'base-test-label'
            llm_type = LLMType.LLAMA_3
            under_test: Evaluation = Evaluation(model_path=model_path, scenario_name=scenario_name,
                                                supports_attn_pruning=supports_attn_pruning,
                                                device=device, repetitions=repetitions, llm_type=llm_type, label=label)

            tokens_by_batch = pack([
                ([1, 52, 123, 666], [1, 1, 1, 1]),
                ([9, 23, 45, 92], [1, 0, 0, 1])
            ])
            batch_count = len(tokens_by_batch)
            under_test.evaluate(tokens_by_batch)

            mock_resolve_model.assert_called_once()
            self.assertTrue(mock_empty_cache.call_count, batch_count)
            self.assertTrue(mock_synchronize.call_count, batch_count)

            for call_idx, tokens in enumerate(tokens_by_batch):
                expected_input_ids = tokens['input_ids']
                expected_attention_mask = tokens['attention_mask']

                kwargs = mock_model.call_args_list[call_idx][1]
                actual_input_ids = kwargs['input_ids']
                actual_attention_mask = kwargs['attention_mask']

                self.assertEqual(expected_input_ids, actual_input_ids)
                self.assertEqual(expected_attention_mask, actual_attention_mask)

    def test_pruned_evaluation(self):
        with ExitStack() as stack:
            mock_model = Mock(nn.Module)
            mock_pruned_model = Mock(nn.Module)
            mock_resolve_model = stack.enter_context(patch('src.evaluation.evaluation.resolve_model'))
            mock_resolve_model.return_value = mock_model

            model_path = "test/path/model"
            scenario_name = "base-scenario"
            supports_attn_pruning = True
            device = "testdevice"
            repetitions = 5
            label = 'base-test-label'
            llm_type = LLMType.LLAMA_3

            layer_idx = 7
            pruning_strategy = Mock(PruningStrategy)
            pruning_strategy.return_value = mock_pruned_model
            under_test: Evaluation = PrunedEvaluation(model_path=model_path, scenario_name=scenario_name,
                                                      supports_attn_pruning=supports_attn_pruning,
                                                      device=device, repetitions=repetitions, llm_type=llm_type,
                                                      label=label,
                                                      layer_idx=layer_idx, pruning_strategy=pruning_strategy)

            mock_resolve_model.assert_called_once()
            self.assertEqual(mock_model, pruning_strategy.call_args[0][0])
            self.assertEqual(mock_pruned_model, under_test.model)


if __name__ == '__main__':
    unittest.main()
