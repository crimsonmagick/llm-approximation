import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, Mock

import torch
from torch import nn

from src.evaluation.evaluation import Evaluation, PrunedEvaluation, EnergyEvaluation
from src.evaluation.pruning import PruningStrategy
from src.metrics.metrics_manager import EnergyCapture
from src.models.model_resolution import LLMType, resolve_model
from tests.util.test_util_mixin import TestUtilMixin

SEQUENCE_COUNT = 6
SEQUENCE_LENGTH = 11
VOCABULARY_SIZE = 23


def pack(tokens_by_batch):
    batches = []
    for input_ids, attention_mask in tokens_by_batch:
        batches.append({'input_ids': torch.tensor(input_ids, dtype=torch.int),
                        'attention_mask': torch.tensor(attention_mask, dtype=torch.int)})
    return batches


class EvaluationTests(TestUtilMixin, unittest.TestCase):

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

            tokens_by_batch = pack([(
                [[1, 52, 123, 666], [9, 23, 45, 92]],
                [[1, 1, 1, 1], [1, 0, 0, 1]]
            )])
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

                self.assertTrue(torch.equal(expected_input_ids, actual_input_ids))
                self.assertTrue(torch.equal(expected_attention_mask, actual_attention_mask))

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
            pruning_strategy = Mock()
            pruning_strategy_instance = Mock()
            pruning_strategy.return_value = pruning_strategy_instance
            pruning_strategy_instance.return_value = mock_pruned_model
            under_test: Evaluation = PrunedEvaluation(model_path=model_path, scenario_name=scenario_name,
                                                      supports_attn_pruning=supports_attn_pruning,
                                                      device=device, repetitions=repetitions, llm_type=llm_type,
                                                      label=label,
                                                      layer_idx=layer_idx, pruning_strategy=pruning_strategy)

            mock_resolve_model.assert_called_once()
            pruning_strategy.assert_called_once()
            self.assertEqual(mock_model, pruning_strategy_instance.call_args[0][0])
            self.assertEqual(mock_pruned_model, under_test.model)

    def test_energy_evaluation(self):
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
            under_test: Evaluation = EnergyEvaluation(model_path=model_path, scenario_name=scenario_name,
                                                      supports_attn_pruning=supports_attn_pruning,
                                                      device=device, repetitions=repetitions, llm_type=llm_type,
                                                      label=label)

            tokens_by_batch = pack([(
                [[1, 52, 123, 666], [9, 23, 45, 92]],
                [[1, 1, 1, 1], [1, 0, 0, 1]]
            )])
            batch_count = len(tokens_by_batch)
            sequence_count = len(tokens_by_batch[0]['input_ids'])
            sequence_length = len(tokens_by_batch[0]['input_ids'][0])
            vocabulary_size = 23

            expected_layer_idx = None
            expected_logits = self.rand_logits(batch_count, sequence_length, vocabulary_size)

            mock_energy_recorder = stack.enter_context(patch("src.evaluation.evaluation.EnergyRecorder"))
            energy_recorder_instance = mock_energy_recorder.return_value
            energy_recorder_instance.start.return_value = None
            expected_time_ms = 240
            expected_energy_mj = 300
            expected_temperature_c = 52
            energy_recorder_instance.end.return_value.get_metrics.return_value \
                = (expected_energy_mj, expected_time_ms, expected_temperature_c)

            stubbed_prediction = SimpleNamespace(logits=expected_logits)
            mock_model.return_value = stubbed_prediction

            mock_accept_energy = stack.enter_context(patch("src.evaluation.evaluation.metrics_manager.accept_energy"))

            predictions = under_test.evaluate(tokens_by_batch)

            mock_resolve_model.assert_called_once()
            self.assertTrue(mock_empty_cache.call_count, batch_count)
            self.assertTrue(mock_synchronize.call_count, batch_count)

            torch.equal(expected_logits, predictions[0].logits)

            for call_idx, tokens in enumerate(tokens_by_batch):
                expected_input_ids = tokens['input_ids']
                expected_attention_mask = tokens['attention_mask']

                kwargs = mock_model.call_args_list[call_idx][1]
                actual_input_ids = kwargs['input_ids']
                actual_attention_mask = kwargs['attention_mask']

                self.assertTrue(torch.equal(expected_input_ids, actual_input_ids))
                self.assertTrue(torch.equal(expected_attention_mask, actual_attention_mask))
                args, kwargs = mock_accept_energy.call_args
                suite: str = kwargs['scenario']
                metrics: EnergyCapture = args[0]

                token_count = expected_attention_mask.sum()

                self.assertEqual(scenario_name, suite)
                self.assertEqual(label, metrics.label)
                self.assertEqual(expected_layer_idx, metrics.layer_idx)
                self.assertEqual(expected_energy_mj / token_count, metrics.average_energy_per_token_mj)
                self.assertEqual(expected_time_ms / token_count, metrics.average_time_per_token_ms)


if __name__ == '__main__':
    unittest.main()
