import unittest
from contextlib import ExitStack
from functools import reduce
from types import SimpleNamespace
from typing import List, Type
from unittest.mock import patch, MagicMock, Mock

import torch
from datasets import DatasetDict, Dataset
from huggingface_hub import repo_info
from torch import nn, Tensor

from src.evaluation.pruning import PruningStrategy
from src.evaluation.scenario import EvaluationScenario
from src.models.model_resolution import LLMType

EOS_TOKEN = 5


class StubEvaluation:
    _created = []

    def __init__(self, *args, **kwargs):
        self.init_args = (args, kwargs)
        self.call_args_list = []
        type(self)._created.append(self)

    def evaluate(self, tokens_by_batch):
        self.call_args_list.append(tokens_by_batch)

    @staticmethod
    def get_creation_history():
        return StubEvaluation._created

    @staticmethod
    def clear_creation_history():
        StubEvaluation._created = []


class StubPrunedEvaluation(StubEvaluation):
    pass


class StubPerplexityEvaluation(StubEvaluation):
    pass


class StubEnergyEvaluation(StubEvaluation):
    pass


EVALUATION_CLASSES = [StubEvaluation, StubPrunedEvaluation, StubPerplexityEvaluation, StubEnergyEvaluation]


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        StubEvaluation.clear_creation_history()
        self._patch_stack = ExitStack()
        self.addCleanup(self._patch_stack.close)

        self.model_path = "test/path/model"
        self.supports_attn_pruning = True
        self.batch_size = 10
        self.llm_type = LLMType.LLAMA_3
        self.evaluation_row_count = 20
        self.rng_seed = 711
        self.scenario_name = "base_scenario"
        self.dataset = ("test/path/dataset", "dataset_name")
        self.device = "totes_real_device0"

        self.mock_manual_seed = self._patch_stack.enter_context(
            patch('src.evaluation.scenario.torch.manual_seed')
        )
        mock_load_dataset = self._patch_stack.enter_context(patch('datasets.load_dataset'))
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = ['z' * 501, "short", None]
        mock_load_dataset.return_value = mock_dataset

        self.num_attention_heads = 40
        self.num_hidden_layers = 32
        self.mock_config = SimpleNamespace(num_attention_heads=self.num_attention_heads,
                                           num_hidden_layers=self.num_hidden_layers)
        self.patch_config = self._patch_stack.enter_context(
            patch('src.evaluation.scenario.AutoConfig.from_pretrained')
        )
        self.patch_config.return_value = self.mock_config

        mock_tokenizer_from_pretrained = patch('src.evaluation.scenario.AutoTokenizer.from_pretrained')
        self.mock_tokenizer_from_pretrained = self._patch_stack.enter_context(mock_tokenizer_from_pretrained)
        self.mock_tokenizer = MagicMock(eos_token=EOS_TOKEN, pad_token_id=None, pad_token=None)
        self.mock_tokenizer_from_pretrained.return_value = self.mock_tokenizer

        self.patched_resolve_model = self._patch_stack.enter_context(patch('src.evaluation.evaluation.resolve_model'))
        self.mock_model = Mock(nn.Module)
        self.patched_resolve_model.return_value = self.mock_model

        self.patched_evaluation = self._patch_stack.enter_context(
            patch('src.evaluation.scenario.Evaluation', new=StubEvaluation))
        self.patched_perplexity_evaluation = self._patch_stack.enter_context(
            patch('src.evaluation.scenario.PerplexityEvaluation', new=StubPerplexityEvaluation))
        self.patched_energy_evaluation = self._patch_stack.enter_context(
            patch('src.evaluation.scenario.EnergyEvaluation', new=StubEnergyEvaluation))
        self.patched_pruned_evaluation = self._patch_stack.enter_context(
            patch('src.evaluation.scenario.PrunedEvaluation', new=StubPrunedEvaluation))

    def _create_scenario(self, **overrides):
        params = {
            "model_path": self.model_path,
            "supports_attn_pruning": self.supports_attn_pruning,
            "batch_size": self.batch_size,
            "llm_type": self.llm_type,
            "evaluation_row_count": self.evaluation_row_count,
            "rng_seed": self.rng_seed,
            "scenario_name": self.scenario_name,
            "dataset": self.dataset,
            "device": self.device,
        }
        params.update(overrides)
        return EvaluationScenario(**params)

    def test_constructor(self):
        scenario = self._create_scenario()
        self.assertTrue(EvaluationScenario, type(scenario))
        self.mock_manual_seed.assert_called_once_with(self.rng_seed)
        self.mock_tokenizer_from_pretrained.assert_called_once_with(
            self.model_path, use_fast=True, device=self.device
        )
        self.assertEqual(EOS_TOKEN, self.mock_tokenizer.pad_token)

    def test_baseline_evaluations(self):
        test_cases = [
            (False, False, 6, [StubEvaluation]),
            (True, False, 10, [StubEnergyEvaluation, StubEvaluation]),
            (False, True, 3, [StubPerplexityEvaluation, StubEvaluation]),
            (True, True, 99, [StubPerplexityEvaluation, StubEnergyEvaluation, StubEvaluation])
        ]
        scenario = self._create_scenario()
        for i, (capture_energy, capture_perplexity, expected_repetitions, expected_types) in enumerate(test_cases):
            with self.subTest(capture_energy=capture_energy, capture_perplexity=capture_perplexity,
                              expected_repetitions=expected_repetitions, expected_types=expected_types):
                scenario.add_baseline_evaluation(capture_energy, capture_perplexity=capture_perplexity,
                                                 repetitions=expected_repetitions)
                expected_length = i + 1
                self.assertEqual(expected_length, len(scenario.deferred_baseline))

                evaluation_to_validate = scenario.deferred_baseline[i]()

                self.validate_types(evaluation_to_validate, expected_types)

                expected_label = f'scenario-{self.scenario_name}-baseline-{i}'
                self.validate_common_attributes(evaluation_to_validate, expected_repetitions, expected_label)

    def test_add_warmup_evaluations(self):
        test_cases = [
            (False, False, 6, [StubEvaluation]),
            (True, False, 10, [StubEnergyEvaluation, StubEvaluation]),
            (False, True, 3, [StubPerplexityEvaluation, StubEvaluation]),
            (True, True, 99, [StubPerplexityEvaluation, StubEnergyEvaluation, StubEvaluation])
        ]
        scenario = self._create_scenario()
        for i, (capture_energy, capture_perplexity, expected_repetitions, expected_types) in enumerate(test_cases):
            with self.subTest(capture_energy=capture_energy, capture_perplexity=capture_perplexity,
                              expected_repetitions=expected_repetitions, expected_types=expected_types):
                scenario.add_warmup_evaluation(capture_energy, capture_perplexity=capture_perplexity,
                                               repetitions=expected_repetitions)
                expected_length = i + 1
                self.assertEqual(expected_length, len(scenario.deferred_warmup))

                evaluation_to_validate = scenario.deferred_warmup[i]()

                self.validate_types(evaluation_to_validate, expected_types)

                expected_label = f'scenario-{self.scenario_name}-warmup-{i}'
                self.validate_common_attributes(evaluation_to_validate, expected_repetitions, expected_label)

    def test_add_pruned_evaluations(self):
        test_cases = [
            (False, False, Mock(PruningStrategy), 6, None, [StubEvaluation, StubPrunedEvaluation]),
            (False, False, None, 9, (0, 3), [StubEvaluation]),
            (True, False, Mock(PruningStrategy), 10, (4, 9),
             [StubEnergyEvaluation, StubEvaluation, StubPrunedEvaluation]),
            (False, True, None, 3, None, [StubPerplexityEvaluation, StubEvaluation]),
            (True, True, Mock(PruningStrategy), 99, (22, 29),
             [StubPerplexityEvaluation, StubEnergyEvaluation, StubPrunedEvaluation, StubEvaluation])
        ]
        scenario = self._create_scenario()
        for i, (capture_energy, capture_perplexity, pruning_strategy, expected_repetitions, layer_range, expected_types) \
                in enumerate(test_cases):
            with self.subTest(capture_energy=capture_energy, capture_perplexity=capture_perplexity,
                              pruning_strategy=pruning_strategy, expected_repetitions=expected_repetitions,
                              layer_range=layer_range, expected_types=expected_types):
                evaluation_name = f'test_pruned_evaluation_{i}'
                pruned_start_idx = len(scenario.deferred_evaluations)
                scenario.add_pruned_evaluations(capture_energy=capture_energy, capture_perplexity=capture_perplexity,
                                                pruning_strategy=pruning_strategy, repetitions=expected_repetitions,
                                                layer_range=layer_range, evaluation_name=evaluation_name)

                start_layer, end_layer = layer_range if layer_range else (0, self.num_hidden_layers - 1)
                expected_length = end_layer - start_layer + 1 if layer_range else self.num_hidden_layers
                deferred_under_validation = scenario.deferred_evaluations[pruned_start_idx:]
                self.assertEqual(expected_length, len(deferred_under_validation))

                for j, layer_idx in enumerate(range(start_layer, end_layer + 1)):
                    evaluation_to_validate = deferred_under_validation[j]()
                    self.validate_types(evaluation_to_validate, expected_types)
                    expected_label = f'scenario-{self.scenario_name}-evaluation-{evaluation_name}-layer-{layer_idx}'
                    self.validate_common_attributes(evaluation_to_validate, expected_repetitions, expected_label)
                    _, init_kwargs = evaluation_to_validate.init_args
                    self.assertEqual(pruning_strategy, init_kwargs['pruning_strategy'])
                    self.assertEqual(layer_idx, init_kwargs['layer_idx'])

    def test_execute(self):
        warmup_repetitions = 25
        energy_repetitions = 50
        perplexity_evaluation_name = "test_pruned_evaluation"
        energy_evaluation_name = "test_energy_evaluation"

        mock_pruning_strategy = Mock(PruningStrategy)

        patched_load_dataset = self._patch_stack.enter_context(
            patch('src.evaluation.scenario.load_dataset'))
        self.batch_size = 2
        self.evaluation_row_count = 3

        def build_entry(text):
            return {"text": text}

        dataset_entries = ["a" * 534, "b" * 512, "", "c" * 500, "d" * 675]  # a, b, and d should remain after filtering
        test_split = [build_entry(text_entry) for text_entry in dataset_entries]
        dataset = DatasetDict({
            "test": Dataset.from_list(test_split)
        })
        patched_load_dataset.return_value = dataset

        expected_batch_1 = Mock(Tensor)
        expected_batch_1.to.return_value = expected_batch_1
        expected_batch_2 = Mock(Tensor)
        expected_batch_2.to.return_value = expected_batch_2
        self.mock_tokenizer.side_effect = [expected_batch_1, expected_batch_2]

        self._create_scenario() \
            .add_warmup_evaluation(repetitions=warmup_repetitions) \
            .add_baseline_evaluation(capture_perplexity=True) \
            .add_baseline_evaluation(capture_energy=True, repetitions=energy_repetitions) \
            .add_per_layer_evaluations(capture_perplexity=True, pruning_strategy=mock_pruning_strategy,
                                       evaluation_name=perplexity_evaluation_name) \
            .add_per_layer_evaluations(capture_energy=True, pruning_strategy=mock_pruning_strategy,
                                       evaluation_name=energy_evaluation_name) \
            .execute()

        self.assertEqual(67, len(StubEvaluation.get_creation_history()))
        self.assertEqual(dataset_entries[0:2], self.mock_tokenizer.call_args_list[0][0][0])
        self.assertEqual(dataset_entries[-1:], self.mock_tokenizer.call_args_list[1][0][0])

        for evaluation in StubEvaluation.get_creation_history():
            self.assertEqual([expected_batch_1, expected_batch_2], evaluation.call_args_list[0])

    def validate_common_attributes(self, to_validate: StubEvaluation, expected_repetitions: int, expected_label: str):
        _, init_kwargs = to_validate.init_args
        expected = self
        self.assertEqual(expected.model_path, init_kwargs['model_path'])
        self.assertEqual(expected.scenario_name, init_kwargs['scenario_name'])
        self.assertEqual(expected.supports_attn_pruning, init_kwargs['supports_attn_pruning'])
        self.assertEqual(expected.device, init_kwargs['device'])
        self.assertEqual(expected.llm_type, init_kwargs['llm_type'])
        self.assertEqual(expected_repetitions, init_kwargs['repetitions'])
        self.assertEqual(expected_label, init_kwargs['label'])

    def validate_types(self, to_validate: StubEvaluation, expected_types: List[Type[StubEvaluation]]):
        expected_classes_present = reduce(
            lambda test_success, class_type: isinstance(to_validate, class_type) and test_success,
            expected_types, True)
        unexpected_types = [e for e in EVALUATION_CLASSES if e not in expected_types]
        unexpected_class_present = reduce(
            lambda test_failure, class_type: isinstance(to_validate, class_type) or test_failure,
            unexpected_types, False)

        self.assertTrue(expected_classes_present)
        self.assertFalse(unexpected_class_present)


if __name__ == '__main__':
    unittest.main()
