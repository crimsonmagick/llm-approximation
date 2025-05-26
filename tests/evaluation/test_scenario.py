import inspect
import unittest
from contextlib import ExitStack
from functools import reduce
from unittest.mock import patch, MagicMock, Mock

from torch import nn

import src.evaluation.scenario
from src.evaluation import scenario as scenario_module
from src.evaluation.evaluation import EnergyEvaluation
from src.evaluation.scenario import EvaluationScenario
from src.models.model_resolution import LLMType

EOS_TOKEN = 5


class StubEvaluation:

    def __init__(self, *args, **kwargs):
        self.init_args = (args, kwargs)
        self.call_args_list = []

    def evaluate(self, tokens_by_batch):
        self.call_args_list.append([[tokens_by_batch], {}])


class StubPrunedEvaluation(StubEvaluation):
    pass


class StubPerplexityEvaluation(StubEvaluation):
    pass


class StubEnergyEvaluation(StubEvaluation):
    pass


EVALUATION_CLASSES = [StubEvaluation, StubPrunedEvaluation, StubPerplexityEvaluation, StubEnergyEvaluation]


class EvaluationTests(unittest.TestCase):
    def setUp(self):
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

        self.mock_config = self._patch_stack.enter_context(
            patch('src.evaluation.scenario.AutoConfig.from_pretrained')
        )
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
            (True, True, 10, [StubPerplexityEvaluation, StubEnergyEvaluation, StubEvaluation])
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
                expected_classes_present = reduce(
                    lambda test_success, class_type: isinstance(evaluation_to_validate, class_type) and test_success,
                    expected_types, True)
                unexpected_types = [e for e in EVALUATION_CLASSES if e not in expected_types]
                unexpected_class_present = reduce(
                    lambda test_failure, class_type: isinstance(evaluation_to_validate, class_type) or test_failure,
                    unexpected_types, False)

                self.assertTrue(expected_classes_present)
                self.assertFalse(unexpected_class_present)
                _, init_kwargs = evaluation_to_validate.init_args
                expected = self
                self.assertEqual(expected.model_path, init_kwargs['model_path'])
                self.assertEqual(expected.scenario_name, init_kwargs['scenario_name'])
                self.assertEqual(expected.supports_attn_pruning, init_kwargs['supports_attn_pruning'])
                self.assertEqual(expected.device, init_kwargs['device'])
                self.assertEqual(expected.llm_type, init_kwargs['llm_type'])
                self.assertEqual(expected_repetitions, init_kwargs['repetitions'])
                expected_label = f'scenario-{expected.scenario_name}-baseline-{i}'
                self.assertEqual(expected_label, init_kwargs['label'])


if __name__ == '__main__':
    unittest.main()
