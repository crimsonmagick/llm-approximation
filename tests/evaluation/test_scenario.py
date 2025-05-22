import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, Mock

from torch import nn
from torch.fx.experimental.unification.dispatch import namespace

from src.evaluation.scenario import EvaluationScenario
from src.models.model_resolution import LLMType

EOS_TOKEN = 5


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
        load_ds = self._patch_stack.enter_context(patch('datasets.load_dataset'))
        ds_mock = MagicMock()
        # make __getitem__ return some dummy data
        ds_mock.__getitem__.return_value = ['z'*501, "short", None]
        load_ds.return_value = ds_mock

        self.mock_config = self._patch_stack.enter_context(
            patch('src.evaluation.scenario.AutoConfig.from_pretrained')
        )
        tok_patcher = patch('src.evaluation.scenario.AutoTokenizer.from_pretrained')
        self.mock_from_pretrained = self._patch_stack.enter_context(tok_patcher)
        self.mock_tokenizer = MagicMock(eos_token=EOS_TOKEN, pad_token_id=None, pad_token=None)
        self.mock_from_pretrained.return_value = self.mock_tokenizer

    def _create_test_scenario(self, **overrides):
        params = {
            "model_path":             self.model_path,
            "supports_attn_pruning":  self.supports_attn_pruning,
            "batch_size":             self.batch_size,
            "llm_type":               self.llm_type,
            "evaluation_row_count":   self.evaluation_row_count,
            "rng_seed":               self.rng_seed,
            "scenario_name":          self.scenario_name,
            "dataset":                self.dataset,
            "device":                 self.device,
        }
        params.update(overrides)
        return EvaluationScenario(**params)

    def test_base_evaluation(self):
        scenario = self._create_test_scenario()
        self.mock_manual_seed.assert_called_once_with(self.rng_seed)
        self.mock_from_pretrained.assert_called_once_with(
            self.model_path, use_fast=True, device=self.device
        )
        self.assertEqual(EOS_TOKEN, self.mock_tokenizer.pad_token)

if __name__ == '__main__':
    unittest.main()
