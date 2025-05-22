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
    def test_base_evaluation(self):
        with ExitStack() as stack:
            model_path = "test/path/model"
            supports_attn_pruning = True
            batch_size = 10
            llm_type = LLMType.LLAMA_3
            evaluation_row_count = 20
            rng_seed = 711
            scenario_name = "base_scenario"
            dataset = ("test/path/dataset", "dataset_name")
            device = "totes_real_device0"

            mock_manual_seed = stack.enter_context(patch('src.evaluation.scenario.torch.manual_seed'))

            load_dataset_mock = stack.enter_context(patch('datasets.load_dataset'))
            dataset_mock = MagicMock()
            data = ['z' * 501, "Short text", None]
            dataset_mock.__getitem__.return_value = data
            load_dataset_mock.return_value = dataset_mock

            # resolve_model_mock = stack.enter_context(patch('src.models.model_resolution.resolve_model'))
            # model_mock = Mock(nn.Module)
            # resolve_model_mock.return_value = model_mock

            mock_auto_config = stack.enter_context(patch('src.evaluation.scenario.AutoConfig.from_pretrained'))

            mock_from_pretrained = stack.enter_context(patch('src.evaluation.scenario.AutoTokenizer.from_pretrained'))
            mock_tokenizer = MagicMock()
            mock_tokenizer.eos_token = EOS_TOKEN
            mock_tokenizer.pad_token_id = None
            mock_tokenizer.pad_token = None
            mock_from_pretrained.return_value = mock_tokenizer

            EvaluationScenario(model_path, supports_attn_pruning=supports_attn_pruning,
                               batch_size=batch_size, llm_type=llm_type,
                               evaluation_row_count=evaluation_row_count,
                               rng_seed=rng_seed,
                               scenario_name=scenario_name,
                               dataset=dataset, device=device)

            mock_manual_seed.assert_called_once_with(rng_seed)

            mock_from_pretrained \
                .assert_called_once_with(model_path, use_fast=True, device=device)
            self.assertEqual(EOS_TOKEN, mock_tokenizer.pad_token)


if __name__ == '__main__':
    unittest.main()
