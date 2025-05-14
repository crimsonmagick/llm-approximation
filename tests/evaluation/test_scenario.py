import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, Mock

from torch import nn
from torch.fx.experimental.unification.dispatch import namespace

from src.models.model_resolution import LLMType


class EvaluationTests(unittest.TestCase):
    def test_base_evaluation(self):
        with ExitStack() as stack:
            load_dataset_mock = stack.enter_context(patch('datasets.load_dataset'))
            dataset_mock = MagicMock()
            data = ['z' * 501, "Short text", None]
            dataset_mock.__getitem__.return_value = data
            load_dataset_mock.return_value = dataset_mock

            resolve_model_mock = stack.enter_context(patch('src.models.model_resolution.resolve_model'))
            model_mock = Mock(nn.Module)
            resolve_model_mock.return_value = model_mock

            model_path = "test/path/model"
            dataset = ("test/path/dataset", "dataset_name")
            evaluation_size_rows = 20
            scenario_name = "base_scenario"
            supports_attn_pruning = True
            device = "testdevice"
            repetitions = 5
            batch_size = 10
            llm_type: LLMType.LLAMA_3





            print('hi')


if __name__ == '__main__':
    unittest.main()
