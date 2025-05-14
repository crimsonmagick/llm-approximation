import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, Mock

from torch import nn
from torch.fx.experimental.unification.dispatch import namespace

from src.evaluation.evaluation import Evaluation
from src.models.model_resolution import LLMType, resolve_model


class EvaluationTests(unittest.TestCase):
    def test_base_evaluation(self):
        with ExitStack() as stack:
            mock_model = Mock(nn.Module)
            mock_resolve_model = stack.enter_context(patch('src.evaluation.evaluation.resolve_model'))
            mock_resolve_model.return_value = mock_model

            model_path = "test/path/model"
            tokens_by_batch = [[1, 52, 123, 666]]
            scenario_name = "base-scenario"
            supports_attn_pruning = True
            device = "testdevice"
            repetitions = 5
            label = 'base-test-label'
            llm_type = LLMType.LLAMA_3
            Evaluation(model_path=model_path, scenario_name=scenario_name, supports_attn_pruning=supports_attn_pruning,
                       device=device, repetitions=repetitions, tokens_by_batch=tokens_by_batch, llm_type=llm_type, label=label)


if __name__ == '__main__':
    unittest.main()
