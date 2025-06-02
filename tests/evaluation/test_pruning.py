import unittest
from types import SimpleNamespace
from typing import List, Dict
from unittest.mock import Mock

from torch import nn

from src.evaluation.pruning import EveryOtherHead


class PruningTest(unittest.TestCase):
    
    def test_every_other(self):
        test_cases = [
            (3, 1, [0]),
            (12, 2, [0]),
            (23, 3, [0, 2]),
            (9, 4, [0, 2]),
            (1, 5, [0, 2, 4])
        ]
        for layer_idx, num_heads, expected_heads in test_cases:
            with self.subTest(layer_idx=layer_idx, num_heads=num_heads, expected_heads=expected_heads):
                mock_evaluation = Mock()
                mock_evaluation.layer_idx = layer_idx
                model = Mock(spec=nn.Module)
                mock_config = SimpleNamespace()
                mock_config.num_attention_heads = num_heads
                model.config = mock_config
                model.prune_heads = Mock()
                EveryOtherHead(mock_evaluation)(model)
                args, _ = model.prune_heads.call_args
                head_dict: Dict[List] = args[0]
                self.assertEqual({layer_idx: expected_heads}, head_dict)


if __name__ == '__main__':
    unittest.main()
