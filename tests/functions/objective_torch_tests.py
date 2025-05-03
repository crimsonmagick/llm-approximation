import json
import unittest
import torch
from torch import Tensor

from src.metrics.function import objective_torch as module_under_test
from tests.util.torch.test_data import load_tensor

GENERATOR_SEED = 12


class ObjectiveTorchTests(unittest.TestCase):
    
    def setUp(self) -> None:
        self.generator = torch.Generator().manual_seed(GENERATOR_SEED)
    
    def test_cross_entropy_no_mask(self):
        sequence_count = 5
        sequence_length = 10
        vocabulary_size = 12
        predicted = self.rand_logits(sequence_count, sequence_length, vocabulary_size)
        labels = self.rand_labels(sequence_count, sequence_length, vocabulary_size)
        expected_cross_entropy = load_tensor("objective_torch_tests/test_cross_entropy_no_mask.json")
        
        attention_mask = torch.ones(sequence_count, sequence_length)
        cross_entropy = module_under_test.cross_entropy(labels, attention_mask, predicted)
        
        torch.testing.assert_close(expected_cross_entropy, cross_entropy)
    
    def rand_logits(self, sequence_count, sequence_length, vocabulary_size) -> Tensor:
        return (torch.rand(sequence_count, sequence_length, vocabulary_size,
                           generator=self.generator) - 0.5) * 10
    
    def rand_labels(self, sequence_count, sequence_length, vocabulary_size) -> Tensor:
        return torch.randint(0, vocabulary_size, size=(sequence_count, sequence_length), generator=self.generator)
    

if __name__ == '__main__':
    unittest.main()
