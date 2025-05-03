import json
import unittest
import torch
from torch import Tensor

from src.metrics.function import objective_torch as module_under_test
from tests.util.torch.device import get_device
from tests.util.torch.test_data import load_tensor, save_tensor

GENERATOR_SEED = 12


class ObjectiveTorchTests(unittest.TestCase):
    
    def setUp(self) -> None:
        self.generator = torch.Generator().manual_seed(GENERATOR_SEED)
        self.device = get_device()
    
    def test_cross_entropy_no_mask(self):
        sequence_count = 5
        sequence_length = 10
        vocabulary_size = 12
        predicted = self.rand_logits(sequence_count, sequence_length, vocabulary_size)
        labels = self.rand_labels(sequence_count, sequence_length, vocabulary_size)
        expected_cross_entropy = load_tensor("objective_torch_tests/test_cross_entropy_no_mask.json",
                                             device=self.device)
        
        attention_mask = torch.ones(sequence_count, sequence_length)
        cross_entropy = module_under_test.cross_entropy(labels, attention_mask, predicted)
        
        torch.testing.assert_close(expected_cross_entropy, cross_entropy)
    
    def test_cross_entropy_masked(self):
        sequence_count = 8
        sequence_length = 20
        vocabulary_size = 17
        
        # Add masks at the beginning and end of the sequence
        mask_1_start = 0
        mask_1_end = 2
        mask_2_start = sequence_length - 3
        mask_2_end = sequence_length
        attention_mask = torch.ones(sequence_count, sequence_length, device=self.device)
        attention_mask[:, mask_1_start:mask_1_end] = 0
        attention_mask[:, mask_2_start:mask_2_end] = 0
        
        predicted = self.rand_logits(sequence_count, sequence_length, vocabulary_size)
        labels = self.rand_labels(sequence_count, sequence_length, vocabulary_size)
        expected_cross_entropy = load_tensor("objective_torch_tests/test_cross_entropy_masked.json", device=self.device)
        
        cross_entropy = module_under_test.cross_entropy(labels, attention_mask, predicted)
        torch.testing.assert_close(expected_cross_entropy, cross_entropy)


    def rand_logits(self, sequence_count, sequence_length, vocabulary_size) -> Tensor:
        return (torch.rand(sequence_count, sequence_length, vocabulary_size,
                           generator=self.generator, device=self.device) - 0.5) * 10


    def rand_labels(self, sequence_count, sequence_length, vocabulary_size) -> Tensor:
        return torch.randint(0, vocabulary_size, size=(sequence_count, sequence_length), generator=self.generator,
                             device=self.device)


if __name__ == '__main__':
    unittest.main()
