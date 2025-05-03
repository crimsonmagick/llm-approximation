import math
import unittest
import torch
from torch import Tensor

from src.metrics.function import objective_torch as module_under_test
from tests.util.torch.device import get_device
from tests.util.torch.test_data import load_tensor, save_tensor


class ObjectiveTorchTests(unittest.TestCase):
    
    def setUp(self) -> None:
        self.device = get_device()
        generator_seed = 12  # arbitrary
        # TODO support generators for different devices - need to save multiple versions of expected tensors
        self.generator_device = torch.device("cpu")
        self.generator = torch.Generator(device=self.generator_device).manual_seed(generator_seed)
    
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
    
    def test_aggregate_perplexity(self):
        cross_entropy = torch.tensor([0, -4.3, -0.5, 2.2, 7.9, 0.753, 0, 0], device=self.device)
        token_count = 5
        perplexity = module_under_test.aggregate_perplexity(cross_entropy, token_count)
        expected_perplexity = 3.3554978370666504
        torch.testing.assert_close(expected_perplexity, perplexity, atol=1e-6, rtol=1e-7)
    
    def test_aggregate_perplexity_zero_tokens(self):
        cross_entropy = torch.ones(5, device=self.device)
        token_count = 0
        perplexity = module_under_test.aggregate_perplexity(cross_entropy, token_count)
        self.assertTrue(math.isnan(perplexity))
        
    def test_aggregate_perplexity_no_tokens(self):
        cross_entropy = torch.ones(5, device=self.device)
        token_count = None
        perplexity = module_under_test.aggregate_perplexity(cross_entropy, token_count)
        self.assertTrue(math.isnan(perplexity))
    
    def test_aggregate_perplexity_empty_losses(self):
        cross_entropy = torch.empty(0)
        token_count = 5
        perplexity = module_under_test.aggregate_perplexity(cross_entropy, token_count)
        self.assertTrue(math.isnan(perplexity))
    
    def test_aggregate_perplexity_no_losses(self):
        cross_entropy = None
        token_count = 5
        perplexity = module_under_test.aggregate_perplexity(cross_entropy, token_count)
        self.assertTrue(math.isnan(perplexity))
    
    def rand_logits(self, sequence_count, sequence_length, vocabulary_size) -> Tensor:
        logits = (torch.rand(sequence_count, sequence_length, vocabulary_size,
                             generator=self.generator, device=self.generator_device) - 0.5) * 10
        return logits.to(device=self.device)
    
    def rand_labels(self, sequence_count, sequence_length, vocabulary_size) -> Tensor:
        labels = torch.randint(0, vocabulary_size, size=(sequence_count, sequence_length), generator=self.generator,
                               device=self.generator_device)
        return labels.to(device=self.device)


if __name__ == '__main__':
    unittest.main()
