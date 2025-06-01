import math
import unittest
import torch

from src.metrics.function import objective_torch as module_under_test
from tests.util.test_util_mixin import TestUtilMixin
from tests.util.torch_utils.test_data import load_tensor


class ObjectivesTest(TestUtilMixin, unittest.TestCase):
    
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


if __name__ == '__main__':
    unittest.main()
