import unittest
from unittest.mock import patch

import torch

from src.metrics.function import objective as module_under_test
from tests.util.test_util_mixin import TestUtilMixin


class TestUtil(TestUtilMixin, unittest.TestCase):
    
    @patch("src.metrics.function.objective_torch.cross_entropy")
    def test_cross_entropy_torch_called(self, mock_cross_entropy):
        sequence_count = 20
        sequence_length = 50
        vocabulary_size = 9623
        predicted = self.rand_logits(sequence_count, sequence_length, vocabulary_size)
        labels = self.rand_labels(sequence_count, sequence_length, vocabulary_size)
        
        attention_mask = torch.ones(sequence_count, sequence_length)
        module_under_test.cross_entropy(labels, attention_mask, predicted)
        
        mock_cross_entropy.assert_called_once()
    
    @patch("src.metrics.function.objective_torch.cross_entropy")
    def test_cross_entropy_torch_not_called(self, mock_cross_entropy):
        with self.assertRaises(TypeError):
            module_under_test.cross_entropy([], [], [])
        mock_cross_entropy.assert_not_called()
    
    @patch("src.metrics.function.objective_torch.aggregate_perplexity")
    def test_aggregate_perplexity_torch_called(self, mock_aggregate_perplexity):
        cross_entropy = torch.tensor([0, -4.3, -0.5, 2.2, 7.9, 0.753, 0, 0], device=self.device)
        module_under_test.aggregate_perplexity(cross_entropy, 12)
        mock_aggregate_perplexity.assert_called_once()
    
    @patch("src.metrics.function.objective_torch.aggregate_perplexity")
    def test_aggregate_perplexity_torch_not_called(self, mock_aggregate_perplexity):
        with self.assertRaises(TypeError):
            module_under_test.aggregate_perplexity([], 10)
        mock_aggregate_perplexity.assert_not_called()


if __name__ == '__main__':
    unittest.main()
