import unittest

import torch
from torch import tensor

from src.llama.models.modeling_pruned_llama import PrunedLlamaSdpaAttention
from transformers.models.llama.modeling_llama import repeat_kv


class TestRepeatKvPruned(unittest.TestCase):
    """
    Tests to assert alignment correctness of :meth:`PrunedLlamaSdpaAttention#repeat_kv_pruned`.
    """
    
    def setUp(self):
        # Initial test parameters, arbitrarily chosen, but kept small for simplicity's sake.
        self.batch_size = 12
        self.seq_len = 4
        self.head_dim = 5
    
    def test_kv_states_uniform(self):
        # Construct pruned_kv_counts for uniform repetition
        n_rep = 2
        num_key_value_heads = 3
        uniform_kv_counts = tensor([n_rep] * num_key_value_heads)
        # Generate dummy tensor for test input
        states_per_kv_head = torch.randn(self.batch_size, num_key_value_heads, self.seq_len, self.head_dim)
        
        expected_repeat_kv = repeat_kv(states_per_kv_head, n_rep)
        under_test_repeat_kv = PrunedLlamaSdpaAttention.repeat_kv(states_per_kv_head, uniform_kv_counts)
        
        self.assertEqual((self.batch_size, num_key_value_heads * n_rep, self.seq_len, self.head_dim),
                         under_test_repeat_kv.shape, "Repeated Key-Values Shape mismatch")
        self.assertTrue(torch.equal(expected_repeat_kv, under_test_repeat_kv),
                        "Repeated Key-Values do not match expected Key-Values")
    
    def test_kv_states_non_uniform(self):
        # Simulate pruning: keep_heads = [0, 1, 3], num_key_value_groups = 2
        pruned_kv_counts = tensor([2, 1])  # Example: group 0 keeps 2 heads, group 1 keeps 1 head
        # Generate dummy tensor for test input
        states_per_kvhead = torch.randn(self.batch_size, len(pruned_kv_counts), self.seq_len, self.head_dim)
        
        expected_repeated_kv = self.get_expected_repeated_kv(states_per_kvhead, pruned_kv_counts)
        expected_repeated_count = sum(pruned_kv_counts)
        
        under_test_repeated_kv = PrunedLlamaSdpaAttention.repeat_kv(states_per_kvhead, pruned_kv_counts)
        
        self.assertEqual((self.batch_size, expected_repeated_count, self.seq_len, self.head_dim),
                         under_test_repeated_kv.shape, "Repeated Key-Values Shape mismatch")
        self.assertTrue(torch.equal(expected_repeated_kv, under_test_repeated_kv),
                        "Repeated Key-Values do not match expected Key-Values")
    
    def get_expected_repeated_kv(self, states_per_kvhead, pruned_kv_counts):
        splits = states_per_kvhead.split(1, dim=1)
        zipped = zip(splits, pruned_kv_counts)
        expected_repeated = tuple(map(lambda t: t[0].expand(self.batch_size, t[1], self.seq_len, self.head_dim), zipped))
        return torch.cat(expected_repeated, dim=1)


if __name__ == '__main__':
    unittest.main()
