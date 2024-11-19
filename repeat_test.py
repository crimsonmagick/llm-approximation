import torch
from torch import tensor

from llama.models.modeling_pruned_llama import PrunedLlamaSdpaAttention

# Sample inputs
batch = 2
num_key_value_heads = 3
seqlen = 4
head_dim = 5
n_rep = 2

# Generate dummy tensor
hidden_states = torch.randn(batch, num_key_value_heads, seqlen, head_dim)

# Repeat using repeat_kv
output_repeat_kv = PrunedLlamaSdpaAttention.repeat_kv(hidden_states, n_rep)

# Construct pruned_kv_counts for uniform repetition
pruned_kv_counts = tensor([n_rep] * num_key_value_heads)

# Repeat using repeat_kv_pruned
output_repeat_kv_pruned = PrunedLlamaSdpaAttention.repeat_kv_pruned(hidden_states, pruned_kv_counts)

# Validate outputs
assert torch.equal(output_repeat_kv, output_repeat_kv_pruned), "Outputs are not equivalent"
assert output_repeat_kv.shape == (batch, num_key_value_heads * n_rep, seqlen, head_dim), "Shape mismatch"
print("Outputs are equivalent and aligned.")

# Simulate pruning: keep_heads = [0, 1, 3], num_key_value_groups = 2
pruned_kv_counts = tensor([2, 1])  # Example: group 0 keeps 2 heads, group 1 keeps 1 head
hidden_states_pruned = torch.randn(batch, len(pruned_kv_counts), seqlen, head_dim)

# Repeat using repeat_kv_pruned with non-uniform counts
output_repeat_kv_pruned = PrunedLlamaSdpaAttention.repeat_kv_pruned(hidden_states_pruned, pruned_kv_counts)

# Check the shape
expected_heads = pruned_kv_counts.sum().item()  # Total heads after repetition
assert output_repeat_kv_pruned.shape == (batch, expected_heads, seqlen, head_dim), "Shape mismatch for pruned case"

print(f"Output shape with pruning: {output_repeat_kv_pruned.shape}")
