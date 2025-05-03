import torch
from torch import Tensor
from torch.nn import functional


def cross_entropy(input_ids: Tensor, attention_mask: Tensor, logits: Tensor) -> Tensor:
    labels = input_ids.detach()  # labels are derived from input, detach to avoid affecting gradients
    labels_trimmed = labels[:, 1:].contiguous()  # Drop the first label - it isn't useful as no logits are generated
    # for it
    labels_flattened = labels_trimmed.view(-1)  # Flatten the labels to a vector, [batch_size * labels_sequence_length],
    # in preparation for cross_entropy calculation
    logits_trimmed = logits[:, :-1].contiguous()  # Last logit has no label to compare with
    logits_flattened = logits_trimmed.view(-1, logits.size(-1))  # Flatten the logits to a matrix [batch_size *
    # sequence_length, vocab_size]
    per_id_loss = functional.cross_entropy(logits_flattened, labels_flattened,
                                           reduction='none')  # vector of per id/token losses
    # apply the attention mask to remove padding, which can skew perplexity measurements
    attention_mask_flattened = attention_mask[:, 1:].reshape(-1).contiguous().to(logits.device)
    return per_id_loss * attention_mask_flattened


def aggregate_perplexity(losses: Tensor, token_count: int) -> float:
    if losses is None or losses.numel() == 0 or not token_count:
        perplexity = float('nan')
    else:
        aggregate_loss = losses.sum()
        perplexity = torch.exp(aggregate_loss / token_count).item()
    return perplexity
