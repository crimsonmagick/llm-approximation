from . import objective_torch as func_torch


def aggregate_perplexity(losses, token_count: int) -> float:
    if _is_torch(losses):
        return func_torch.aggregate_perplexity(losses, token_count)
    else:
        _raise_error()


def cross_entropy(input_ids, attention_mask, logits):
    if _is_torch(input_ids) and _is_torch(attention_mask) and _is_torch(logits):
        return func_torch.cross_entropy(input_ids, attention_mask, logits)
    else:
        _raise_error()


def _is_torch(tensor):
    return 'torch' in str(type(tensor))


def _raise_error():
    raise TypeError("Only PyTorch is currently supported.")
