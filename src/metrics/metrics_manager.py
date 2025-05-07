import sys
from typing import NamedTuple, List


class MetricsCapture(NamedTuple):
    label: str
    layer_idx: int
    head_idxs: List[int]
    perplexity: float
    average_energy_per_token_mj: float
    average_time_per_token_ms: float
    allocated_memory: float
    temperature: float


_saved_metrics = dict()
_header = (
    'label',
    'layer_idx',
    'head_idxs',
    'perplexity',
    'average_energy_per_token_mj',
    'average_time_per_token_ms',
    'allocated_memory',
    'temperature'
)


def clear_saved():
    global _saved_metrics
    _saved_metrics = dict()


def get_metrics():
    return [_header] + list(_saved_metrics.values())


def save_metrics(captured: MetricsCapture):
    _saved_metrics[captured.label] = captured
    return sys.modules[__name__]
