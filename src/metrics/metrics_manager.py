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


def get_metrics(suite='default'):
    return [_header] + list(_saved_metrics.setdefault(suite, dict()).values())


def accept(captured: MetricsCapture, suite='default'):
    if _saved_metrics.get(suite) is None:
        _saved_metrics[suite] = dict()
    suite_metrics = _saved_metrics[suite]
    suite_metrics[captured.label] = captured
