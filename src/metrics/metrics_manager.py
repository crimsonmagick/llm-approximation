from typing import NamedTuple, List


class EnergyCapture(NamedTuple):
    label: str
    layer_idx: int
    head_idxs: List[int]
    average_energy_per_token_mj: float
    average_time_per_token_ms: float

class PerplexityMetricsCapture(NamedTuple):
    label: str
    layer_idx: int
    head_idxs: List[int]
    perplexity: float


_saved_energy_metrics = dict()
_saved_perplexity_metrics = dict()

_energy_header = (
    'label',
    'layer_idx',
    'head_idxs',
    'average_energy_per_token_mj',
    'average_time_per_token_ms',
)
_perplexity_header = (
    'label',
    'layer_idx',
    'head_idxs',
    'perplexity'
)


def clear_saved():
    global _saved_energy_metrics
    _saved_energy_metrics = dict()


def get_energy_metrics(suite='default'):
    return [_energy_header] + list(_saved_energy_metrics.setdefault(suite, dict()).values())


def accept_energy(captured: EnergyCapture, suite='default'):
    if _saved_energy_metrics.get(suite) is None:
        _saved_energy_metrics[suite] = dict()
    suite_metrics = _saved_energy_metrics[suite]
    suite_metrics[captured.label] = captured

def accept_perplexity(captured: PerplexityMetricsCapture, suite='default'):
    if _saved_perplexity_metrics.get(suite) is None:
        _saved_perplexity_metrics[suite] = dict()
    suite_metrics = _saved_perplexity_metrics[suite]
    suite_metrics[captured.label] = captured


def get_perplexity_metrics(suite='default'):
    return [_perplexity_header] + list(_saved_perplexity_metrics.setdefault(suite, dict()).values())
