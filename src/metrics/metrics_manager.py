import csv
import os
from abc import ABC, abstractmethod
from typing import NamedTuple, List, Iterable, Tuple

metrics_directory = 'results/metrics'

_energy_loggers = dict()
_perplexity_loggers = dict()


class EnergyCapture(NamedTuple):
    label: str
    layer_idx: int
    head_idxs: List[int]
    average_energy_per_token_mj: float
    average_time_per_token_ms: float


class PerplexityCapture(NamedTuple):
    label: str
    layer_idx: int
    head_idxs: List[int]
    perplexity: float


class MetricLogger(ABC):

    @staticmethod
    @abstractmethod
    def _get_header() -> Iterable[str]:
        pass

    @staticmethod
    @abstractmethod
    def _get_filename() -> str:
        pass

    def __init__(self, suite: str):
        output_dir = os.path.join(metrics_directory, suite)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, self._get_filename())

        self.file = open(output_path, 'w')

        self.writer = csv.writer(self.file)
        self.writer.writerow(self._get_header())

    def log(self, captured: Tuple):
        self.writer.writerow(captured)

    def close(self):
        self.file.close()


class EnergyLogger(MetricLogger):

    @staticmethod
    def _get_filename() -> str:
        return 'energy-metrics.csv'

    @staticmethod
    def _get_header() -> Iterable[str]:
        return (
            'label',
            'layer_idx',
            'head_idxs',
            'average_energy_per_token_mj',
            'average_time_per_token_ms',
        )


class PerplexityLogger(MetricLogger):

    @staticmethod
    def _get_filename() -> str:
        return 'perplexity-metrics.csv'

    @staticmethod
    def _get_header() -> Iterable[str]:
        return (
            'label',
            'layer_idx',
            'head_idxs',
            'perplexity'
        )


def accept_energy(captured: EnergyCapture, *, suite):
    if _energy_loggers.get(suite) is None:
        _energy_loggers[suite] = EnergyLogger(suite)
    _energy_loggers[suite].log(captured)


def accept_perplexity(captured: PerplexityCapture, *, suite):
    if _perplexity_loggers.get(suite) is None:
        _perplexity_loggers[suite] = PerplexityLogger(suite)
    _perplexity_loggers[suite].log(captured)
