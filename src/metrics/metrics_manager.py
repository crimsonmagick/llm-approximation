import csv
import os
from abc import ABC, abstractmethod
from typing import NamedTuple, Iterable, Tuple

metrics_directory = 'results/scenarios'

class EnergyCapture(NamedTuple):
    label: str
    pruning_strategy: str
    pruning_metadata: str
    average_energy_per_token_mj: float
    average_time_per_token_ms: float


class PerplexityCapture(NamedTuple):
    label: str
    pruning_strategy: str
    pruning_metadata: str
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

    def __init__(self, scenario: str):
        output_dir = os.path.join(metrics_directory, scenario)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, self._get_filename())

        self.file = open(output_path, 'w')

        self.writer = csv.writer(self.file)
        self.writer.writerow(self._get_header())

    def log(self, captured: Tuple):
        self.writer.writerow(captured)
        self.file.flush()

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
            'pruning_strategy',
            'pruning_metadata',
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
            'pruning_strategy',
            'pruning_metadata',
            'perplexity'
        )
