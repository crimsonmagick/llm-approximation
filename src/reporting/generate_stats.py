import argparse
import csv
import os
from typing import NamedTuple

import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt


def get_model_name(evaluation_str):
    return 'Baseline' if evaluation_str == 'Baseline' else f'Layer {evaluation_str}'


class ModelComparison(NamedTuple):
    model_a: str
    model_b: str
    t_stat: float
    t_p: float
    w_stat: float
    w_p: float
    skew: float


class ModelStats(NamedTuple):
    model: str
    max: float
    min: float
    range: float
    mean: float
    norm_range: float
    median: float
    norm_range: float
    std_dev: float
    cv: str  # coefficient of variation
    ci: str  # confidence interval


class StatGenerator:
    def __init__(self, *, stat_dir, confidence_level, metric_name, metric_by_layer, quantiles_enabled=False):
        self.stat_dir = stat_dir
        self.confidence_level = confidence_level
        self.metric_name = metric_name
        self.metric_by_layer = metric_by_layer
        self.quantiles_enabled = quantiles_enabled

    def generate_stats(self):
        filename = f'{self.metric_name}-stats.csv'
        output_path = os.path.join(self.stat_dir, filename)
        file = open(output_path, 'w')
        csv_writer = csv.writer(file)
        field_names = vars(ModelStats)['_fields']
        csv_writer.writerow(field_names)

        for layer_idx, metrics in self.metric_by_layer.items():
            model = get_model_name(layer_idx)

            metrics_sorted = metrics.copy()
            metrics_sorted.sort()

            metric_max = max(metrics_sorted)
            metric_min = min(metrics_sorted)
            metric_range = metric_max - metric_min
            metric_mean = np.mean(metrics_sorted)
            metric_norm_range = metric_range / metric_mean
            metric_median = np.median(metrics_sorted)
            metric_std_dev = np.std(metrics_sorted, ddof=1)
            metric_cv = metric_std_dev / metric_mean
            metric_ci = stats.t.interval(self.confidence_level, df=len(metrics_sorted) - 1, loc=metric_mean,
                                         scale=metric_std_dev / np.sqrt(len(metrics_sorted)))
            model_stats = ModelStats(model=model, max=metric_max, min=metric_min, range=metric_range, mean=metric_mean,
                                     norm_range=metric_norm_range, median=metric_median, std_dev=metric_std_dev,
                                     cv=metric_cv,
                                     ci=metric_ci)
            csv_writer.writerow(model_stats)
        file.close()
        return self

    def generate_comparison(self):
        filename = f'{self.metric_name}-comparison.csv'
        output_path = os.path.join(self.stat_dir, filename)
        file = open(output_path, 'w')
        csv_writer = csv.writer(file)
        field_names = vars(ModelComparison)['_fields']
        csv_writer.writerow(field_names)

        for layer_idx, metrics_a in self.metric_by_layer.items():
            model_a = get_model_name(layer_idx)
            for layer_b, metrics_b in self.metric_by_layer.items():
                if metrics_a != metrics_b:
                    model_b = get_model_name(layer_b)
                    t_stat, t_p, w_stat, w_p, skew, _, _, diff = self._analyze_diff(metrics_a, metrics_b)
                    comparison = ModelComparison(model_a, model_b, t_stat, t_p, w_stat, w_p, skew)
                    csv_writer.writerow(comparison)
                    if self.quantiles_enabled:
                        self._graph_quantiles(diff, model_a, model_b)
        file.close()
        return self

    @staticmethod
    def _analyze_diff(metrics_a, metrics_b):
        diff = np.array(metrics_a) - np.array(metrics_b)
        skew = stats.skew(diff)
        shap_w, shap_p = stats.shapiro(diff)

        t_stat, t_p = stats.ttest_rel(metrics_a, metrics_b)
        w_stat, w_p = stats.wilcoxon(metrics_a, metrics_b)

        return t_stat, t_p, w_stat, w_p, skew, shap_w, shap_p, diff

    def _graph_quantiles(self, diff, model_a, model_b):
        title = f"{model_a} - {model_b} {self.metric_name} ProbPlot"
        print(f"Graphing {self.metric_name} {title}")
        stats.probplot(diff, dist="norm", plot=plt)
        plt.title(title)
        output_dir = os.path.join(self.stat_dir, 'quantiles', self.metric_name)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{model_a}-{model_b}')
        plt.savefig(output_path)
        plt.clf()


def main():
    parser = argparse.ArgumentParser(description="Computes metric stats.")
    parser.add_argument(
        '--input-path',
        type=str,
        help='Input path for csv file, required',
        required=True
    )
    parser.add_argument(
        '--confidence-level',
        type=float,
        help='Input path for csv file, required',
        required=True
    )
    parser.add_argument(
        '--quantiles-enabled',
        action='store_true',
        help='Enables graphing model diff quantiles, optional'
    )
    args = parser.parse_args()
    input_path = args.input_path
    confidence_level = args.confidence_level
    quantiles_enabled = args.quantiles_enabled

    baseline_energies = []
    pruned_by_layer_energies = dict()
    baseline_time = []
    pruned_by_layer_time = dict()

    with open(input_path, mode='r') as file:
        reader = csv.DictReader(file)

        for row in reader:
            label = row['label']
            pruning_strategy = row['pruning_strategy']
            metadata = row['pruning_metadata']
            mj_per_token = float(row['average_energy_per_token_mj'])
            time_per_token = float(row['average_time_per_token_ms'])
            if "baseline" in label:
                baseline_energies.append(mj_per_token)
                baseline_time.append(time_per_token)
            elif pruning_strategy == 'EveryOtherHead' and metadata:
                split_metadata = metadata.split('|')
                if len(split_metadata) > 0:
                    layer_idx = split_metadata[0]
                    layer_energies = pruned_by_layer_energies.setdefault(layer_idx, [])
                    layer_energies.append(mj_per_token)
                    layer_times = pruned_by_layer_time.setdefault(layer_idx, [])
                    layer_times.append(time_per_token)

    baseline_model_name = 'Baseline'
    by_evaluation_name_energies = pruned_by_layer_energies.copy()
    by_evaluation_name_energies[baseline_model_name] = baseline_energies
    by_evaluation_name_times = pruned_by_layer_time.copy()
    by_evaluation_name_times[baseline_model_name] = baseline_time

    stat_dir = os.path.dirname(input_path)
    StatGenerator(stat_dir=stat_dir, confidence_level=confidence_level, quantiles_enabled=quantiles_enabled,
                  metric_name="energy", metric_by_layer=by_evaluation_name_energies) \
        .generate_stats() \
        .generate_comparison()
    StatGenerator(stat_dir=stat_dir, confidence_level=confidence_level, quantiles_enabled=quantiles_enabled,
                  metric_name="time", metric_by_layer=by_evaluation_name_times) \
        .generate_stats() \
        .generate_comparison()


if __name__ == '__main__':
    main()
