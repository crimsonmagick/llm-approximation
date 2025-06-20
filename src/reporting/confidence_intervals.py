import argparse
import csv

import numpy as np
import scipy.stats as stats


def get_evaluation_name(evaluation_str):
    return 'Baseline' if evaluation_str == 'Baseline' else f'Layer {evaluation_str}'


def print_metrics(metric_name, metrics, confidence_level):
    metrics_sorted = metrics.copy()
    metrics_sorted.sort()

    metric_max = max(metrics_sorted)
    metric_min = min(metrics_sorted)
    metric_mean = np.mean(metrics_sorted)
    metric_median = np.median(metrics_sorted)
    metric_std_dev = np.std(metrics_sorted, ddof=1)
    metric_ci = stats.t.interval(confidence_level, df=len(metrics_sorted) - 1, loc=metric_mean,
                                 scale=metric_std_dev / np.sqrt(len(metrics_sorted)))

    print('--------------------------')
    print(f'--------{metric_name}-----------')
    print('--------------------------')
    print(f"mean={metric_mean}\n"
          f"median={metric_median}\n"
          f"min={metric_min}\n"
          f"max={metric_max}\n"
          f"range={metric_max - metric_min}\n"
          f"std_dev={metric_std_dev}\n"
          f"confidence_interval=({metric_ci[0]}, {metric_ci[1]})\n")


def analyze_diff(metrics_a, metrics_b):
    diff = np.array(metrics_a) - np.array(metrics_b)
    skew = stats.skew(diff)
    shap_w, shap_p = stats.shapiro(diff)

    t_stat, t_p = stats.ttest_rel(metrics_a, metrics_b)
    w_stat, w_p = stats.wilcoxon(metrics_a, metrics_b)

    return t_stat, t_p, w_stat, w_p, skew, shap_w, shap_p

def compare_layers(metric_description, metrics_by_layer, confidence_level):
    alpha = round(1 - confidence_level, 2)
    for layer_idx, energies_a in metrics_by_layer.items():
        evaluation_name_a = metric_description + ' ' +  get_evaluation_name(layer_idx)
        print_metrics(evaluation_name_a, energies_a, confidence_level)

        for layer_b, energies_b in metrics_by_layer.items():
            evaluation_name_b = get_evaluation_name(layer_b)
            t_stat, t_p, w_stat, w_p, _, _, _ = analyze_diff(energies_a, energies_b)
            if w_p < alpha and t_p < alpha:
                print(
                    f"{metric_description} - {evaluation_name_a} vs {evaluation_name_b}: t_stat: {t_stat}, t_p: {t_p}, w_stat: {w_stat}, w_p: {w_p}")


def main():
    parser = argparse.ArgumentParser(description="Calculates baseline metrics.")
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
    args = parser.parse_args()
    input_path = args.input_path
    confidence_level = args.confidence_level

    baseline_energies = []
    pruned_by_layer_energies = dict()
    baseline_time = []
    pruned_by_layer_time = dict()

    # # Read and process the CSV file
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

    by_evaluation_name_energies = pruned_by_layer_energies.copy()
    by_evaluation_name_energies['Baseline'] = baseline_energies
    by_evaluation_name_times= pruned_by_layer_time.copy()
    by_evaluation_name_times['Baseline'] = baseline_time

    compare_layers("Energy", by_evaluation_name_energies, confidence_level)
    compare_layers("Time", by_evaluation_name_times, confidence_level)




if __name__ == '__main__':
    main()
