import argparse
import csv

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def get_evaluation_name(evaluation_str):
    return 'Baseline' if evaluation_str == 'Baseline' else f'Layer {evaluation_str}'


def print_metrics(metric_name, energies, confidence_level):
    energies_sorted = energies.copy()
    energies_sorted.sort()

    energy_max = max(energies_sorted)
    energy_min = min(energies_sorted)
    energy_mean = np.mean(energies_sorted)
    energy_median = np.median(energies_sorted)
    energy_std_dev = np.std(energies_sorted, ddof=1)
    energy_ci = stats.t.interval(confidence_level, df=len(energies_sorted) - 1, loc=energy_mean,
                                 scale=energy_std_dev / np.sqrt(len(energies_sorted)))

    print('--------------------------')
    print(f'--------{metric_name}-----------')
    print('--------------------------')
    print(f'energies={energies}')
    print(f"energy_mean={energy_mean}\n"
          f"energy_median={energy_median}\n"
          f"energy_min={energy_min}\n"
          f"energy_max={energy_max}\n"
          f"range={energy_max - energy_min}\n"
          f"energy_std_dev={energy_std_dev}\n"
          f"energy_confidence_interval=({energy_ci[0]}, {energy_ci[1]})\n")


def analyze_diff(dataset_a, dataset_b, confidence_level):
    name_a, energies_a = dataset_a
    name_b, energies_b = dataset_b

    diff = np.array(energies_a) - np.array(energies_b)
    skew = stats.skew(diff)

    t_stat, t_p = stats.ttest_rel(energies_a, energies_b)
    w_stat, w_p = stats.wilcoxon(energies_a, energies_b)
    shap_w, shap_p = stats.shapiro(diff)
    alpha = round(1 - confidence_level, 2)
    if w_p < alpha and t_p < alpha:
        # print(f"{name_a} vs {name_b}: skew: {skew}, , shap_t: {shap_w}, shap_p: {shap_p}, t_stat: {t_stat}, t_p: {t_p}, w_stat: {w_stat}, w_p: {w_p}")
        print(f"{name_a} vs {name_b}: t_stat: {t_stat}, t_p: {t_p}, w_stat: {w_stat}, w_p: {w_p}")


if __name__ == '__main__':
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
    pruned_by_layer = dict()

    # # Read and process the CSV file
    with open(input_path, mode='r') as file:
        measurements_per_layer = 20
        reader = csv.DictReader(file)

        for row in reader:
            label = row['label']
            pruning_strategy = row['pruning_strategy']
            metadata = row['pruning_metadata']
            mj_per_token = float(row['average_energy_per_token_mj'])
            if "baseline" in label:
                baseline_energies.append(mj_per_token)
            elif pruning_strategy == 'EveryOtherHead' and metadata:
                split_metadata = metadata.split('|')
                if len(split_metadata) > 0:
                    layer_a = split_metadata[0]
                    layer_energies = pruned_by_layer.setdefault(layer_a, [])
                    layer_energies.append(mj_per_token)

    by_evaluation_name = pruned_by_layer.copy()
    by_evaluation_name['Baseline'] = baseline_energies
    # analyze_diff(('Baseline', baseline_energies), ('Layer 0', pruned_by_layer['0']))

    for layer_a, energies_a in by_evaluation_name.items():
        evaluation_name_a = get_evaluation_name(layer_a)
        print_metrics(evaluation_name_a, energies_a, confidence_level)

        for layer_b, energies_b in by_evaluation_name.items():
            evaluation_name_b = get_evaluation_name(layer_b)
            analyze_diff((evaluation_name_a, energies_a), (evaluation_name_b, energies_b), confidence_level)
