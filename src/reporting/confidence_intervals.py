import argparse
import csv

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


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

def analyze_diff(dataset_a, dataset_b):
    name_a, energies_a = dataset_a
    name_b, energies_b = dataset_b
    diff = np.array(energies_a) - np.array(energies_b)

    stats.probplot(energies_a, dist="norm", plot=plt)
    plt.title(f"{name_a} ProbPlot")
    plt.show()

    stats.probplot(energies_b, dist="norm", plot=plt)
    plt.title(f"{name_b} ProbPlot")
    plt.show()

    plt.title(f"{name_a} vs {name_b}")
    plt.xlabel("Iteration")
    plt.ylabel("mJ")
    plt.plot(energies_a, label=name_a)
    plt.plot(energies_b, label=name_b)
    plt.legend()
    plt.show()

    plt.title(f"{name_a} - {name_b} diff")
    plt.xlabel("Iteration")
    plt.ylabel("mJ delta")
    plt.plot(diff, label="Diff")
    plt.legend()
    plt.show()


    stats.probplot(diff, dist="norm", plot=plt)
    plt.title(f"{name_a} - {name_b} ProbPlot")
    plt.show()

    t_stat, t_p = stats.ttest_rel(energies_a, energies_b)
    print(f"{name_a} vs {name_b}: t_stat: {t_stat}, t_p: {t_p}")



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
                    layer = split_metadata[0]
                    layer_energies = pruned_by_layer.setdefault(layer, [])
                    layer_energies.append(mj_per_token)

    print_metrics('Baseline', baseline_energies, confidence_level)
    for layer, energies in pruned_by_layer.items():
        print_metrics(f'Layer {layer}', energies, confidence_level)
    analyze_diff(('Baseline', baseline_energies), ('Layer 1', pruned_by_layer['1']))
    analyze_diff(('Layer 0', pruned_by_layer['0']), ('Layer 1', pruned_by_layer['1']))
    analyze_diff(('Layer 1', pruned_by_layer['1']), ('Layer 2', pruned_by_layer['2']))



