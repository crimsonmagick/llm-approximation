import argparse
import csv

import numpy as np
import scipy.stats as stats


def print_metrics(metric_name, energies, confidence_level):
    energies.sort()

    energy_max = max(energies)
    energy_min = min(energies)
    energy_mean = np.mean(energies)
    energy_median = np.median(energies)
    energy_std_dev = np.std(energies, ddof=1)
    energy_ci = stats.t.interval(confidence_level, df=len(energies) - 1, loc=energy_mean,
                                 scale=energy_std_dev / np.sqrt(len(energies)))

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
