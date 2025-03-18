import argparse
import csv
import os
import re
import string
from statistics import mean, median, stdev
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

def add_entry(metric_by_layer, layer_idx, batch_idx, entry):
    if layer_idx not in metric_by_layer:
        metric_by_layer[layer_idx] = list()
    metric_by_batch = metric_by_layer[layer_idx]
    while len(metric_by_batch) <= batch_idx:
        metric_by_batch.append(list())
    metric_by_batch[batch_idx].append(entry)

def flatten(metric_by_batch):
    return [metric for batch_list in
            metric_by_batch for metric in
            batch_list]

def generate_layer_energy_metrics(layer_name, layer_energy_per_batch):
    if len(layer_energy_per_batch) > 1:
        for batch_idx in range(len(layer_energy_per_batch)):
            energies_sorted = sorted(layer_energy_per_batch[batch_idx])

            energy_max = max(energies_sorted)
            energy_min = min(energies_sorted)
            energy_mean = mean(energies_sorted)
            energy_median = median(energies_sorted)
            energy_std_dev = stdev(energies_sorted)

            print(f"layer={layer_name}, batch={batch_idx}, energy_mean={energy_mean}\n"
                  f"layer={layer_name}, batch={batch_idx}, energy_median={energy_median}\n"
                  f"layer={layer_name}, batch={batch_idx}, energy_min={energy_min}\n"
                  f"layer={layer_name}, batch={batch_idx}, energy_max={energy_max}\n"
                  f"layer={layer_name}, batch={batch_idx}, range={energy_max - energy_min}\n"
                  f"layer={layer_name}, batch={batch_idx}, energy_std_dev={energy_std_dev}")

    aggregate_energies_sorted = [energy for batch_list in layer_energy_per_batch for energy in batch_list]
    energy_max = max(aggregate_energies_sorted)
    energy_min = min(aggregate_energies_sorted)
    energy_mean = mean(aggregate_energies_sorted)
    energy_median = median(aggregate_energies_sorted)
    energy_std_dev = stdev(aggregate_energies_sorted)
    print(f"layer_name={layer_name}, aggregate_energy_mean={energy_mean}\n"
          f"layer_name={layer_name}, aggregate_energy_median={energy_median}\n"
          f"layer_name={layer_name}, aggregate_energy_min={energy_min}\n"
          f"layer_name={layer_name}, aggregate_energy_max={energy_max}\n"
          f"layer_name={layer_name}, aggregate_range={energy_max - energy_min}\n"
          f"layer_name={layer_name}, aggregate_energy_std_dev={energy_std_dev}")


def generate_stat_report(input_path: str, report_name: str):
    # Read and process the CSV file
    with open(input_path, mode='r') as file:
        reader = csv.DictReader(file)

        perplexity_by_layer = dict()
        energy_by_layer = dict()
        time_by_layer = dict()
        memory_by_layer = dict()

        perplexities = []
        energy_per_token = []
        time_per_token = []
        memory_allocated = []

        for row in reader:
            layer_idx = row['layer_idx'] if row['layer_idx'] else 'baseline'
            test_perplexity = float(row['perplexity'])
            test_energy = float(row['average_energy_per_token_mj'])
            test_time = float(row['average_time_per_token_ms'])
            test_memory = int(row['allocated_memory'])
            label = row['label']
            batch_idx = int(label[len(label) - 1])

            add_entry(perplexity_by_layer, layer_idx, batch_idx, test_perplexity)
            add_entry(energy_by_layer, layer_idx, batch_idx, test_energy)
            add_entry(time_by_layer, layer_idx, batch_idx, test_time)
            add_entry(memory_by_layer, layer_idx, batch_idx, test_memory)

    # generate baseline stats per batch
    generate_layer_energy_metrics('baseline', energy_by_layer['baseline'])

    # generate baseline means
    perplexities_baseline = flatten(perplexity_by_layer.pop('baseline'))
    perplexity_baseline_mean = mean(perplexities_baseline)
    energy_baseline = flatten(energy_by_layer.pop('baseline'))
    energy_baseline_mean = mean(energy_baseline)
    time_baseline = flatten(time_by_layer.pop('baseline'))
    time_baseline_mean = mean(time_baseline)
    memory_baseline = flatten(memory_by_layer.pop('baseline'))
    memory_baseline_mean = mean(memory_baseline)

    # Normalize and store final values
    layers = sorted(perplexity_by_layer.keys(), key=lambda x: int(x))
    for layer_idx in layers:
        generate_layer_energy_metrics(layer_idx, energy_by_layer[layer_idx])
        perplexities.append((perplexity_baseline_mean / mean(flatten(perplexity_by_layer[layer_idx]))) * 100)
        energy_per_token.append(mean(flatten(energy_by_layer[layer_idx])) / energy_baseline_mean * 100)
        time_per_token.append(mean(flatten(time_by_layer[layer_idx])) / time_baseline_mean * 100)
        memory_allocated.append(mean(flatten(memory_by_layer[layer_idx])) / memory_baseline_mean * 100)

    return {
      "report_name": report_name,
      "layers": layers,
      "perplexity": perplexities,
      "energy_per_token": energy_per_token,
      "time_per_token": time_per_token,
      "memory_allocated": memory_allocated
    }

def main():
    parser = argparse.ArgumentParser(description="Generates graphs of Llama model testing.")
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Optional output dir. Defaults to "./generated/{first-metric-name}".')
    parser.add_argument(
        '--input-paths',
        nargs='+',
        type=str,
        help='Required list of input CSV file paths (space-separated).'
    )
    args = parser.parse_args()
    input_paths: List[string] = args.input_paths
    metrics_names = [re.search(r'([^/]+)\.csv$', input_path).group(1)
                     for input_path in input_paths]
    base_output_path = './generated/' + metrics_names[0] if args.output_dir is None \
        else args.output_dir
    os.makedirs(base_output_path, exist_ok=True)

    reports = [generate_stat_report(input_path, report_name) for (input_path, report_name) in zip(input_paths, metrics_names)]

    # Display tables and plots
    plt.figure()
    for report in reports:
        perplexity = report['perplexity']
        report_name = report['report_name']
        plt.plot(report['layers'], perplexity, label=report_name)
        print(f'\nPerplexity for {report_name}')
        print(pd.DataFrame({"Layer Index": report['layers'], "Perplexity (%)": perplexity}))
    plt.title("Per Layer Perplexity After Attention Head Pruning")
    plt.xlabel("Layer Index")
    plt.ylabel("Perplexity Compared to Baseline (Percent)")
    plt.legend()
    plt.savefig(f'{base_output_path}/perplexity.png')
    plt.show()

    plt.figure()
    for report in reports:
        energy_per_token = report['energy_per_token']
        report_name = report['report_name']
        plt.plot( energy_per_token, label=report_name)
        print(f'\nEnergy per Token for {report_name}')
        print(pd.DataFrame({"Layer Index": report['layers'], "Energy Usage (%)": energy_per_token}))
    plt.title("Energy Usage Per Token After Attention Head Pruning")
    plt.xlabel("Layer Index")
    plt.ylabel("Energy Usage Compared to Baseline (Percent)")
    plt.legend()
    plt.savefig(f'{base_output_path}/energy.png')
    plt.show()

    plt.figure()
    for report in reports:
        time_per_token = report['time_per_token']
        report_name = report['report_name']
        plt.plot(report['layers'], time_per_token, label=report_name)
        print(f'\nInference Time for {report_name}')
        print(pd.DataFrame({"Layer Index": report['layers'], "Inference Time (%)": time_per_token}))
    plt.title("Inference Time Per Token After Attention Head Pruning")
    plt.xlabel("Layer Index")
    plt.ylabel("Inference Time Compared to Baseline (Percent)")
    plt.legend()
    plt.savefig(f'{base_output_path}/evaluation.png')
    plt.show()

    plt.figure()
    for report in reports:
        memory_allocated = report['memory_allocated']
        report_name = report['report_name']
        plt.plot(report['layers'], memory_allocated, label=report_name)
        print(f'Memory Allocation for {report_name}')
        print(pd.DataFrame({"Layer Index": report['layers'], "Allocated Memory (%)": memory_allocated}))

    plt.ylim(99.63, 99.64)  # Adjust the y-axis limits to zoom into the range of interest
    plt.title("Allocated Memory for Model After Attention Head Pruning")
    plt.xlabel("Layer Index")
    plt.ylabel("Allocated Memory Compared to Baseline (Percent)")
    plt.savefig(f'{base_output_path}/allocated.png')
    plt.show()

if __name__ == '__main__':
    main()
