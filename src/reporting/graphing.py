import argparse
import csv
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generates graphs of Llama model testing.")
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./generated',
        help='Optional output dir. Defaults to "./generated".')
    parser.add_argument(
        '--input-path',
        type=str,
        help='Input path for csv file, required'
    )
    args = parser.parse_args()
    input_path = args.input_path
    out_subpath = re.search(r'([^/]+)\.csv$', input_path).group(1)
    base_output_path = args.output_dir + '/' + out_subpath
    os.makedirs(base_output_path, exist_ok=True)

    # Read and process the CSV file
    with open(input_path, mode='r') as file:
        measurements_per_layer = 20
        reader = csv.DictReader(file)

        baseline = next(reader)
        baseline_perplexity = float(baseline['perplexity'])
        baseline_energy = float(baseline['average_energy_per_token_mj'])
        baseline_time = float(baseline['average_time_per_token_ms'])
        baseline_memory = float(baseline['allocated_memory'])

        perplexity_sum_by_layer = dict()
        energy_sum_by_layer = dict()
        time_sum_by_layer = dict()
        memory_sum_by_layer = dict()

        perplexities = []
        energy_per_token = []
        time_per_token = []
        memory_allocated = []

        for row in reader:
            layer_idx = row['layer_idx']
            test_perplexity = float(row['perplexity'])
            test_energy = float(row['average_energy_per_token_mj'])
            test_time = float(row['average_time_per_token_ms'])
            test_memory = int(row['allocated_memory'])

            if layer_idx in perplexity_sum_by_layer:
                perplexity_sum_by_layer[layer_idx] = perplexity_sum_by_layer[layer_idx] + test_perplexity
            else:
                perplexity_sum_by_layer[layer_idx] = test_perplexity

            if layer_idx in energy_sum_by_layer:
                energy_sum_by_layer[layer_idx] = energy_sum_by_layer[layer_idx] + test_energy
            else:
                energy_sum_by_layer[layer_idx] = test_energy

            if layer_idx in time_sum_by_layer:
                time_sum_by_layer[layer_idx] = time_sum_by_layer[layer_idx] + test_time
            else:
                time_sum_by_layer[layer_idx] = test_time

            if layer_idx in memory_sum_by_layer:
                memory_sum_by_layer[layer_idx] = memory_sum_by_layer[layer_idx] + test_memory
            else:
                memory_sum_by_layer[layer_idx] = test_memory

        # Normalize and store final values
        for layer_idx in sorted(perplexity_sum_by_layer.keys(), key=lambda x: int(x)):
            perplexities.append((baseline_perplexity / perplexity_sum_by_layer[layer_idx] / measurements_per_layer) * 100)
            energy_per_token.append((energy_sum_by_layer[layer_idx] / baseline_energy / measurements_per_layer)* 100)
            time_per_token.append((time_sum_by_layer[layer_idx] / baseline_time / measurements_per_layer) * 100)
            memory_allocated.append((memory_sum_by_layer[layer_idx] / baseline_memory / 2) * 100)

        # Create dataframes for the tables
        layers = list(sorted(perplexity_sum_by_layer.keys(), key=lambda x: int(x)))
        perplexity_table = pd.DataFrame({"Layer Index": layers, "Perplexity (%)": perplexities})
        energy_table = pd.DataFrame({"Layer Index": layers, "Energy Usage (%)": energy_per_token})
        time_table = pd.DataFrame({"Layer Index": layers, "Evaluation Time (%)": time_per_token})
        memory_table = pd.DataFrame({"Layer Index": layers, "Allocated Memory (%)": memory_allocated})

        # Display tables and plots
        print("\nPer Layer Perplexity Table")
        print(perplexity_table)
        plt.figure()
        plt.plot(perplexities)
        plt.title("Per Layer Perplexity After Attention Head Pruning")
        plt.xlabel("Layer Index")
        plt.ylabel("Perplexity Compared to Baseline (Percent)")
        plt.savefig(f'{base_output_path}/perplexity.png')
        plt.show()

        print("\nEnergy Usage Table")
        print(energy_table)
        plt.figure()
        plt.plot(energy_per_token)
        plt.title("Energy Usage Per Token After Attention Head Pruning")
        plt.xlabel("Layer Index")
        plt.ylabel("Energy Usage Compared to Baseline (Percent)")
        plt.savefig(f'{base_output_path}/energy.png')
        plt.show()

        print("\nEvaluation Time Table")
        print(time_table)
        plt.figure()
        plt.plot(time_per_token)
        plt.title("Evaluation Time Per Token After Attention Head Pruning")
        plt.xlabel("Layer Index")
        plt.ylabel("Evaluation Time Compared to Baseline (Percent)")
        plt.savefig(f'{base_output_path}/evaluation.png')
        plt.show()

        print("\nAllocated Memory Table")
        print(memory_table)
        plt.figure()
        plt.plot(memory_allocated)
        plt.ylim(99.63, 99.64)  # Adjust the y-axis limits to zoom into the range of interest
        plt.title("Allocated Memory for Model After Attention Head Pruning")
        plt.xlabel("Layer Index")
        plt.ylabel("Allocated Memory Compared to Baseline (Percent)")
        plt.savefig(f'{base_output_path}/allocated.png')
        plt.show()
