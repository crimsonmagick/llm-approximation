import argparse
import csv
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generates graphs of Llama model testing.")
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./generated',
        help='Optional output dir. Defaults to the "./output_dir".')
    parser.add_argument(
        '--input-path',
        type=str,
        help='Input path for csv file, required'
    )
    args = parser.parse_args()
    # label,layer_idx,head_idxs,perplexity,average_energy_per_token_mj,average_time_per_token_ms,allocated_memory
    with open(args.input_path, mode='r') as file:
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
                perplexity_sum_by_layer[layer_idx] = (perplexity_sum_by_layer[layer_idx] + test_perplexity) / 2
            else:
                perplexity_sum_by_layer[layer_idx] = test_perplexity
            
            if layer_idx in energy_sum_by_layer:
                energy_sum_by_layer[layer_idx] = (energy_sum_by_layer[layer_idx] + test_energy) / 2
            else:
                energy_sum_by_layer[layer_idx] = test_energy
            
            if layer_idx in time_sum_by_layer:
                time_sum_by_layer[layer_idx] = (time_sum_by_layer[layer_idx] + test_time) / 2
            else:
                time_sum_by_layer[layer_idx] = test_time
            
            if layer_idx in memory_sum_by_layer:
                memory_sum_by_layer[layer_idx] = (memory_sum_by_layer[layer_idx] + test_memory) // 2
            else:
                memory_sum_by_layer[layer_idx] = test_memory
        
        # Normalize and store final values
        for layer_idx in perplexity_sum_by_layer:
            perplexities.append((baseline_perplexity / perplexity_sum_by_layer[layer_idx]) * 100)
            energy_per_token.append((energy_sum_by_layer[layer_idx]) / baseline_energy  * 100)
            time_per_token.append((time_sum_by_layer[layer_idx] / baseline_time) * 100)
            memory_allocated.append((memory_sum_by_layer[layer_idx] / baseline_memory) * 100)
        
        # Accuracy (Perplexity)
        plt.figure()
        plt.plot(perplexities)
        plt.title("Per Layer Perplexity After Attention Head Pruning")
        plt.xlabel("Layer Index")
        plt.ylabel("Perplexity Compared to Baseline (Percent)")
        plt.show()
        
        # Energy Usage
        plt.figure()
        plt.plot(energy_per_token)
        plt.title("Energy Usage Per Token After Attention Head Pruning")
        plt.xlabel("Layer Index")
        plt.ylabel("Energy Usage Compared to Baseline (Percent)")
        plt.show()
        
        # Evaluation time Per Token
        plt.figure()
        plt.plot(time_per_token)
        plt.title("Evaluation Time Per Token After Attention Head Pruning")
        plt.xlabel("Layer Index")
        plt.ylabel("Evaluation Time Compared to Baseline (Percent)")
        plt.show()
        
        # Memory Allocation
        plt.figure()
        plt.plot(memory_allocated)
        plt.ylim(99.63, 99.64)  # Adjust the y-axis limits to zoom into the range of interest
        plt.title("Allocated Memory for Model After Attention Head Pruning")
        plt.xlabel("Layer Index")
        plt.ylabel("Allocated Memory Compared to Baseline (Percent)")
        plt.show()

