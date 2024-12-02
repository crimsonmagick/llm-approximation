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
        test_time = float(baseline['average_time_per_token_ms'])
        test_memory = float(baseline['allocated_memory'])
        
        perplexity_sum_by_layer = dict()
        energy_sum_by_layer = dict()
        perplexities = []
        energy_per_token = []
        for row in reader:
            layer_idx = row['layer_idx']
            test_perplexity = float(row['perplexity'])
            test_energy = float(row['average_energy_per_token_mj'])
            test_time = float(row['average_time_per_token_ms'])
            test_memory = float(row['allocated_memory'])
            
            if layer_idx in perplexity_sum_by_layer:
                perplexity_sum_by_layer[layer_idx] = (perplexity_sum_by_layer[layer_idx] + test_perplexity) / 2
            else:
                perplexity_sum_by_layer[layer_idx] = float(row['perplexity'])
        for p in perplexity_sum_by_layer.values():
            perplexities.append((baseline_perplexity / p) * 100)
        
        plt.plot(perplexities)
        plt.title("Per Layer Accuracy After Attention Head Pruning")
        plt.xlabel("Layer Index")
        plt.ylabel("Accuracy (Percent)")
        plt.show()
