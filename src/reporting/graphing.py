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
    
    with open(args.input_path, mode='r') as file:
        reader = csv.DictReader(file)
        baseline = next(reader)
        baseline_perplexity = float(baseline['perplexity'])
        perplexity_sum_by_layer = dict()
        perplexities = []
        for row in reader:
            layer_idx = row['layer_idx']
            if layer_idx in perplexity_sum_by_layer:
                perplexity_sum_by_layer[layer_idx] = (perplexity_sum_by_layer[layer_idx] + float(row['perplexity'])) / 2
            else:
                perplexity_sum_by_layer[layer_idx] = float(row['perplexity'])
        for p in perplexity_sum_by_layer.values():
            perplexities.append((baseline_perplexity / p ) * 100)
        plt.plot(perplexities)
        plt.title("Per Layer Accuracy After Attention Head Pruning")
        plt.xlabel("Layer Index")
        plt.ylabel("Accuracy (Percent)")
        plt.show()
