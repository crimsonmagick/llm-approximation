import argparse
import csv
from statistics import mean, stdev, median

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculates baseline metrics.")
    parser.add_argument(
        '--input-path',
        type=str,
        help='Input path for csv file, required'
    )
    args = parser.parse_args()
    input_path = args.input_path
    energies = []
    
    # # Read and process the CSV file
    with open(input_path, mode='r') as file:
        measurements_per_layer = 20
        reader = csv.DictReader(file)
        
        for row in reader:
            label = row['label']
            idx = int(label[len(label) - 1])
            mj_per_token = float(row['average_energy_per_token_mj'])
            energies.append(mj_per_token)
    
    energies.sort()
    
    energy_max = max(energies)
    energy_min = min(energies)
    energy_mean = mean(energies)
    energy_median = median(energies)
    energy_std_dev = stdev(energies)

    print(energies)
    print(f"energy_mean={energy_mean}\n"
          f"energy_median={energy_median}\n"
          f"energy_min={energy_min}\n"
          f"energy_max={energy_max}\n"
          f"range={energy_max - energy_min}\n"
          f"energy_std_dev={energy_std_dev}")
