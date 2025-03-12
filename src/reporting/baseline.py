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

    # Read and process the CSV file
    with open(input_path, mode='r') as file:
        measurements_per_layer = 20
        reader = csv.DictReader(file)

        energies_sorted = []

        for row in reader:
            energies_sorted.append(float(row['average_energy_per_token_mj']))
        energies_sorted.sort()

        energy_max = max(energies_sorted)
        energy_min = min(energies_sorted)
        energy_mean = mean(energies_sorted)
        energy_median = median(energies_sorted)
        energy_std_dev = stdev(energies_sorted)

        print(energies_sorted)
        print(f"energy_mean={energy_mean}\n"
              f"energy_median={energy_median}\n"
              f"energy_min={energy_min}\n"
              f"energy_max={energy_max}\n"
              f"range={energy_max - energy_min}\n"
              f"energy_std_dev={energy_std_dev}")
