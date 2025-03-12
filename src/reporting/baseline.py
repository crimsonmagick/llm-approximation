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
    num_batch = 10
    batch_energies_sorted = list()
    for batch_idx in range(num_batch):
      batch_energies_sorted.append(list())

    # # Read and process the CSV file
    with open(input_path, mode='r') as file:
        measurements_per_layer = 20
        reader = csv.DictReader(file)

        for row in reader:
            label = row['label']
            idx = int(label[len(label) - 1])
            mj_per_token = float(row['average_energy_per_token_mj'])
            batch_energies_sorted[idx].append(mj_per_token)

    for idx in range(num_batch):
      energies_sorted = batch_energies_sorted[idx]
      energies_sorted.sort()

      energy_max = max(energies_sorted)
      energy_min = min(energies_sorted)
      energy_mean = mean(energies_sorted)
      energy_median = median(energies_sorted)
      energy_std_dev = stdev(energies_sorted)

      print(energies_sorted)
      print(f"batch {idx}: energy_mean={energy_mean}\n"
            f"batch {idx}: energy_median={energy_median}\n"
            f"batch {idx}: energy_min={energy_min}\n"
            f"batch {idx}: energy_max={energy_max}\n"
            f"batch {idx}: range={energy_max - energy_min}\n"
            f"batch {idx}: energy_std_dev={energy_std_dev}")

    aggregate_energies_sorted = [energy for batch_list in batch_energies_sorted for energy in batch_list]
    energy_max = max(aggregate_energies_sorted)
    energy_min = min(aggregate_energies_sorted)
    energy_mean = mean(aggregate_energies_sorted)
    energy_median = median(aggregate_energies_sorted)
    energy_std_dev = stdev(aggregate_energies_sorted)
    print(f"aggregate_energy_mean={energy_mean}\n"
          f"aggregate_energy_median={energy_median}\n"
          f"aggregate_energy_min={energy_min}\n"
          f"aggregate_energy_max={energy_max}\n"
          f"aggregate_range={energy_max - energy_min}\n"
          f"aggregate_energy_std_dev={energy_std_dev}")
