import argparse
import csv

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
        for row in reader:
            print(row)