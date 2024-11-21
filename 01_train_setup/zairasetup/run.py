import argparse
from setup.training import TrainSetup

def run(input_file, output_dir=None, threshold=None, direction=None, parameters=None):
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print(f'Direction {direction}')
    print(f'Threshold {threshold}')
    ts = TrainSetup(input_file, output_dir, time_budget=120, task="classification", threshold=threshold, direction=direction, parameters=parameters)
    if ts.is_done():
        return
    ts.setup()

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Process an input file and output directory.")

    # Add arguments
    parser.add_argument("--input_file", "-i", required=True, help="Path to the input file.")
    parser.add_argument("--model_dir", "-o", required=False, help="Path to the directory where the model will be stored.")
    parser.add_argument("--cutoff", "-c", required=False, help="Cutoff to binarize data, i.e. to separate actives and inactives. By convention, actives = 1 and inactives = 0, check 'direction'.")
    parser.add_argument("--direction", "-d", required=False, help="Direction of the actives: 'high' means that high values are actives, 'low' means that low values are actives.")
    parser.add_argument("--parameters", "-p", required=False, help="Path to parameters file in JSON format.")
    # Parse arguments
    args = parser.parse_args()

    # Access arguments
    input_file = args.input_file
    output_dir = args.model_dir
    threshold = args.cutoff
    direction = args.direction
    parameters = args.parameters

    run(input_file, output_dir, threshold, direction, parameters)