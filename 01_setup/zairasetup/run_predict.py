import argparse
from setup.prediction import PredictSetup

def run(input_file, model_dir, output_dir=None):
    ps = PredictSetup(input_file, model_dir, output_dir, time_budget=120)
    if ps.is_done():
        return
    ps.setup()

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Process an input file and output directory.")

    # Add arguments
    parser.add_argument("--input_file", "-i", required=True, help="Path to the input file.")
    parser.add_argument("--model_dir", "-m", required=True, help="Path to the directory of the fitted model.")
    parser.add_argument("--output_dir", "-o", required=False, help="Path to the directory where the model will be stored.")
    # Parse arguments
    args = parser.parse_args()

    # Access arguments
    input_file = args.input_file
    model_dir = args.model_dir
    output_dir = args.output_dir

    run(input_file, model_dir, output_dir)