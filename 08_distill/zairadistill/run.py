import argparse
import os
from distill import Distiller

def run(model_dir, output_path):
    d = Distiller(model_dir, output_path)
    d.run()

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Process an model directory and output path.")

    # Add arguments
    parser.add_argument("--model_dir", "-m", required=True, help="Path to the directory of the fitted ZairaChem model.")
    parser.add_argument("--output_path", "-o", required=False, help="Path to where the ONNX model file will be stored.")
    # Parse arguments
    args = parser.parse_args()

    # Access arguments
    model_dir = args.model_dir
    output_path = args.output_path
    if output_path is not None:
        output_path = os.path.join(model_dir, "distill", "distilled_model.onnx")

    run(model_dir, output_path)