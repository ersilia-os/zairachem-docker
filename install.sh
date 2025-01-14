#!/bin/bash

# Initialize Conda
eval "$(conda shell.bash hook)"

# List of folders containing the pyproject.toml files
folders=(
    "01_setup"
    "02_describe"
    "03_treat"
    "04_estimate"
    "05_pool"
    "06_report"
    "07_finish"
)

# Loop through each folder and create the corresponding Conda environment
for folder in "${folders[@]}"; do
    toml_file="$folder/pyproject.toml"

    if [ -f "$toml_file" ]; then
        # Extract the environment name from the .toml file
        env_name=$(grep -Po '(?<=name = ")[^"]*' "$toml_file")

        if [ -n "$env_name" ]; then
            echo "Creating Conda environment: $env_name from $toml_file"

            # Create the Conda environment using the .toml file
            conda create -n "$env_name" python=3.12 -y
            echo "Environment $env_name created successfully!"
            conda activate "$env_name"
            cd "$folder" || exit
            pip install -e .
            cd - || exit
        else
            echo "Error: No environment name found in $toml_file. Skipping..."
        fi
    else
        echo "Warning: $toml_file not found. Skipping..."
    fi
done

echo "All specified environments have been processed!"