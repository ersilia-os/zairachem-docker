#!/bin/bash

# Activate Conda
eval "$(conda shell.bash hook)"

# Initialize variables
input_file=""
model_dir=""
cutoff=""
direction=""
parameters=""
clean=""
flush=""
anonymize=""

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i|--input_file) input_file="$2"; shift ;;
        -m|--model_dir) model_dir="$2"; shift ;;
        -c|--cutoff) cutoff="$2"; shift ;;
        -d|--direction) direction="$2"; shift ;;
        -p|--parameters) parameters="$2"; shift ;;
        --clean) clean="$2"; shift ;;
        --flush) flush="$2"; shift ;;
        --anonymize) anonymize="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check required arguments
if [ -z "$input_file" ]; then
    echo "Error: --input_file (-i) is required."
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate zairasetup

# Start building the command for run_fit.py
command="python 01_setup/zairasetup/run_fit.py -i '$input_file' -m '$model_dir'"

# Add optional arguments only if provided
if [ -n "$cutoff" ]; then
    command="$command -c '$cutoff'"
fi
if [ -n "$direction" ]; then
    command="$command -d '$direction'"
fi
if [ -n "$parameters" ]; then
    command="$command -p '$parameters'"
fi

# Run the command
eval "$command"

#python 01_setup/zairasetup/run_fit.py -i "$input_file" -m "$model_dir" -c "$cutoff" -d "$direction" -p "$parameters"

conda activate zairadescribe
python 02_describe/zairadescribe/run.py

conda activate zairatreat
python 03_treat/zairatreat/run.py

conda activate zairaestimate
python 04_estimate/zairaestimate/run.py

conda activate zairapool
python 05_pool/zairapool/run.py

conda activate zairareport
python 06_report/zairareport/run.py

conda activate zairafinish
python 07_finish/zairafinish/run.py --clean "$clean" --flush "$flush" --anonymize "$anonymize"