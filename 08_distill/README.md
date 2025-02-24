# 08. Distilling

## Run from Docker

```bash


```

## Run from system 

```bash
conda create -n zairadistill python=3.10
conda activate zairadistill
cd 08_distill
python pip install -e .
python zairadistill/run.py --model_dir $MODEL_DIR --output_path $OUTPUT_PATH
```

## High level overview

### Differences with ZairaChem v1
