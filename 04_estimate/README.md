# 03. Estimators

## Run from Docker

```bash


```

## Run from system 

```bash
conda create -n zairaestimate python=3.12
cd 03_estimate
python pip install -e .
python zairaestimate/run.py
```

## High level overview

### Differences with ZairaChem v1

## To Solve
- Why the evaluate pipeline runs the evaluator twice if not_predict? it generates two files that are basically the same at this point. 
- When to use heavy_fit with flaml? and with which fold? currently the option exists but is set to simple by default
- Do we trust that all descriptors that should be treated are treated or we want a sanity check and if the treated.h5 file does not exist for a descriptor in the treated list we don't use it?