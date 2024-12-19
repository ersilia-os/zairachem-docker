# 05. Pooling

## Run from Docker

```bash


```

## Run from system 

```bash
conda create -n zairapool python=3.12
cd 05_pool
python pip install -e .
python zairapool/run.py
```

## High level overview
In this case, the Pooling pipeline consists of a single step to create an ensemble model that will be stored in the pool folder. It follows the same structure as the estimators, with a Fitter and Predictor that run the corresponding Classifier and Regressor classes.

### Differences with ZairaChem v1
Currently only bagging is implemented, though a placeholder for other types of pooling is in place.