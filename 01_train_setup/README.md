# 01 Train Setup

This step sets up the folder infrastructure required to run ZairaChem

### Differences with ZairaChem v1
The Augmentation capabilities were not in use so they are not implemented in this version. Please check the legacy code of ZairaChem for more information (zairachem/augmentation).
ZairaChem v2, like v1, is focused in Classifiers. The option to train Regressors is available as a placeholder for future developments.

### Questions:
- Model dir or Output dir? what is the best naming here?
- is_values_column works as expected?
- Ensemble modes: only bagging available, do we want to include others? ZairaBase still has option for Blending, consider removing?
- Lazy: the lazy functionality is not being used, consider removing entirely?
- Pandas: ZairaBase requires pandas installed, but this could actually be removed
- The predict functionality will have its own package, should it be removed from here? (predict.py and ForPredict classes in tasks.py)
- The "_MAX_EMPTY" parameter in tasks.py is not being used, remove completely?
- The utils.py and tasks.py contain a number of regression-related questions, what do we do with them? I have set the imports inside the class to avoid unnecessary installs at the moment
- Add proper documentation for each file?

## Run from Docker

```bash


```

## Run from system 

```bash
conda create -n zairasetup python=3.12
cd 01_train_setup
python pip install -e .
python zairasetup/run.py --input_file $INPUT_FILE --model_dir $OUTPUT_DIR --cutoff 0.1 --direction low 
```

The cutoff and direction arguments are only required if the data is not already binarised.
