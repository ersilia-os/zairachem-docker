# 01 Train Setup

This step sets up the folder infrastructure required to run ZairaChem

## Run from Docker

```bash


```

## Run from system 

```bash
conda create -n zairasetup python=3.12
conda activate zairasetup
cd 01_setup
pip install -e .
python zairasetup/run_fit.py --input_file $INPUT_FILE --model_dir $OUTPUT_DIR --cutoff 0.1 --direction low 
python zairasetup/run_predict.py --input_file $INPUT_FILE --model_dir $MODEL_DIR --output_dir $OUTPUT_DIR 
```
For fitting, a `.csv` file with minimum a SMILES column and a values column is required. For predicting, only a `.csv` column is required, but if values are passed, the full report (AUC values, ROC curves, etc) will be generated at predict time as well. IF the model was trained on already binarised data, the data for prediction must also be binarised already. No direction or cut-off can be specified at predict time.
The task is expressely set at classification (see sections below) but an argument can be added to allow regression. The cutoff and direction arguments are only required if the data is not already binarised.

## High level overview

This is the first step of the ZairaChem pipeline. It comprises the following:
1. Initialisation: the output directory and all necessary subdirectories are created, along with session initialisation (`session.json`) and parameter setting. The parameters are automatically defined, but the option to pass a parameters file remains (see `zairabase/config` for an example). The input data is copied into the root of the output directory as `raw_input` for reference.
2. Input Normalisation: the input file is automatically processed to identify the column containing the SMILES (it does not have to be labelled as "smiles" necessarily) and all the other relevant columns (molecular identifier, values, qualifier, date and group). If more than one value column is passed (for example, a binarised column and a continuous experimental output), the pipeline will stop as it won't know which column to use. If you want to enforce the use of the binarised column please label it as "bin". The input is also processed to eliminate duplicate SMILES (the median value will be kept for each duplicated SMILES), and a predefined ID will be given to each unique molecule. A mapping file refering to the original index and the new index will allow the developer to identify which molecules have been deduplicated. This step ends up providing four files: `compounds.csv` which contains the compound ID (CID) and SMILES, `values.csv` which contains the values associated to each CID, and, if available, the qualifier, `mapping.csv` which maps the original index with the new index and CID, and `input_schema.json` which saves the original column names used
3. Standardisation: the input smiles are standardised using the ChEMBL molecule standardiser, and saved in the `compounds_std.csv` file, which contains a smiles and smiles_std column.
4. Folding: three 5-fold divisions are created for future crossvalidation: (a) random folds, (b) scaffold folds based on MurckoScaffolds so that each fold contains different scaffolds and (c) cluster folds using a KMeans approach on Morgan Fingerprints. The folds to which each smiles belong are saved in a `folds.csv` file. 
6. Tasks: classification and regression. At the current moment only classification is allowed. If the data is not binarised, it will automatically be converted to [0,1] according to the direction and threshold set by the user. In the case where the user has passed the values as a binary classification but one SMILES is repeated with 0 and 1 respectively, it will be eliminated from the dataset. the tasks are saved in a `tasks.csv`file, but currently only the selected cut-off (expert if available, one of the percentiles if not) will be saved. Regression data is not calculated.
7. Merge: unifies the standardised smiles, folds and tasks into a final `data.csv` file
8. Clean: cleans up intermediate files, leaving only the `data.csv` for downstream processing along with the mapping and schemas.

### Differences with ZairaChem v1
The Augmentation capabilities were not in use so they are not implemented in this version. Please check the legacy code of ZairaChem for more information (zairachem/augmentation).
ZairaChem v2, like v1, is focused in Classifiers. The option to train Regressors is available as a placeholder for future developments. For Regressor training, the pipeline is prepared to do several transformations in Y (install flaml[automl] first), but no column is currently saved in the clean step, which will need to be modified.