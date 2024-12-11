# 01 Train Setup

This step sets up the folder infrastructure required to run ZairaChem

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

## High level overview

This is the first step of the ZairaChem pipeline. It comprises the following:
1. Initialisation: the output directory and all necessary subdirectories are created, along with session initialisation (`session.json`) and parameter setting. The parameters are automatically defined, but the option to pass a parameters file remains. The input data is copied into the root of the output directory as `raw_input` for reference.
2. Input Normalisation: the input file is automatically processed to identify the column containing the SMILES (it does not have to be labelled as "smiles" necessarily) and all the other relevant columns (molecular identifier, values, qualifier, date and group). If more than one value column is passed (for example, a binarised column and a continuous experimental output), only the one that appears first in the dataframe will be considered. If you want to enforce the use of the binarised column please label it as                   "bin". Column names used are saved in the `input_schema.json`. The input is also processed to eliminate duplicate SMILES (the median value will be kept for each duplicated SMILES), and a predefined ID will be given to each unique molecule. A mapping file refering to the original index and the new index will allow the developer to identify which molecules have been deduplicated. This step ends up providing a `data0.csv` file which contains the following fields: compound_id,smiles,assay_id,qualifier,value.
3. Standardisation: the input smiles are standardised using the ChEMBL molecule standardiser, and saved in the `compounds_std.csv` file
4. Folding: three 5-fold divisions are created for future crossvalidation: (a) random folds, (b) scaffold folds based on MurckoScaffolds so that each fold contains different scaffolds and (c) cluster folds using a KMeans approach on Morgan Fingerprints. The folds to which each smiles belong are saved in a `folds.csv` file. 
6. Tasks: classification and regression. At the current moment only classification is allowed. If the data is not binarised, it will automatically be converted to [0,1] according to the direction and threshold set by the user. In the case where the user has passed the values as a binary classification but one SMILES is repeated with 0 and 1 respectively, it will be eliminated from the dataset.  
7. Merge: unifies the standardised smiles, folds and tasks.


### Differences with ZairaChem v1
The Augmentation capabilities were not in use so they are not implemented in this version. Please check the legacy code of ZairaChem for more information (zairachem/augmentation).
ZairaChem v2, like v1, is focused in Classifiers. The option to train Regressors is available as a placeholder for future developments.

### Questions:
- Model dir or Output dir? what is the best naming here?
- Schema.py: sniff size is to avoid loading the entire dataframe, why 1000 if later only 100 is checked? Max Empty is not being used?
- is_values_column works as expected?
- Smiles dedupe (file.py): if the same smiles has 1 and 0 in a binary case? it will give what?
- Assay table: purpose not clear
- Values table: is it really needed? Data0 already contains the information
- Qualifier data is not used ever (see _get_data in tasks.py)
- Why is the regression called when continuous data is passed, even if it is for a classifier?
- Ensemble modes: only bagging available, do we want to include others? ZairaBase still has option for Blending, consider removing?
- Lazy: the lazy functionality is not being used, consider removing entirely?
- Pandas: ZairaBase requires pandas installed, but this could actually be removed
- The predict functionality will have its own package, should it be removed from here? (predict.py and ForPredict classes in tasks.py)
- The "_MAX_EMPTY" parameter in tasks.py is not being used, remove completely?

- The utils.py and tasks.py contain a number of regression-related questions, what do we do with them? I have set the imports inside the class to avoid unnecessary installs at the moment
- Add proper documentation for each file?

