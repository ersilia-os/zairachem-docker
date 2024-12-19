# 04. Estimators

## Run from Docker

```bash


```

## Run from system 

```bash
conda create -n zairaestimate python=3.12
cd 04_estimate
python pip install -e .
python zairaestimate/run.py
```

## High level overview
The estimators pipeline is composed of a few selected autoML techniques that will be trained on all the available descriptors. The structure of each estimator is the same, summarised below:
1. `estimators/pipe.py` will call each estimator sequentially. Make sure to include the desired estimators there if you add a new one and update the available estimators in the `vars.py` file on ZairaBase.
2. Each estimator has its individual folder, in where five files are available:
- `pipe.py` is the organiser of the estimator, calling sequentially on estimate, assemble and performance
- `estimate.py` is where the actual model training takes place. As a template, this file should contain a Fitter and Predictor class, for fitting and running predictions, respectively. Some common functions for the Fitter and Predictor are specified in the BaseEstimatorIndividual available in `estimators/base.py`. Do not modify the base as it is used across estimators. The Fitter and Predictor class should contain only minimal changes, if at all. They simply call on the Regressor and Classifier tasks from the actual estimator file (see below). Then, the Fitter and Predictor are used via the IndividualEstimator (per each descriptor) and Estimator class (collating all descriptors). At the end of the estimate process, a folder with {estimator_name}_estimator should be created in the `model_dir/estimators` folder, with the trained model. Specify the folder name in the ESTIMATORS_FAMILY_SUBFOLDER variable
- `estimator_name.py` contains the actual functions to run the estimator. Generally speaking, it needs a Classifier and Regressor class where the X and y are required parameters of the class. X and y are defined by the BaseEstimatorIndividual and should only be modified (for example to a suitable format for the model) inside the Regressor or Classifier class, but not at the Fitter level. The minimal functions of the Classifier and Regressor must be fit, save and load. A ClassifierArtifact and RegressorArtifact, with the model as the only required parameter of the class, contain the predictor functions (predict in the case of the regressor and both predict and predict_proba in the case of the classifier). Do not modify the output structure of the Artifacts as it is prepared for downstream processing. Additional classes can be added to this file if needed to run the model.
- `assemble.py` loads the results of the training (calculated on the train set itself) and appends it to the molecule list, both mapped and unmapped. Do not modify.
- `performance.py` calculates the performance metrics (auc_roc, precision and recall for classifiers and r2 and mean_squared_error for the regressors) and saves a clf_report or reg_report as a .json file. Do not modify.

### Differences with ZairaChem v1
We have reformatted the entire estimator pipeline. Previously, different descriptors were calculated with different methods, including AutoGluon, Flaml, KerasTuner and MolMap. Currently we are using both FLAML and KerasTuner with all descriptors. Fitting from the 2d descriptors (PCA, UMAP, LolP) is not currently being done.

## To Solve
- Why the evaluate pipeline runs the evaluator twice if not_predict? it generates two files that are basically the same at this point. 
- When to use heavy_fit with flaml? and with which fold? currently the option exists but is set to simple by default
- Do we trust that all descriptors that should be treated are treated or we want a sanity check and if the treated.h5 file does not exist for a descriptor in the treated list we don't use it?