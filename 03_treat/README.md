# 02 Molecular descriptors

This step treats the descriptors that require it and applies dimensionality reduction techniques

## Run from Docker

```bash


```

## Run from system 

```bash
conda create -n zairatreat python=3.12
conda activate zairatreat
cd 03_treat
python pip install -e .
python zairatreat/run.py
```

## High level overview
This module only contains 3 steps:
1. Treat descriptors to eliminate columns that are mostly NaN, impute missing values and remove variance (all descriptors). Those files are saved as .joblib files in each descriptor folder. 
2. Scale descriptors that require it (i.e rdkit fingerprints, mordred...):The descriptors to be treated are specified in the `zairabase/vars.py`. If a new descriptor is passed in a `config.json` but not specified in the scaled list, it will not be scaled. Those files are stored in the respective descriptor's folder. The final descriptors file is named `treated.h5` (scaled or not))
3. Calculate 2D and 4D descriptors from the reference descriptor using PCA, UMAP and LOLP. The transformed PCA, LOLP and UMAP are stored in the root of the `/descriptors` folder.

### Differences with ZairaChem v1

A few improvements have been made to the Describe step:
- All descriptors are calculated at the first step (Raw Descriptors) and will undergo NanFiltering, Imputing and VarianceFiltering. Only those specified in the SCALED_DESCRIPTORS list in the `00_zaira_base/vars.py` file will be scaled. This is designed to be user-error proven. If a descriptor is added through a config file, it will be calculated, even if it has never been specified to ZairaChem before. The only thing a user should decide is whether to scale them or not. We can also add a long list of Scaled descriptors from the Ersilia Model Hub to make it more error safe. 
- The FPSIM2 folder has been eliminated as FPSIM is now pip installable (since v0.4)
- The label used for the PCA, UMAP and LOLP is the Bin column, not the AUX