# 02 Molecular descriptors

This step calculates several descriptors for the molecules

## Run from Docker

```bash


```

## Run from system 

```bash
conda create -n zairadescribe python=3.12
cd 02_describe
python pip install -e .
python zairadescribe/run.py
```

## High level overview
This module only contains 3 steps:
1. Calculate Raw descriptors. Using only Ersilia Models, it calculates the descriptors specified in the `data/parameters.json` file. Those come either from a config.json or are pulled from `vars.py`. If the reference descriptor is not already included in the list, it will be calculated additionally. Descriptors are stored in the `/descriptors` folder under their respective names.
2. Treat descriptors that require it (i.e rdkit fingerprints, mordred...): imputes for missing values and normalises them. The descriptors to be treated are specified in the `zairabase/vars.py`. If a new descriptor is passed in a `config.json` but not specified in the treated list, it will not be treated. Treated files are stored in the respective descriptor's folder.
3. Calculate 2D and 4D descriptors from the reference descriptor using PCA, UMAP and LOLP. The transformed PCA, LOLP and UMAP are stored in the root of the `/descriptors` folder.

### Differences with ZairaChem v1

A few improvements have been made to the Describe step:
- All descriptors must be fetched as Docker images from the Ersilia Model Hub
- All descriptors are calculated at the first step (Raw Descriptors) and only those specified in the TREATED_DESCRIPTORS list in the `00_zaira_base/vars.py` file will be treated. This is designed to be user-error proven. If a descriptor is added through a config file, it will be calculated, even if it has never been specified to ZairaChem before. The only thing a user should decide is whether to treat them or not. We can also add a long list of TREATED descriptors from the Ersilia Model Hub to make it more error safe. 
- The reference descriptor is specified in the `00_zaira_base/vars.py` and calculated at the same time as the others. If the user passes it as part of the descriptors in the config_file it will still work fine. 
- The FPSIM2 folder has been eliminated as FPSIM is now pip installable (since v0.4)
- The label used for the PCA, UMAP and LOLP is the Bin column, not the AUX