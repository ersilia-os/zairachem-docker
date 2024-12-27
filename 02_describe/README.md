# 02 Molecular descriptors

This step calculates several descriptors for the molecules using only Ersilia Model Hub models.

## Run from system 

```bash
conda create -n zairadescribe python=3.12
cd 02_describe
python pip install -e .
python zairadescribe/run.py
```

## High level overview
This module only contains only one step:
1. Calculate Raw descriptors: Itt calculates the descriptors specified in the `data/parameters.json` file. Those come either from a config.json or are pulled from `vars.py`. If the reference descriptor is not already included in the list, it will be calculated additionally. Descriptors are stored in the `/descriptors` folder under their respective names. Slugs from the Ersilia Model Hub are used to identify the models.

The describe step for the fit and predict are the same. If in between the setup and the describe the user wishes to change session, the symlink of the zairachem/session.json file to the session.json file of the desired output directory must be manually updated.

### Differences with ZairaChem v1

A few improvements have been made to the Describe step:
- All descriptors must be fetched as Docker images from the Ersilia Model Hub
- All descriptors are calculated at the first step (Raw Descriptors) instead of in several rounds (reference, eosce, etc)
- The reference descriptor is specified in the `00_zaira_base/vars.py` and calculated at the same time as the others. If the user passes it as part of the descriptors in the config_file it will still work fine. 