# 06. Reporting

## Run from Docker

```bash


```

## Run from system 

```bash
conda create -n zairareport python=3.12
conda activate zairareport
cd 06_report
python pip install -e .
python zairareport/run.py
```

## High level overview
The results files on the `pool` folder are read and two final tables (output_table and performance_table) are created, containing both individual and pooled results analysis (like AUC, Accuracy, etc...). Equally, plots are automatically generated

### Differences with ZairaChem v1
None

### TODO
- Revise unused functions like the applicability. 
- Create a single report file with all the plots
