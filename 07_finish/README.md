# 07. Finishing

## Run from Docker

```bash


```

## Run from system 

```bash
conda create -n zairafinish python=3.12
conda activate zairafinish
cd 07_finish
python pip install -e .
python zairafinish/run.py
```

## High level overview
The finisher offers three options:
- Clean: Clean directory at the end of the pipeline. Only precalculated descriptors are removed
- Flush:Flush directory at the end of the pipeline. Only data, results and reports are kept. Use with caution: the original trained model will be flushed too.
- Anonymize: Remove all information about training set, including smiles, physchem propertie and descriptors

### Differences with ZairaChem v1
