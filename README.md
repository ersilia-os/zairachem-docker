<p align="left">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License: GPL v3"></a>
  <img src="https://img.shields.io/badge/python-%E2%89%A5%203.10-blue.svg" alt="Python >= 3.10">
  <a href="https://ersilia.io"><img src="https://img.shields.io/badge/Powered%20by-Ersilia-6c5ce7.svg" alt="Powered by Ersilia"></a>
</p>

# ZairaChem, automated ML modeling for QSAR

ZairaChem trains structure-activity (QSAR) models straight from data files without manual feature engineering. Give it a labelled dataset and it will:

1. **Featurize** the molecules with a curated set of [Ersilia](https://ersilia.io) descriptor models.
2. **Train and pool** per-descriptor models into a single consensus predictor.
3. **Predict** activities for new molecules and produce a **report** with plots and metrics.

## Installation

ZairaChem requires [Docker](https://docs.docker.com/engine/install/) to be installed and running.

The quickest way to install ZairaChem is the interactive installer. It sets up a Python environment, installs
ZairaChem, pulls the Docker base images, and (optionally) fetches the default Ersilia models:

```bash
git clone https://github.com/ersilia-os/zairachem-docker
cd zairachem-docker
bash install.sh
```

Run `bash install.sh --help` for non-interactive flags.

Check the install (this also prints a Docker readiness line):

```bash
zairachem --help
```

## Quickstart

Train a classifier and predict, using the example dataset shipped in `dev/data.csv`
(475 molecules, columns `smiles,dili`). Make sure Docker is running first.

```bash
# 1. Train a classification model
zairachem fit -i example/data_train.csv -m example_model

# 2. Predict on new molecules (reusing the example here)
zairachem predict -i example/smiles_test.csv -m example_model -o predictions_test
```

## CLI reference

Run `zairachem <command> --help` for the authoritative, always-current options.

### `fit` — train a model from a labelled CSV

| Option | Default | Description |
| --- | --- | --- |
| `-i, --input-file` | *(required)* | Input CSV (must contain a SMILES column). |
| `-m, --model-dir` | *(required)* | Where the trained model is written. |
| `-c, --classification` / `-r, --regression` | classification | Model type. |
| `-f, --featurizer-ids` | curated set | Descriptor model IDs (comma-separated) or a JSON file. |
| `-p, --projection-ids` | `eos1klk` | Projection model(s) for the report's 2-D map. |
| `-s, --store` | off | Cache descriptor precalculations in an isaura project to speed up re-runs. |
| `--override` | off | Wipe & rebuild the model dir if it exists (otherwise resume/abort). |
| `-b, --batch-size` | `10000` | Rows per chunk for large datasets (controls memory). |
| `--workers` | `1` | Descriptor models to featurize in parallel. |
| `--skip-report` | off | Skip the plots/HTML report (still writes the results tables). |
| `--keep-intermediate-data` | off | Keep descriptor matrices etc. instead of cleaning up. |
| `--anonymize` | off | Blank out SMILES / InChIKey in all outputs. |
| `--max-descriptors` | `3` | Cap the ensemble at the top-K descriptors, pre-screened by mean held-out AUROC across the chemistry-aware splits (classification only). Set ≥ the number you provide (e.g. `10`) to train all. Hard cap is 10. |
| `--evaluate` | off | Held-out validation (classification only). Bare = all schemas (random, scaffold, scaffold_det, butina); or a subset, e.g. `--evaluate scaffold,random`. |
| `--repeats` | `3` | Held-out repeats per schema when `--evaluate` is set (total folds = 1 + 3 × repeats). |

### `predict` — predict with a trained model

Same core options as `fit` (`-i`, `-m`, `-s`, `--override`, `-b`, `--workers`, `--skip-report`,
`--keep-intermediate-data`, `--anonymize`), plus:

| Option | Default | Description |
| --- | --- | --- |
| `-o, --output-dir` | *(required)* | Where predictions and the report are written. |

## About the Ersilia Open Source Initiative

ZairaChem is a project of the [Ersilia Open Source Initiative](https://ersilia.io) done in collaboration with the [H3D Centre](https://h3d.uct.ac.za/) in Cape Town, South Africa.
