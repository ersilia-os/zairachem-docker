<p align="left">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License: GPL v3"></a>
  <img src="https://img.shields.io/badge/python-3.11%E2%80%933.13-blue.svg" alt="Python 3.11-3.13">
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

Train a classifier and predict, using the example DILI data shipped in `example/`:
`data_train.csv` (380 molecules, columns `smiles,dili`) to train, and `smiles_test.csv`
(95 molecules, `smiles` only) to predict. Make sure Docker is running first.

```bash
# 1. Train a classification model
zairachem fit -i example/data_train.csv -m example_model

# 2. Predict on new molecules
zairachem predict -i example/smiles_test.csv -m example_model -o predictions_test
```

## Commands

| Command | Description |
| --- | --- |
| `zairachem fit` | Train a QSAR model from a labelled CSV (runs every step below). |
| `zairachem predict` | Predict activities for new molecules with a trained model. |
| `zairachem setup` | Standardize and prepare the input molecules. |
| `zairachem describe` | Compute molecular descriptors for each featurizer. |
| `zairachem treat` | Impute and scale the descriptor matrix. |
| `zairachem estimate` | Train the per-descriptor base estimators. |
| `zairachem pool` | Combine the per-descriptor estimators into a consensus. |
| `zairachem report` | Render the plots, tables and HTML report. |
| `zairachem finish` | Assemble final outputs and clean up intermediate data. |

Run `zairachem <command> --help` for its options. `setup` creates a model directory (`-m`) from the
input CSV; the later step commands (`describe` to `finish`) each run one stage on it.

By default `fit` uses a curated set of Ersilia descriptors (`-f` to choose your own). A descriptor
with no published transformer is skipped at setup. The default Mol-JEPA (`eos9q2i`) is licensed for
non-commercial use only; pass `-f` without it to exclude it.

## Rank references

lazy-qsar positions every prediction (`rank`) against a fixed library of 50,000 molecules, so each
descriptor a fit trains needs that library featurized with it. A fit downloads the published matrices
for the descriptors it trains into `~/zairachem/rank_reference/` (up to ~1.6 GB for the default
featurizers; `ZAIRACHEM_HOME` moves the folder). One that is not published (e.g. a custom featurizer)
is computed on its model server and cached there, so later fits reuse it.

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit organization fueling sustainable research in the Global South. Ersilia's main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), an open-source repository of AI/ML models for antimicrobial drug discovery.

ZairaChem is developed in collaboration with the [H3D Centre](https://h3d.uct.ac.za/) in Cape Town, South Africa.

![Ersilia Logo](assets/Ersilia_Brand.png)
