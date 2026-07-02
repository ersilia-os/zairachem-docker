<div id="top"></div>


<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License: GPL v3"></a>
  <img src="https://img.shields.io/badge/python-%E2%89%A5%203.10-blue.svg" alt="Python >= 3.10">
  <a href="https://ersilia.io"><img src="https://img.shields.io/badge/Powered%20by-Ersilia-6c5ce7.svg" alt="Powered by Ersilia"></a>
</p>

---
# ZairaChem, automated ML modeling for QSAR
## What is ZairaChem?

ZairaChem trains structure–activity (QSAR) models straight from a CSV of SMILES — no manual feature engineering. Give it a labelled dataset and it will:

1. **Featurize** the molecules with a curated set of [Ersilia](https://ersilia.io) descriptor models,
2. **Train and pool** per-descriptor models into a single consensus predictor (classification or regression),
3. **Predict** activities for new molecules and render a self-contained **HTML report** with the plots and metrics.

It's built for chemists and modelers who want a strong baseline model over a compound library without writing ML code. ZairaChem is a project of the [Ersilia Open Source Initiative](https://ersilia.io).
## Requirements

- **Docker**, installed and **running** — ZairaChem serves the descriptor models in containers during the `describe` step. Start Docker before you run `fit` or `predict`.
- **Python ≥ 3.10** (a dedicated conda environment is recommended).
- **[Ersilia](https://github.com/ersilia-os/ersilia)**, installed with the default descriptor models fetched. The default set is:
  - featurizers: `eos3l5f`, `eos8aa5`, `eos4u6p`, `eos9o72`, `eos4ex3`, `eos82v1`
  - projection (for the report's 2-D map): `eos1klk`

  See the [Ersilia docs](https://ersilia.gitbook.io/ersilia-book/) for installation, then fetch each
  model with `ersilia fetch <eos-id>`.
## Installation

```bash
conda create -n zairachem python=3.11 -y
conda activate zairachem

git clone https://github.com/ersilia-os/zairachem-docker
cd zairachem-docker
pip install -e .
```

Check the install (this also prints a Docker readiness line):

```bash
zairachem --help
```

## Quickstart

Train a classifier and predict, using the example dataset shipped in `dev/data.csv`
(475 molecules, columns `smiles,dili`). Make sure Docker is running first.

```bash
# 1. Train a classification model
zairachem fit -i dev/data.csv -m ./dili_model -c

# 2. Predict on new molecules (reusing the example here)
zairachem predict -i dev/data.csv -m ./dili_model -o ./predictions
```

**Input** — a CSV with a SMILES column (auto-detected). For `fit`, add a label column: `0`/`1` for
classification, or a numeric value for regression.

```csv
smiles,dili
CC(=O)OCC[N+](C)(C)C,0
O=C(O)c1cccnc1,1
```

**Output** — predictions and a report land under the model dir (`fit`) or the `-o` dir (`predict`):

- `./dili_model/results/output.csv` — the predictions
- `./dili_model/report/report.html` — open in a browser for the plots and metrics

```csv
compound_id,smiles,clf,clf_bin
CID000,CC(=O)OCC[N+](C)(C)C,0.13,0
CID002,O=C(O)c1cccnc1,0.61,1
```

## CLI reference

Run `zairachem <command> --help` for the authoritative, always-current options. Add `-v`/`--verbose`
**before** the command for detailed logs (e.g. `zairachem -v fit ...`).

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
| `--evaluate` | off | Held-out validation (classification only). Bare = all schemas (random, scaffold, scaffold_det, butina); or a subset, e.g. `--evaluate scaffold,random`. |
| `--repeats` | `3` | Held-out repeats per schema when `--evaluate` is set (total folds = 1 + 3 × repeats). |

### `predict` — predict with a trained model

Same core options as `fit` (`-i`, `-m`, `-s`, `--override`, `-b`, `--workers`, `--skip-report`,
`--keep-intermediate-data`, `--anonymize`), plus:

| Option | Default | Description |
| --- | --- | --- |
| `-o, --output-dir` | *(required)* | Where predictions and the report are written. |

### Advanced: run the pipeline step by step

`fit` runs the full pipeline; you can also run (or re-run) individual steps against a model dir with
`-m`. Order matters:

```bash
zairachem setup -i data.csv -c   # standardize & prepare the molecules
zairachem describe               # compute descriptors (needs Docker)
zairachem treat                  # impute & scale the descriptor matrix
zairachem estimate               # train the per-descriptor estimators
zairachem pool                   # combine them into a consensus
zairachem report                 # render plots, tables & report.html
zairachem finish                 # assemble final outputs & clean up
```

Re-running a single step is handy for iterating — e.g. `zairachem report -m ./dili_model` to
re-render the report without redoing training.

## Links · License · Citation

- **Ersilia Model Hub** — https://ersilia.io · https://github.com/ersilia-os/ersilia
- **This repository** — https://github.com/ersilia-os/zairachem-docker

Licensed under the **GNU General Public License v3.0** (see [LICENSE](LICENSE)).

If you use ZairaChem in your research, please cite:

> Turon, G., Hlozek, J., Woodland, J. G., et al. *First fully-automated AI/ML virtual screening
> cascade implemented at a drug discovery centre in Africa.* Nature Communications, 2023.
> <!-- TODO: confirm the exact DOI, e.g. https://doi.org/10.1038/s41467-023-41512-2 -->

<p align="right"><a href="#top">↑ back to top</a></p>
