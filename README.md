<div id="top"></div>
<p align="center">
  <img src="/asset/zairachem_logo.png" height="250" alt="Zairachem logo">
</p>
<h2 align="center"> Welcome to Zairachem!</h2>
To install ZairaChem v2 do the following:

``` 
git clone https://github.com/ersilia-os/zairachem-docker
cd zairachem-docker
conda create -n zairachem python=3.11 -y
conda activate zairachem
pip install -e .
```
Since `zairachem` is depend in the `docker` and `docker-compose` we need to have them installed. Use this instruction for more detail [here](https://docs.docker.com/engine/install/ubuntu/). 
- To install `docker-compose` in MacOS we can simply execute:
```bash
brew install docker-compose
```
Once ZairaChem is installed in your environment, the CLI is available as:

```bash
zairachem [COMMAND] [OPTIONS]
```

---

### Commands

#### 🔹 `fit`

Train a model on your input data.
Runs preprocessing, descriptor computation, imputation, training, pooling, reporting, and finalization.

**Usage:**

```bash
zairachem fit -i INPUT_FILE [-m MODEL_DIR] [OPTIONS]
```

**Options:**

* `-i, --input-file` **\[required]**: Path to the input file.
* `-m, --model-dir`: Directory where the model is stored.
* `-c, --cutoff`: Cutoff threshold  `<float>`.
* `-d, --direction`: `high`/`low`.
* `-p, --parameters`: `<parameters_file.json>`.
* `--clean`: `True/False`.
* `--flush`:` True/False`.
* `--anonymize`: `True/False`.

**Example:**

```bash
zairachem fit -i data.csv -m ./models --clean
```

---

#### 🔹 `predict`

Run predictions on new data using a trained model.
Also executes the full post-processing and reporting pipeline.

**Usage:**

```bash
zairachem predict -i INPUT_FILE [-m MODEL_DIR] [-o OUTPUT_DIR] [OPTIONS]
```

**Options (in addition to `fit` options):**

* `-o, --output-dir`: Directory to save outputs.
* `--override-dir`: Overwrite the output directory if it already exists.

**Example:**

```bash
zairachem predict -i new_data.csv -m ./models -o ./results --override-dir
```

## Commands for executing each step in zairachem

| Command                                                     | What it does                                             |
| ----------------------------------------------------------- | -------------------------------------------------------- |
| `zairachem setup -i input.csv`                              | Preprocess input and prepare working artifacts.          |
| `zairachem describe`                                        | Compute molecular descriptors for prepared inputs.       |
| `zairachem treat`                                           | Impute/clean features produced by `describe`.            |
| `zairachem estimate [--time-budget-sec N]`                  | Train/estimate models (supports a time budget).          |
| `zairachem pool [--time-budget-sec N]`                      | Ensemble/bag results from `estimate`.                    |
| `zairachem report [--plot-name NAME]`                       | Generate analysis report and plots.                      |
| `zairachem finish [--clean --flush --anonymize]`            | Finalize: cleanup, flush caches, optional anonymization. |
