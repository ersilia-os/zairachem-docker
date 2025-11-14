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
* `-c/-r, --classification/--regression`: type of model
* `-e, --eos-ids`: Ersilia models to use for featurization and projection.
* `--clean`: `True/False`.
* `--flush`:` True/False`.
* `--anonymize`: `True/False`.
* `--enable-store/-es`: `True/False`: enables fetching or storing precalculation from isaura store
* `--nearest-neighbor/-nn`: `True/False`: enables nearest search neighbor search to find similar compounds
* `--contribute-store/-cs`: `True/False`: enables contributing precalculations stored in custom isaura projects to the default projects [isaura-public/isaura-private]
* `--access/-a`: `public/private`: defines where to read the precalculation store [`public` `->` `isaura-public` and `private` `->` `isaura-private`]
The eos-ids file must be a .json file with the following structure:
```bash
{
    "featurizer_ids": [
        "eos5axz",
        "eos4u6p"
    ],
    "projection_ids": [
        "eos2db3"
    ]
}
```
`parameter.json`
```json
{
  "task": "classification",
  "featurizer_ids": [
    "eos5axz",
    "eos2gw4"
  ],
  "projection_ids": [
    "eos2db3"
  ],
  "enable_cache": true,
  "access": "public",
  "enable_nns": false,
  "contribute_cache": true,
  "latest_featurizer_version": {
    "eos5axz": "v1",
    "eos2gw4": "v2"
  }
}%
```

**Example:**

```bash
zairachem fit -i data.csv -m ./models -c -e ./descriptors.json --clean
```

---

#### 🔹 `predict`

Run predictions on new data using a trained model.
Also executes the full post-processing and reporting pipeline.

**Usage:**

```bash
zairachem predict -i INPUT_FILE -m MODEL_DIR [-o OUTPUT_DIR] [OPTIONS]
```

**Options (in addition to `fit` options):**

* `-o, --output-dir`: Directory to save outputs.
* `--override-dir`: Overwrite the output directory if it already exists.

**Example:**

```bash
zairachem predict -i new_data.csv -m ./models -o ./results --clean --override-dir
```

## Commands for executing each step in zairachem

| Command                                                     | What it does                                             |
| ----------------------------------------------------------- | -------------------------------------------------------- |
| `zairachem setup -i input.csv -c`                           | Preprocess input and prepare working artifacts.          |
| `zairachem describe`                                        | Compute molecular descriptors for prepared inputs.       |
| `zairachem treat`                                           | Impute/clean features produced by `describe`.            |
| `zairachem estimate`                                        | Train/estimate models.                                    |
| `zairachem pool`                                            | Bag results from `estimate`.                             |
| `zairachem report [--plot-name NAME]`                       | Generate analysis report and plots.                      |
| `zairachem finish [--clean --flush --anonymize]`            | Finalize: cleanup, flush caches, optional anonymization. |
