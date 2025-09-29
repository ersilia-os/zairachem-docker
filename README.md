<div id="top"></div>
<p align="center">
  <img src="/asset/zairachem_logo.png" height="250" alt="Zairachem logo">
</p>
<h2 align="center"> Welcome to the Zairachem!</h2>
To install ZairaChem v2 do the following:

``` 
git clone https://github.com/ersilia-os/zairachem-docker
cd zairachem-docker
conda create -n zairachem python=3.11 -y
conda activate zairachem
pip install -e .
```
Since `zairachem` is depend in the `docker` and `docker-compose` we need to have them installed. Use this instruction for more detail (here)[https://docs.docker.com/engine/install/ubuntu/].
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
* `-c, --cutoff`: Cutoff threshold (e.g., probability or value).
* `-d, --direction`: Direction of processing (e.g., forward or backward).
* `-p, --parameters`: Additional model parameters as a string.
* `--clean`: Run in clean mode.
* `--flush`: Flush caches and temporary files.
* `--anonymize`: Anonymize outputs.

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
