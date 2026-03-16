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
The input file must be a `.csv` file with two columns: one containing SMILES strings and the other containing a binary output (0,1)

**Usage:**

```bash
zairachem fit -i INPUT_FILE [-m MODEL_DIR] [OPTIONS]
```

**Options:**

* `-i, --input-file` **\[required]**: Path to the input file.
* `-m, --model-dir`: Directory where the model is stored.
* `-c/-r, --classification/--regression`: type of model
* `-e, --eos-ids`: Ersilia models to use for featurization and projection.
* `--clean`: Clean descriptors at the end of the run to save space.
* `--flush`: Flush model checkpoints to save space (useful for cross-validations).
* `--anonymize`: Anonymize the inputs entirely.
* `-ct, --clean-target [all|model|predict]`: Target for clean/flush/anonymize operations (default: `all`).
* `-bs, --batch-size`: Batch size for chunked processing (default: 10000). Controls memory usage for large datasets.
* `--enable-store/-es [PROJECT]`: Enables reading precalculations from isaura store. Reads from `isaura-public` by default, or specify a custom project name (e.g., `-es my_project`).
* `--nearest-neighbor/-nn`: Enables nearest search neighbor search to find similar compounds.
* `--contribute-store/-cs [PROJECT]`: Enables uploading precalculations to isaura store. Without a project name, writes to a temporary bucket (`zairatemp`), copies to `isaura-public`, then removes the temp data. With a project name (e.g., `-cs my_project`), writes directly to that project only.

The eos-ids file must be a `.json` file with the following structure:
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
And the created `parameter.json` will look like:
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
  "read_store": "isaura-public",
  "enable_nns": false,
  "contribute_store": "my_project",
  "latest_featurizer_version": {
    "eos5axz": "v1",
    "eos2gw4": "v2"
  }
}
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

---

### Clean Target Options

The `--clean-target` (`-ct`) option allows granular control over which directories are affected by `--clean`, `--flush`, and `--anonymize` operations:

| Target | Description |
| ------ | ----------- |
| `all` | (Default) Apply to both model and prediction directories |
| `model` | Apply only to the model directory |
| `predict` | Apply only to the prediction directory (only valid during prediction) |

**Use Cases:**

1. **Keep model, clean predictions only** (most common for production):
   ```bash
   zairachem predict -i data.csv -m ./models --clean --clean-target predict
   ```

2. **Clean model only** (rare, but supported):
   ```bash
   zairachem predict -i data.csv -m ./models --clean --clean-target model
   ```

3. **Clean everything** (default behavior, useful for cross-validation):
   ```bash
   zairachem predict -i data.csv -m ./models --clean --clean-target all
   ```

---

### Batch Size for Large Datasets

For large datasets (e.g., 100k+ molecules), use the `--batch-size` (`-bs`) option to control memory usage:

```bash
# Process in chunks of 5000 molecules
zairachem fit -i large_data.csv -m ./models -bs 5000 -es

# Predict with custom batch size
zairachem predict -i large_predict.csv -m ./models -bs 5000 -es
```

The batch size controls:
- Chunked reading/writing of H5 descriptor files
- Batched API calls to Ersilia models
- Chunked processing in treatment, estimation, and pooling stages

Default batch size is 10,000 rows.

---

## Commands for executing each step in zairachem

| Command                                                     | What it does                                             |
| ----------------------------------------------------------- | -------------------------------------------------------- |
| `zairachem setup -i input.csv -c`                           | Preprocess input and prepare working artifacts.          |
| `zairachem describe [-bs BATCH_SIZE]`                       | Compute molecular descriptors for prepared inputs.       |
| `zairachem treat [-bs BATCH_SIZE]`                          | Impute/clean features produced by `describe`.            |
| `zairachem estimate [-bs BATCH_SIZE]`                       | Train/estimate models.                                   |
| `zairachem pool [-bs BATCH_SIZE]`                           | Bag results from `estimate`.                             |
| `zairachem report [--plot-name NAME]`                       | Generate analysis report and plots.                      |
| `zairachem finish [--clean --flush --anonymize] [-ct TARGET]` | Finalize: cleanup, flush caches, optional anonymization. |
