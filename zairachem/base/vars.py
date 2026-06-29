import os, socket
from pathlib import Path


def get_free_ports(n):
  ports = []
  sockets = []
  try:
    for _ in range(n):
      s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      s.bind(("", 0))
      s.listen(1)
      port = s.getsockname()[1]
      ports.append(port)
      sockets.append(s)
  finally:
    for s in sockets:
      s.close()
  return ports


BASE_DIR = os.path.join(str(Path.home()), "zairachem")
if not os.path.exists(BASE_DIR):
  os.makedirs(BASE_DIR)

LOGGING_FILE = "console.log"
SESSION_FILE = "session.json"
ORG = "ersiliaos"
NGINX_HOST_PORT = 80
REDIS_IMAGE = "redis:latest"
NGINX_IMAGE = "nginx:alpine"
DEFAULT_ISAURA_BUCKET = "isaura-public"
NETWORK_NAME = "ersilia_network"
GITHUB_ORG = "ersilia-os"
GITHUB_CONTENT_URL = f"https://raw.githubusercontent.com/{GITHUB_ORG}"
GITHUB_ERSILIA_REPO = "ersilia"
PREDEFINED_COLUMN_FILE = "model/framework/columns/run_columns.csv"
COMPOUNDS_FILENAME = "compounds.csv"
STANDARD_COMPOUNDS_FILENAME = "compounds_std.csv"
MAPPING_FILENAME = "mapping.csv"
VALUES_FILENAME = "values.csv"
TASKS_FILENAME = "tasks.csv"
PARAMETERS_FILE = "parameters.json"
INPUT_SCHEMA_FILENAME = "input_schema.json"
RAW_INPUT_FILENAME = "raw_input"
RAW_DESC_FILENAME = "raw.h5"
FILTERED_DESC_FILENAME = "filtered.h5"
TREATED_DESC_FILENAME = "treated.h5"
SIMPLE_EVALUATION_FILENAME = "evaluation.json"
SIMPLE_EVALUATION_VALIDATION_FILENAME = "evaluation_validation_set.json"

DEFAULT_PUBLIC_BUCKET = ["isaura-public"]
DEFAULT_PRIVATE_BUCKET = ["isaura-private"]

MAPPING_ORIGINAL_COLUMN = "orig_idx"
MAPPING_DEDUPE_COLUMN = "uniq_idx"
COMPOUND_IDENTIFIER_COLUMN = "compound_id"
SMILES_COLUMN = "smiles"
STANDARD_SMILES_COLUMN = "standard_smiles"
VALUES_COLUMN = "value"

# Model-folder layout. Top-level dirs separate concerns: inputs (the run's input + derived data),
# metadata (run config/provenance), results (user-facing deliverables), report (visual report),
# transformers (the fitted eosframes transformers predict needs), and pipeline (disposable internal
# artifacts). The internal subfolders nest under pipeline/ via their constant values, so consumers
# that join these constants relocate automatically.
DATA_SUBFOLDER = "inputs"  # the run's input + derived data (was "data")
METADATA_SUBFOLDER = "metadata"  # parameters.json + provenance.json
RESULTS_SUBFOLDER = "results"  # user-facing deliverables (output.csv, tables, xlsx)
TRANSFORMERS_SUBFOLDER = "transformers"  # per-featurizer fitted transformers (predict reads these)
DATA_FILENAME = "data.csv"
# Bare one-column ("smiles") list of the run's compounds, for ad-hoc manual use.
SMILES_LIST_FILENAME = "smiles.csv"
ERSILIA_DATA_FILENAME = "ersilia_data.csv"
REFERENCE_FILENAME = "reference.csv"
PROVENANCE_FILENAME = "provenance.json"
# Numbered to reflect the order the steps run (descriptors -> estimators -> pool).
DESCRIPTORS_SUBFOLDER = "pipeline/00_descriptors"
ESTIMATORS_SUBFOLDER = "pipeline/01_estimators"
POOL_SUBFOLDER = "pipeline/02_pool"
REPORT_SUBFOLDER = "report"
# Deliverables live under results/. OUTPUT_FILENAME / OUTPUT_XLSX_FILENAME are deliverable-only, so
# they carry the results/ prefix and cascade through is_done()/required-artifacts/finish. The two
# table filenames are bare because they are ALSO written inside report/ (report/<table>); finish
# copies them to results/ via RESULTS_SUBFOLDER.
OUTPUT_FILENAME = "results/output.csv"
OUTPUT_TABLE_FILENAME = "output_table.csv"
PERFORMANCE_TABLE_FILENAME = "performance_table.csv"
OUTPUT_XLSX_FILENAME = "results/output.xlsx"
# Report-only 2-D projections (row-aligned to data.csv); written by the treat/manifolds step.
PROJECTIONS_FILENAME = "projections.csv"
PROJECTIONS_MANIFEST_FILENAME = "projections.json"

Y_HAT_FILE = "y_hat.joblib"

RESULTS_UNMAPPED_FILENAME = "results_unmapped.csv"
RESULTS_MAPPED_FILENAME = "results_mapped.csv"

CLF_REPORT_FILENAME = "clf_report.json"
REG_REPORT_FILENAME = "reg_report.json"

MIN_CLASS = 30
N_FOLDS = 5
RANDOM_SEED = 42

_CONFIG_FILENAME = "config.json"

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
ZAIRACHEM_DATA_PATH = os.path.join(PACKAGE_ROOT, "data")

# Reference library used to pre-fit the eosframes descriptor transformers. The treat step
# downloads the transformer fitted on this library for each featurizer and applies it (rather than
# fitting a scaler per run). The name is recorded per model in parameters.json so predict reuses it.
DEFAULT_REFERENCE_LIBRARY = "ersilia_reference_library_v0"
# Public base URL under which each reference library is a path segment holding one flat object per
# featurizer/version, named "<eos_id>_<version>_transformer.json". The full object URL is
# "<base>/<reference_library>/<eos_id>_<version>_transformer.json".
REFERENCE_LIBRARY_S3_BASE_URL = "https://eosvc-public.s3.amazonaws.com/eosframes/output"
# Per-model local copies of fetched transformers live under transformers/ as
# "<eos_id>_<version>_transformer.json.gz" (gzipped, compact JSON). Predict reads them from there.
TRANSFORMER_SUFFIX = "_transformer.json.gz"

DEFAULT_FEATURIZERS = ["eos3l5f", "eos8aa5", "eos4u6p", "eos9o72", "eos4ex3", "eos82v1"]
# Default Ersilia projection model for the report's 2-D embedding: eos1klk (lazy-chemvis PCA/UMAP/
# t-SNE/TMAP over the Ersilia reference library). The built-in MW-vs-LogP projection (computed
# locally with RDKit) is always shown in addition. Override or disable via --projection-ids.
DEFAULT_PROJECTIONS = ["eos1klk"]
ALL_FEATURIZER = DEFAULT_FEATURIZERS + DEFAULT_PROJECTIONS

DEFAULT_ISAURA_BATCH_SIZE = 10000
DEFAULT_API_BATCH_SIZE = 1000
DEFAULT_CHUNK_SIZE = 10000
