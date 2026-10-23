import os
import socket
from pathlib import Path


def get_free_ports(n):
  """Return ``n`` distinct free local TCP ports (released on return, so a later bind can still race)."""
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


# Per-user ZairaChem folder (logs, session link, rank-reference cache). ZAIRACHEM_HOME overrides it.
BASE_DIR = os.path.expanduser(
  os.environ.get("ZAIRACHEM_HOME", os.path.join(str(Path.home()), "zairachem"))
)
if not os.path.exists(BASE_DIR):
  os.makedirs(BASE_DIR)

LOGGING_FILE = "console.log"
SESSION_FILE = "session.json"
ORG = "ersiliaos"
REDIS_IMAGE = "redis:7.4.2"
DEFAULT_ISAURA_BUCKET = "isaura-public"
NETWORK_NAME = "ersilia_network"
# Compose project of the model servers shared by all runs of this user (see describe/compose.py).
SERVERS_PROJECT_NAME = "zairachem-servers"
GITHUB_ORG = "ersilia-os"
GITHUB_CONTENT_URL = f"https://raw.githubusercontent.com/{GITHUB_ORG}"
PREDEFINED_COLUMN_FILE = "model/framework/columns/run_columns.csv"
COMPOUNDS_FILENAME = "compounds.csv"
STANDARD_COMPOUNDS_FILENAME = "compounds_std.csv"
MAPPING_FILENAME = "mapping.csv"
#: Duplicate structure of the input and, for regression, how much repeats of one compound
#: disagreed -- the assay's own reproducibility, which is what an RMSE has to be read against.
REPLICATES_FILENAME = "replicates.json"
#: Written by the pooler at predict, beside the input it describes: how many of the molecules
#: being predicted were in the training set.
TRAIN_OVERLAP_FILENAME = "train_overlap.json"
VALUES_FILENAME = "values.csv"
TASKS_FILENAME = "tasks.csv"
PARAMETERS_FILE = "parameters.json"
INPUT_SCHEMA_FILENAME = "input_schema.json"
RAW_INPUT_FILENAME = "raw_input"
RAW_DESC_FILENAME = "raw.h5"
TREATED_DESC_FILENAME = "treated.h5"
# lazy-qsar's rank reference: a fixed library of drug-like molecules that `rank` is a position
# against. Describe featurizes it next to the run's own compounds (RAW), treat scales it with the same
# transformer (TREATED, written as one plain H5 because lazy-qsar's reference_h5_file reads a single
# file), and estimate hands it to fit. Fit-only: a trained model carries the scored reference itself.
RANK_REFERENCE_SMILES_FILENAME = "rank_reference.csv"
# A projection model's coordinates, computed during describe and read by the projections step.
PROJECTION_CACHE_FILENAME = "projection.json"
RANK_REFERENCE_RAW_FILENAME = "rank_reference_raw.h5"
RANK_REFERENCE_TREATED_FILENAME = "rank_reference.h5"
# Each descriptor's scores of the treated reference (proba, OOF percentile, AD), written by estimate;
# pool pushes them through its fitted weighting to place the pooled `rank` (clf_rank) on the library.
RANK_REFERENCE_SIGNALS_FILENAME = "reference_signals.npz"
# lazy-qsar's binary cutoff on the rank scale (lazyqsar.utils.ranking.DECISION_RANK: beats 99% of
# the reference). Mirrored for the report, which must not import lazy-qsar.
RANK_DECISION_CUTOFF = 0.65

# Named classification cutoffs, all stored in results/cutoffs.json (see pool/cutoffs.py):
# - screening: the call. The pooled probability at rank RANK_DECISION_CUTOFF against the rank
#   reference (its 99th percentile, the top 1% of drug-like chemical space), as lazy-qsar's binary.
# - kappa: maximises Cohen's kappa (GHOST) on the training out-of-fold pooled scores.
# - balanced: lazy-qsar's balanced-accuracy cutoff on each descriptor's out-of-fold scores.
# - default: DEFAULT_CUTOFF on the probability (per-descriptor calls; the pooled call without a
#   rank reference).
CUTOFF_SCREENING = "screening"
CUTOFF_KAPPA = "kappa"
CUTOFF_BALANCED = "balanced"
CUTOFF_DEFAULT = "default"
DEFAULT_CUTOFF = 0.5
CUTOFFS_FILENAME = "cutoffs.json"
# Descriptors the pooled model uses and those it left out (pool/), read by predict runs.
POOLED_DESCRIPTORS_FILENAME = "pooled_descriptors.json"
# Rank-reference matrices (describe/descriptors/rank_reference.py): one per featurizer/version under
# "<RANK_REFERENCE_DIR>/<REFERENCE_ID>_n<n>/", downloaded on demand from the public bucket (published
# from this repository's data/rank_reference with eosvc), or computed and cached when unpublished.
RANK_REFERENCE_DIR = os.path.join(BASE_DIR, "rank_reference")
RANK_REFERENCE_S3_BASE_URL = (
  "https://eosvc-public.s3.amazonaws.com/zairachem-docker/data/rank_reference"
)
SIMPLE_EVALUATION_FILENAME = "evaluation.json"
SIMPLE_EVALUATION_VALIDATION_FILENAME = "evaluation_validation_set.json"


MAPPING_ORIGINAL_COLUMN = "orig_idx"
MAPPING_DEDUPE_COLUMN = "uniq_idx"
COMPOUND_IDENTIFIER_COLUMN = "compound_id"
SMILES_COLUMN = "smiles"
STANDARD_SMILES_COLUMN = "standard_smiles"
VALUES_COLUMN = "value"

# Model-folder layout. Top-level dirs separate concerns: inputs (the run's input + derived data),
# metadata (run config/provenance), results (user-facing deliverables), report (visual report),
# transformers (the fitted eosframes transformers predict needs), descriptors (the featurized
# matrices, computed once and shared), model (the final production estimators + pooler) and folds
# (the held-out evaluation sub-runs, only written when --evaluate is set). Every internal subfolder
# nests under one of these via its constant value, so consumers that join these constants relocate
# automatically.
DATA_SUBFOLDER = "inputs"  # the run's input + derived data (was "data")
METADATA_SUBFOLDER = "metadata"  # parameters.json + provenance.json
RESULTS_SUBFOLDER = "results"  # user-facing deliverables (output.csv, tables, xlsx)
TRANSFORMERS_SUBFOLDER = "transformers"  # per-featurizer fitted transformers (predict reads these)
DATA_FILENAME = "data.csv"
# Bare one-column ("smiles") list of the run's compounds, for ad-hoc manual use.
SMILES_LIST_FILENAME = "smiles.csv"
ERSILIA_DATA_FILENAME = "ersilia_data.csv"
PROVENANCE_FILENAME = "provenance.json"
# Descriptors are featurized once on the full dataset and shared by the final model and every fold.
# The estimators + pooler of the final production model live under model/; each held-out fold gets a
# mirror of that structure under folds/<fold_name>/ (see FOLDS_SUBFOLDER, written only by --evaluate).
DESCRIPTORS_SUBFOLDER = "descriptors"
ESTIMATORS_SUBFOLDER = "model/estimators"
POOL_SUBFOLDER = "model/pool"
# The final production model (estimators + pooler); parent of the two constants above.
MODEL_SUBFOLDER = "model"
# Held-out evaluation sub-runs, one dir per fold (only written when --evaluate is set).
FOLDS_SUBFOLDER = "folds"
SPLITS_FILENAME = "splits.json"  # metadata/: the fold index definitions computed at setup
# metadata/: the descriptor subset chosen by --max-descriptors pre-screening (a subset of the
# featurizers in done_eos.json). When present it overrides done_eos.json for training/pool/report.
SELECTED_EOS_FILENAME = "selected_eos.json"
PROXY_SCORES_FILENAME = "proxy_scores.json"  # metadata/: per-descriptor proxy AUROC from screening
VALIDATION_TABLE_FILENAME = "validation_table.csv"  # report/: per-fold held-out metrics
VALIDATION_PREDICTIONS_FILENAME = (
  "validation_predictions.csv"  # report/: per-fold held-out y_true/y_score
)
REPORT_SUBFOLDER = "report"
# Deliverables live under results/. OUTPUT_FILENAME is deliverable-only, so it carries the results/
# prefix and cascades through is_done()/required-artifacts/finish. The two table filenames are bare
# because they are ALSO written inside report/ (report/<table>); finish copies them to results/ via
# RESULTS_SUBFOLDER.
OUTPUT_FILENAME = "results/output.csv"
OUTPUT_TABLE_FILENAME = "output_table.csv"
# Columns of results/output.csv per task, from the internal pooled-results names (in this order).
# Classification delivers `score`, the rank against the reference library; regression delivers
# `value`, the prediction in the target's own units (it has no rank, and its interval is in the
# output table). The similarity and spread exist only at predict; `label` (the input's truth, when
# it has one) is added from data.csv. The full output table (report/table.py) keeps more.
OUTPUT_COLUMNS = {
  "classification": {
    "compound_id": "compound_id",
    "smiles": "smiles",
    "clf_rank": "score",
    "train_similarity": "similarity",
    "descriptor_std": "spread",
  },
  "regression": {
    "compound_id": "compound_id",
    "smiles": "smiles",
    "reg": "value",
    "train_similarity": "similarity",
    "descriptor_std": "spread",
  },
}
OUTPUT_LABEL_COLUMN = "label"
# Decimals of the scores in results/output.csv. Classification scores are in [0, 1], so four
# decimals are more than they resolve; a regression value is in the target's own units, where
# rounding to four decimals is harmless but rounding to fewer would not be.
OUTPUT_DECIMALS = 4
# Prefix of the per-descriptor columns in the output table, by task (proba_<eos_id> / value_<eos_id>).
OUTPUT_DESCRIPTOR_PREFIX = {"classification": "proba_", "regression": "value_"}
# Columns blanked by `finish --anonymize`.
OUTPUT_SENSITIVE_COLUMNS = ("input_smiles", "smiles", "inchikey")
PERFORMANCE_TABLE_FILENAME = "performance_table.csv"
# Report-only 2-D projections (row-aligned to data.csv); written by the treat/manifolds step.
PROJECTIONS_FILENAME = "projections.csv"
PROJECTIONS_MANIFEST_FILENAME = "projections.json"

Y_HAT_FILE = "y_hat.joblib"

RESULTS_UNMAPPED_FILENAME = "results_unmapped.csv"
RESULTS_MAPPED_FILENAME = "results_mapped.csv"


#: The ``inputs/data.csv`` column holding the target of each task.
TARGET_COLUMN = {"classification": "bin", "regression": "val"}

#: How each task records a descriptor's inner cross-validation skill in its ``cv_report.json``:
#: ``(out-of-fold key, in-sample key, display label, value at chance)``. lazy-qsar writes AUROC for
#: a classifier and Spearman's rho for a regressor, and everything that ranks, plots or prints a
#: descriptor's skill reads it through here so the report cannot rank one way in a figure and
#: another way in the table beside it.
CV_SKILL = {
  "classification": ("oof_auc", "train_auc", "AUROC", 0.5),
  "regression": ("oof_rho", "train_rho", "Spearman \u03c1", 0.0),
}

#: The held-out skill a descriptor must clear to be worth including, per task: AUROC for
#: classification, Spearman's rho for regression. lazy-qsar's pruning rule and the screening proxy
#: both use it, and a pooled model that does not clear it is reported as having no usable signal.
#: The regression floor is the classification one mapped onto rho, as lazy-qsar scales it:
#: ``AUROC - 0.5`` is about ``rho / 2``.
SKILL_FLOOR = {"classification": 0.55, "regression": 0.10}
MIN_CLASS = 30
# Fewest compounds a regression target may have (lazy-qsar's own floor is 10; this leaves room
# for the out-of-fold splits its heads are calibrated on).
MIN_REGRESSION_ROWS = 30
RANDOM_SEED = 42


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

# A descriptor with no published reference-library transformer is skipped at setup, so one can be
# listed here before its transformer exists and joins runs once it is published.
DEFAULT_FEATURIZERS = [
  "eos3l5f",
  "eos8aa5",
  "eos4u6p",
  "eos9o72",
  "eos4ex3",
  "eos82v1",
  "eos9q2i",
  "eos8gop",
]
# Default Ersilia projection model for the report's 2-D embedding: eos1klk (lazy-chemvis PCA/UMAP/
# t-SNE/TMAP over the Ersilia reference library). The built-in MW-vs-LogP projection (computed
# locally with RDKit) is always shown in addition. Override or disable via --projection-ids.
DEFAULT_PROJECTIONS = ["eos1klk"]
# Hard caps on how many Ersilia Model Hub models a single run may pull. Each model is a Docker
# container featurizing the full dataset, so more models means proportionally more compute and
# disk; these ceilings keep a run tractable. Exceeding either cap is rejected at setup.
MAX_FEATURIZERS = 10
# Default --max-descriptors: the pre-screen keeps up to this many descriptors that add to the pool.
DEFAULT_MAX_DESCRIPTORS = 5
MAX_PROJECTIONS = 3

DEFAULT_ISAURA_BATCH_SIZE = 10000
DEFAULT_CHUNK_SIZE = 10000

# Molecules per request to a model server, the same for every model (fewer molecules go in one
# request). A failed request halves the size for the rest of that model's run, down to the minimum,
# before single molecules are isolated.
DESCRIBE_BATCH_SIZE = 10000
MIN_DESCRIBE_BATCH_SIZE = 8
# Most molecules a projection model gets (one request, never split): fewer than the server's chunking
# line (SERVER_CHUNKING_MOLECULES), since a projection must embed its molecules together.
MAX_PROJECTION_MOLECULES = 9999
# Longest wait for one model-server reply: a base plus a generous allowance per molecule (the slowest
# model measured needs ~0.2 s per molecule), finite so that a server whose worker died does not
# leave the request waiting forever.
DESCRIBE_READ_TIMEOUT_S = 900
DESCRIBE_READ_TIMEOUT_PER_MOLECULE_S = 1.0
# Copies of a model a server may run for one request (its max_workers). Left unset, the server
# (ersilia-pack) runs up to 16, sized from the whole VM's memory, so a large model runs out of memory;
# ZairaChem gives each server one, and more only for requests the server cuts into chunks anyway
# (SERVER_CHUNKING_MOLECULES or more), as far as the describe scheduler has cores and memory free.
DESCRIBE_SERVER_WORKERS = 1
MAX_DESCRIBE_SERVER_WORKERS = 4
# Model servers cut a request of this many molecules or more into chunks computed separately, this
# many per server worker, so each model process sees a chunk of n / (chunks * workers) molecules.
SERVER_CHUNKING_MOLECULES = 10000
SERVER_CHUNKS_PER_WORKER = 4
# Docker VM memory kept for the VM itself (kernel, Docker daemon). The describe scheduler subtracts
# what the running containers hold (idle model servers, redis) and shares the rest between the
# models running at once, by what each one's predicted peak adds to its idle server
# (describe/descriptors/profiles.py).
DESCRIBE_MEMORY_RESERVE_GB = 1.0
# What the describe scheduler assumes for a model before its own requests have been measured on
# this machine (its profile then takes over; describe/descriptors/profiles.py). Generic on purpose:
# any Ersilia model can be a descriptor, so nothing here is specific to one model.
DEFAULT_DESCRIBE_MEMORY_GB = 3.0
DEFAULT_DESCRIBE_CORES = 1.0
MODEL_PROFILES_FILENAME = "model_profiles.json"  # under BASE_DIR
# A featurizer that still fails on more than this fraction of molecules is dropped at fit (an error at
# predict, where the trained model needs it).
DESCRIBE_MAX_FAILED_FRACTION = 0.1
# ...but always allow this many, so one bad molecule does not fail a small input.
DESCRIBE_MIN_FAILED_ALLOWED = 10
