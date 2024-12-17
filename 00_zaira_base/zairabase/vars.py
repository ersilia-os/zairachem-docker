import os
from pathlib import Path

BASE_DIR = os.path.join(str(Path.home()), "zairachem")
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

LOGGING_FILE = "console.log"
SESSION_FILE = "session.json"

# Environmental variables

COMPOUNDS_FILENAME = "compounds.csv"
STANDARD_COMPOUNDS_FILENAME = "compounds_std.csv"
MAPPING_FILENAME = "mapping.csv"
VALUES_FILENAME = "values.csv"
TASKS_FILENAME = "tasks.csv"
FOLDS_FILENAME = "folds.csv"
PARAMETERS_FILE = "parameters.json"
INPUT_SCHEMA_FILENAME = "input_schema.json"
RAW_INPUT_FILENAME = "raw_input"
RAW_DESC_FILENAME = "raw.h5"
TREATED_FILE_NAME = "treated.h5"


MAPPING_ORIGINAL_COLUMN = "orig_idx"
MAPPING_DEDUPE_COLUMN = "uniq_idx"
COMPOUND_IDENTIFIER_COLUMN = "compound_id"
USER_COMPOUND_IDENTIFIER_COLUMN = "orig_compound_id"
SMILES_COLUMN = "smiles"
STANDARD_SMILES_COLUMN = "standard_smiles"
DATE_COLUMN = "date"
QUALIFIER_COLUMN = "qualifier"
VALUES_COLUMN = "value"
GROUP_COLUMN = "group"
DIRECTION_COLUMN = "direction"
AUXILIARY_TASK_COLUMN = "clf_aux"

DATA_SUBFOLDER = "data"
DATA_FILENAME = "data.csv"
REFERENCE_FILENAME = "reference.csv"
INTERPRETABILITY_SUBFOLDER = "interpretability"
APPLICABILITY_SUBFOLDER = "applicability"
DESCRIPTORS_SUBFOLDER = "descriptors"
ESTIMATORS_SUBFOLDER = "estimators"
POOL_SUBFOLDER = "pool"
RESULTS_FILENAME = "results_unmapped.csv" #TODO CONSOLIDATE WITH RESULTS_MAPPED_FILENAME
LITE_SUBFOLDER = "lite"
REPORT_SUBFOLDER = "report"
DISTILL_SUBFOLDER = "distill"
OUTPUT_FILENAME = "output.csv"
OUTPUT_TABLE_FILENAME = "output_table.csv"
PERFORMANCE_TABLE_FILENAME = "performance_table.csv"
OUTPUT_XLSX_FILENAME = "output.xlsx"

Y_HAT_FILE = "y_hat.joblib"

RESULTS_UNMAPPED_FILENAME = "results_unmapped.csv"
RESULTS_MAPPED_FILENAME = "results_mapped.csv"

CLF_REPORT_FILENAME = "clf_report.json"
REG_REPORT_FILENAME = "reg_report.json"

CLF_PERCENTILES = [1, 10, 25, 50]

MIN_CLASS = 30
N_FOLDS = 5

_CONFIG_FILENAME = "config.json"

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
ZAIRACHEM_DATA_PATH = os.path.join(PACKAGE_ROOT, "data")

# Ersilia Model Hub

ERSILIA_HUB_DEFAULT_MODELS = [
    "morgan-counts",
    "rdkit-fingerprint",
    "mordred",
    "cc-signaturizer",
    "molfeat-chemgpt",
    "eosce",
]

REFERENCE_DESCRIPTOR = "grover-embedding"

TREATED_DESCRIPTORS = [
    "morgan-counts",
    "rdkit-fingerprint",
    "mordred",
]

DEFAULT_ESTIMATORS = [
    "baseline-classic",
    "baseline-fingerprint",
    "flaml-individual-descriptors",
    #"tabpfn-individual-descriptors",
    "autogluon-manifolds",
    "kerastuner-reference-embedding",
    "kerastuner-eosce-embedding",
    "molmap",
]

ENSEMBLE_MODE = (
    "bagging"  # bagging, blending, stacking / at the moment only bagging is available
)

DEFAULT_PRESETS = "standard"  # the other option is lazy