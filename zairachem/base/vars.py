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
NETWORK_NAME = "ersilia_network"

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
FILTERED_DESC_FILENAME = "filtered.h5"
TREATED_DESC_FILENAME = "treated.h5"
SIMPLE_EVALUATION_FILENAME = "evaluation.json"
SIMPLE_EVALUATION_VALIDATION_FILENAME = "evaluation_validation_set.json"


MAPPING_ORIGINAL_COLUMN = "orig_idx"
MAPPING_DEDUPE_COLUMN = "uniq_idx"
COMPOUND_IDENTIFIER_COLUMN = "compound_id"
SMILES_COLUMN = "smiles"
STANDARD_SMILES_COLUMN = "standard_smiles"
VALUES_COLUMN = "value"

DATA_SUBFOLDER = "data"
DATA_FILENAME = "data.csv"
ERSILIA_DATA_FILENAME = "ersilia_data.csv"
REFERENCE_FILENAME = "reference.csv"
DESCRIPTORS_SUBFOLDER = "descriptors"
ESTIMATORS_SUBFOLDER = "estimators"
POOL_SUBFOLDER = "pool"
REPORT_SUBFOLDER = "report"
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
ERSILIA_HUB_DEFAULT_MODELS = ["eos5axz", "eos4u6p", "eos2gw4", "eos8aa5"]
ERSILIA_HUB_DEFAULT_MODELS_DTYPE = {"eos5axz":"int32", "eos4u6p":"float", "eos2gw4":"float", "eos8aa5": "float"}
ERSILIA_HUB_DEFAULT_MODELS_DIMS = {"eos5axz":2048, "eos4u6p":3200, "eos2gw4":1024, "eos8aa5": 2304}
ERSILIA_HUB_DEFAULT_MODELS_WITH_PORT = {
  k: v for k, v in zip(ERSILIA_HUB_DEFAULT_MODELS, get_free_ports(len(ERSILIA_HUB_DEFAULT_MODELS)))
}

DEFAULT_PROJECTIONS = ["eos2db3"]
