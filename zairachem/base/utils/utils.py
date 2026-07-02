import csv, shutil, subprocess, requests, sys, time
from io import StringIO
from zairachem.base.utils.logging import logger
from zairachem.base.utils.terminal import run_command
from zairachem.base.vars import (
  GITHUB_CONTENT_URL,
  PREDEFINED_COLUMN_FILE,
)

try:
  from isaura.manage import IsauraInspect
except ImportError:
  IsauraInspect = None


def write_smiles_list(data_dir, smiles):
  """Write a bare one-column (``smiles``) CSV of the run's compounds for ad-hoc manual use.

  Parameters
  ----------
  data_dir : str
      The run's ``data`` subfolder, where the file (``smiles.csv``) is written.
  smiles : list of str
      The run's (standardized) input SMILES.
  """
  import os

  import pandas as pd

  from zairachem.base.vars import SMILES_LIST_FILENAME

  pd.DataFrame({"smiles": list(smiles)}).to_csv(
    os.path.join(data_dir, SMILES_LIST_FILENAME), index=False
  )


def install_docker_compose(install_file):
  if shutil.which("docker-compose") is None:
    logger.warning("docker‑compose not found; running installer…")
    try:
      run_command(["bash", install_file])
      logger.info("Installation complete.")
    except subprocess.CalledProcessError as e:
      logger.error(f"Installation failed (exit code {e.returncode})", file=sys.stderr)
      sys.exit(e.returncode)
  else:
    logger.info("docker‑compose is already installed.")


_SCHEMA_CACHE = {}


def fetch_schema_from_github(model_id):
  """Fetch a model's output schema (column names, dtypes, width) from its GitHub repo, or None.

  Cached per process (successful results only — a failed fetch is not cached, so a transient error can
  still recover on retry). This matters because the describe path asks for the schema several times per
  model (dtype, dims, placeholder row, hybrid read), and each uncached call is a GitHub round-trip.
  """
  if model_id in _SCHEMA_CACHE:
    return _SCHEMA_CACHE[model_id]
  st = time.perf_counter()
  try:
    response = requests.get(f"{GITHUB_CONTENT_URL}/{model_id}/main/{PREDEFINED_COLUMN_FILE}")
  except requests.RequestException:
    logger.warning("Couldn't fetch column name from github!")
    return None

  csv_data = StringIO(response.text)
  reader = csv.DictReader(csv_data)

  if "name" not in reader.fieldnames:
    logger.warning("Couldn't fetch column name from github. Column name not found.")
    return None

  if "type" not in reader.fieldnames:
    logger.warning("Couldn't fetch data type from github. Column name not found.")
    return None

  rows = list(reader)
  col_name = [row["name"] for row in rows if row["name"]]
  col_dtype = [row["type"] for row in rows if row["type"]]
  shape = len(col_dtype)
  if len(col_name) == 0 and len(col_dtype) == 0:
    return None
  et = time.perf_counter()
  logger.info(f"Column metadata fetched in {et - st:.2} seconds!")
  result = (col_name, col_dtype, shape)
  _SCHEMA_CACHE[model_id] = result
  return result


_METADATA_CACHE = {}


def fetch_model_metadata(model_id):
  """Fetch an Ersilia model's metadata dict from its GitHub repo, or None if unavailable.

  Tries ``metadata.json`` first, then ``metadata.yml`` (models use one or the other). Returns the
  parsed mapping (keys like ``Task``, ``Subtask``, ``Output Dimension``) or ``None`` on any failure.
  Cached per process so each model is fetched at most once.
  """
  if model_id in _METADATA_CACHE:
    return _METADATA_CACHE[model_id]
  import json

  import yaml

  result = None
  for fname, parse in (("metadata.json", json.loads), ("metadata.yml", yaml.safe_load)):
    try:
      r = requests.get(f"{GITHUB_CONTENT_URL}/{model_id}/main/{fname}", timeout=10)
    except requests.RequestException:
      continue
    if r.status_code == 200 and r.text.strip():
      try:
        parsed = parse(r.text)
      except Exception:
        continue
      if isinstance(parsed, dict):
        result = parsed
        break
  _METADATA_CACHE[model_id] = result
  return result


def post(data, url):
  """POST inputs to a served Ersilia model and return its JSON results (a list of per-row dicts).

  Raises ``RuntimeError`` if the request fails or the server returns an error response, so callers
  never mistake an error body (e.g. ``{"detail": ...}``) for results.
  """
  try:
    res = requests.post(url=url, json=data)
  except requests.RequestException as e:
    raise RuntimeError(f"Could not reach the model server at {url}: {e}") from e
  if res.status_code != 200:
    detail = ""
    try:
      detail = res.json().get("detail", "")
    except Exception:
      detail = (res.text or "")[:200]
    raise RuntimeError(f"Model server at {url} returned HTTP {res.status_code}: {detail}")
  body = res.json()
  if not isinstance(body, list):
    detail = body.get("detail", body) if isinstance(body, dict) else body
    raise RuntimeError(f"Model server at {url} returned an unexpected response: {detail}")
  return body


def get_bucket_records(bucket):
  insp = IsauraInspect(model_id="_", model_version="_", cloud=False)
  return insp.inspect_models(bucket, prefix_filter="")
