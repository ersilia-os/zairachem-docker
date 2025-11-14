import csv, shutil, subprocess, requests, sys, time
from io import StringIO
from zairachem.base.utils.logging import logger
from zairachem.base.utils.terminal import run_command
from zairachem.base.vars import (
  DEFAULT_PROJECTIONS,
  GITHUB_CONTENT_URL,
  PREDEFINED_COLUMN_FILE,
  DEFAULT_PUBLIC_BUCKET,
  DEFAULT_PRIVATE_BUCKET,
)
from isaura.manage import IsauraInspect


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


def fetch_schema_from_github(model_id):
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
  return col_name, col_dtype, shape


def post(data, url):
  try:
    logger.info(f"Computing projection using {DEFAULT_PROJECTIONS}. Assigned url {url}")
    res = requests.post(url=url, json=data)
    return res.json()
  except requests.RequestException as e:
    logger.critical(f"Error occured when computing the projection -> {e}")


def resolve_default_bucket(access):
  return DEFAULT_PUBLIC_BUCKET[0] if access == "public" else DEFAULT_PRIVATE_BUCKET[0]


def get_bucket_records(bucket):
  insp = IsauraInspect(model_id="_", model_version="_", cloud=False)
  return insp.inspect_models(bucket, prefix_filter="")


def latest_version(model, bucket):
  if model is None:
    return None
  records = get_bucket_records(bucket)
  versions = []
  for r in records:
    name = r["model"]
    if name.startswith(model + "/"):
      v = name.split("/")[1]
      if v[1:].isdigit():
        versions.append(int(v[1:]))
  return "v" + str(max(versions)) if versions else "v1"
