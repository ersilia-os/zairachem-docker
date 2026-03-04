import csv, hashlib, joblib, json, os
import numpy as np
import pandas as pd
from zairachem.describe.descriptors.utils import get_model_url
from zairachem.treat.imputers import DescriptorBase
from zairachem.base.utils.matrices import Hdf5, Data
from zairachem.base.utils.utils import fetch_schema_from_github, post, latest_version
from zairachem.base.utils.logging import logger
from zairachem.base.vars import (
  DATA_FILENAME,
  DATA_SUBFOLDER,
  DESCRIPTORS_SUBFOLDER,
  PARAMETERS_FILE,
  ERSILIA_DATA_FILENAME,
  DEFAULT_PROJECTIONS,
)

try:
  from isaura.manage import IsauraCopy, IsauraReader, IsauraRemover, IsauraWriter
except Exception as e:
  logger.warning(f"Isaura modules could not be imported: {e}")
  IsauraCopy = None
  IsauraReader = None
  IsauraRemover = None
  IsauraWriter = None

ZAIRATEMP_BUCKET = "zairatemp"

MAX_COMPONENTS = 4


class Manifolds(DescriptorBase):
  def __init__(self):
    DescriptorBase.__init__(self)
    self.path = self.get_output_dir()
    self.input_file = os.path.join(self.path, DATA_SUBFOLDER, ERSILIA_DATA_FILENAME)
    self.url = get_model_url(DEFAULT_PROJECTIONS[0])
    self.model_id = DEFAULT_PROJECTIONS[0]
    params_file = os.path.join(self.trained_path, "data", PARAMETERS_FILE)
    with open(params_file, "r") as f:
      params = json.load(f)
      self.reference_eos_id = params["featurizer_ids"][0]
    self.params = self._load_params()
    self.read_store = self.params.get("read_store")
    self.contribute_store = self.params.get("contribute_store")
    self.nns = bool(self.params.get("enable_nns", False))
    self.version = None

  def _load_params(self):
    try:
      with open(os.path.join(self.path, DATA_SUBFOLDER, PARAMETERS_FILE), "r") as f:
        params = json.load(f)
      return params
    except Exception as e:
      logger.error(f"Error loading params from disk: {e}")
      raise

  def _save_params(self, params):
    try:
      with open(os.path.join(self.path, DATA_SUBFOLDER, PARAMETERS_FILE), "w") as f:
        json.dump(params, f, indent=2)
    except Exception as e:
      logger.error(f"Error saving params to disk: {e}")
      raise

  def resolve_version(self, model_id, bucket):
    try:
      if "latest_projection_version" not in self.params:
        self.params["latest_projection_version"] = {}

      if model_id in self.params["latest_projection_version"]:
        return self.params["latest_projection_version"][model_id]

      version = latest_version(model_id, bucket)
      self.params["latest_projection_version"][model_id] = version
      self._save_params(self.params)
      return version
    except Exception as e:
      logger.error(f"Error resolving version for model {model_id}: {e}")
      raise

  def _load_data(self):
    with open(self.input_file, "r") as f:
      reader = csv.reader(f)
      h = next(reader)
      return ([row[0] for row in reader], h)

  def load(self, file_name):
    return joblib.load(file_name)

  def save(self, obj, file_name):
    joblib.dump(obj, file_name)

  def finalize(self, algo_name):
    algo_path = os.path.join(self.trained_path, algo_name + ".joblib")
    self.save(self.X, algo_path)

    file_name = os.path.join(self.path, DESCRIPTORS_SUBFOLDER, algo_name + ".h5")
    data = Data()
    data.set(inputs=self.inputs, values=self.X, features=None)
    Hdf5(file_name).save(data)
    data.save_info(file_name.split(".")[0] + ".json")

  def group_by_prefix(self, names):
    groups = {}
    for name in names:
      prefix, *_ = name.split("_", 1)
      groups.setdefault(prefix, []).append(name)
    return groups

  def _get_ersilia_df(self, inputs, data, cols):
    try:
      keys = [hashlib.md5(inp.encode()).hexdigest() for inp in inputs]
      df = pd.DataFrame({"key": keys, "input": inputs})
      data_df = pd.DataFrame(data, columns=cols, dtype=str)
      return df.join(data_df)
    except Exception as e:
      logger.error(f"Error creating Ersilia DataFrame: {e}")
      raise

  def _contribute(self, inputs, data, cols):
    if not self.contribute_store:
      return
    df = self._get_ersilia_df(inputs, data, cols)
    is_temp = self.contribute_store == ZAIRATEMP_BUCKET
    write_bucket = ZAIRATEMP_BUCKET if is_temp else self.contribute_store
    try:
      logger.info(f"Writing projection precalculations to bucket: {write_bucket}")
      w = IsauraWriter(
        input_csv=None,
        model_id=self.model_id,
        model_version=self.version,
        bucket=write_bucket,
      )
      w.write(df=df)
      if is_temp:
        logger.info(f"Copying precalculations from {ZAIRATEMP_BUCKET} to isaura-public")
        c = IsauraCopy(
          model_id=self.model_id,
          model_version=self.version,
          bucket=ZAIRATEMP_BUCKET,
        )
        c.copy()
        logger.info(f"Removing temporary data from {ZAIRATEMP_BUCKET}")
        r = IsauraRemover(
          model_id=self.model_id,
          model_version=self.version,
          bucket=ZAIRATEMP_BUCKET,
        )
        r.remove()
    except Exception as e:
      logger.error(f"Error in Isaura contribute workflow for projections: {e}")

  def _run_api(self):
    logger.info(f"Computing projections via API: {self.url}")
    data = post(self.inputs, self.url)
    cols = fetch_schema_from_github(self.model_id)[0]
    return data, cols

  def run(self):
    self.inputs = self._load_data()[0]
    data = None
    cols = None

    if self.contribute_store or self.read_store:
      self.version = self.resolve_version(self.model_id, self.read_store)

    if not self.read_store:
      logger.warning(f"Isaura read store is disabled for projections (no -es flag provided)!")
      data, cols = self._run_api()
    else:
      try:
        logger.info(f"Reading projection precalculations from bucket: {self.read_store}")
        df = pd.DataFrame(columns=["input"], data=self.inputs)
        r = IsauraReader(
          model_id=self.model_id,
          model_version=self.version,
          input_csv=None,
          approximate=self.nns,
          bucket=self.read_store,
        )
        df = r.read(df=df)
        cols = df.columns.difference(["key", "input"]).tolist()
        values = df[cols].values
        data = [{col: row[i] for i, col in enumerate(cols)} for row in values]
      except SystemExit as e:
        logger.info(f"Fall back to API for computing projections due to SystemExit: {e}")
        data, cols = self._run_api()
      except Exception as e:
        logger.error(f"Unhandled error reading projection cache: {e}")
        raise

    self._contribute(self.inputs, [[d[col] for col in cols] for d in data], cols)

    algos = self.group_by_prefix(cols)
    self.trained_path = os.path.join(self.path, DESCRIPTORS_SUBFOLDER)
    for algo, dims in algos.items():
      self.X = [[d[dim] for dim in dims] for d in data]
      self.finalize(algo)
