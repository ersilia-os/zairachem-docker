import csv
import json
import hashlib
import os
import time
import requests
import re
import pandas as pd
import numpy as np
from zairachem.base.utils.logging import logger
from zairachem.base import ZairaBase
from zairachem.base.utils.utils import (
  fetch_schema_from_github,
  resolve_default_bucket,
  latest_version,
)
from zairachem.base.vars import DATA_SUBFOLDER, PARAMETERS_FILE
from urllib.parse import urlparse

try:
  from isaura.manage import IsauraCopy, IsauraReader, IsauraWriter
except Exception as e:
  logger.warning(f"Isaura modules could not be imported: {e}")
  IsauraCopy = None
  IsauraReader = None
  IsauraWriter = None

_RING_TOKEN_RE = re.compile(r"%(?:\d{2})|[0-9]")
_ALLOWED_RE = re.compile(
  r"^[A-Za-z0-9@+\-\[\]\(\)=#$:.\\/%,*\.]+(?:\s+[A-Za-z0-9@+\-\[\]\(\)=#$:.\\/%,*\.]+)?$"
)
DEFAULT_BATCH_SIZE = 100_000


class BinaryStreamClient(ZairaBase):
  def __init__(
    self, csv_path, model_id=None, url=None, project_name=None, batch_size=DEFAULT_BATCH_SIZE
  ):
    try:
      super(ZairaBase, self).__init__()
      self.logger = logger
      self.batch_size = batch_size
      self.csv_path = csv_path
      self.url = url
      self.model_id = model_id

      self.path = self.get_output_dir()
      self.input_data, self.input_header = self._load_data()
      self.params = self._load_params()
      self.access = self.params["access"]
      self.enable_cache = bool(self.params["enable_cache"])
      self.default_bucket = resolve_default_bucket(self.access)
      self.nns = bool(self.params["enable_nns"])
      self.contribute_cache = bool(self.params["contribute_cache"])
      self._feature_len = None
      self.project_name = project_name
    except Exception as e:
      logger.error(f"Error during BinaryStreamClient initialization: {e}")
      raise

  def resolve_dtype(self):
    self.schema = fetch_schema_from_github(self.model_id)
    return "float" if "float" in set(self.schema[1]) else "int"

  def resolve_dims(self):
    self.schema = fetch_schema_from_github(self.model_id)
    return self.schema[0]

  def resolve_version(self, model_id, bucket):
    try:
      if "latest_featurizer_version" not in self.params:
        self.params["latest_featurizer_version"] = {}

      if model_id in self.params["latest_featurizer_version"]:
        return self.params["latest_featurizer_version"][model_id]

      version = latest_version(model_id, bucket)
      self.params["latest_featurizer_version"][model_id] = version
      self._save_params(self.params)
      return version
    except Exception as e:
      logger.error(f"Error resolving version for model {model_id}: {e}")
      raise

  def _save_params(self, params):
    try:
      with open(os.path.join(self.path, DATA_SUBFOLDER, PARAMETERS_FILE), "w") as f:
        json.dump(params, f, indent=2)
    except Exception as e:
      logger.error(f"Error saving params to disk: {e}")
      raise

  def _load_params(self):
    try:
      with open(os.path.join(self.path, DATA_SUBFOLDER, PARAMETERS_FILE), "r") as f:
        params = json.load(f)
      return params
    except Exception as e:
      logger.error(f"Error loading params from disk: {e}")
      raise

  def _load_data(self):
    try:
      with open(self.csv_path, "r") as f:
        reader = csv.reader(f)
        h = next(reader)
        return ([row[0] for row in reader], h)
    except Exception as e:
      logger.error(f"Error loading data from CSV {self.csv_path}: {e}")
      raise

  def decode_binary_stream(self, response, chunk_size=8192):
    try:
      it = response.iter_content(chunk_size=chunk_size)
      header_buf = bytearray()
      for chunk in it:
        header_buf.extend(chunk)
        if b"\n" in header_buf:
          header_line, remainder = header_buf.split(b"\n", 1)
          break
      else:
        raise IOError("Stream ended before header")
      meta = json.loads(header_line.decode("utf-8"))
      dtype = np.dtype(meta["dtype"])
      shape = tuple(meta["shape"])
      total_bytes = int(np.prod(shape) * dtype.itemsize)
      arr = np.empty(shape, dtype=dtype)
      view = memoryview(arr).cast("B")
      read = len(remainder)
      view[:read] = remainder

      for chunk in it:
        view[read : read + len(chunk)] = chunk
        read += len(chunk)
        if read >= total_bytes:
          break

      if read < total_bytes:
        raise IOError(f"Incomplete read: got {read} of {total_bytes} bytes")
      return arr, meta
    except Exception as e:
      logger.error(f"Error decoding binary stream: {e}")
      raise

  def _strip_brackets(self, s):
    try:
      return re.sub(r"\[[^\[\]]*\]", "", s)
    except Exception as e:
      logger.error(f"Error stripping brackets from string: {e}")
      raise

  def _balanced(self, s, open_ch, close_ch):
    try:
      c = 0
      for ch in s:
        if ch == open_ch:
          c += 1
        elif ch == close_ch:
          c -= 1
          if c < 0:
            return False
      return c == 0
    except Exception as e:
      logger.error(f"Error checking balance for string: {e}")
      raise

  def _rings_even(self, s):
    try:
      t = self._strip_brackets(s)
      tokens = _RING_TOKEN_RE.findall(t)
      counts = {}
      for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1
      return all(v % 2 == 0 for v in counts.values())
    except Exception as e:
      logger.error(f"Error checking ring parity for string: {e}")
      raise

  def _is_smiles(self, text):
    try:
      if not isinstance(text, str):
        return False
      s = text.strip()
      if not s or "\n" in s or "\t" in s or "  " in s:
        return False
      if not _ALLOWED_RE.fullmatch(s):
        return False
      if not self._balanced(s, "(", ")"):
        return False
      if not self._balanced(s, "[", "]"):
        return False
      if "()" in s or "[]]" in s or "[[" in s:
        return False
      if not self._rings_even(s):
        return False
      return True
    except Exception as e:
      logger.error(f"Error validating SMILES string: {e}")
      raise

  def _placeholder_row(self):
    try:
      if self._feature_len is None:
        self._feature_len = len(fetch_schema_from_github(self.model_id)[0])
      return np.full((1, self._feature_len), np.nan, dtype=float)
    except Exception as e:
      logger.error(f"Error creating placeholder row: {e}")
      raise

  def _root_from_url(self):
    p = urlparse(self.url)
    host = p.hostname or "localhost"
    port = p.port or (443 if (p.scheme or "http") == "https" else 80)
    scheme = p.scheme or "http"
    return f"{scheme}://{host}:{port}"

  def _ensure_ready(self, attempts=60, sleep_s=0.5):
    if getattr(self, "_ready", False):
      return
    root = self._root_from_url()
    for i in range(1, attempts + 1):
      try:
        r = requests.get(root, timeout=2)
        if 200 <= r.status_code < 500:
          self._ready = True
          self.logger.info(f"Probe OK {root} on attempt {i}")
          return
        self.logger.info(f"Probe {root} attempt {i} got {r.status_code}")
      except Exception as e:
        self.logger.info(f"Probe {root} attempt {i} error: {e}")
      time.sleep(sleep_s)
    raise RuntimeError(f"Server not ready after {attempts} attempts at {root}")

  def _try_request(self, batch):
    try:
      self._ensure_ready()
      params = {"output_type": "heavy"}
      headers = {
        "Content-Type": "application/json",
        "Accept": "*/*",
        "Connection": "close",
      }
      payload = json.dumps(batch, separators=(",", ":"))
      response = requests.post(
        self.url, params=params, data=payload, headers=headers, stream=True, timeout=(10, 300)
      )
      response.raise_for_status()
      array, results = self.decode_binary_stream(response)
      if self._feature_len is None:
        arr0 = array[0] if isinstance(array, (list, tuple)) else array
        arr0 = np.asarray(arr0)
        self._feature_len = arr0.shape[-1] if arr0.ndim > 1 else arr0.shape[0]
      return array, results
    except Exception as e:
      logger.error(f"Error in _try_request for batch of size {len(batch)}: {e}")
      raise

  def _fetch_or_split(self, batch, idx_offset, depth=0):
    try:
      try:
        array, results = self._try_request(batch)
        if isinstance(array, np.ndarray):
          arrays_out = [array[i : i + 1] for i in range(array.shape[0])]
        else:
          arrays_out = [np.asarray(row).reshape(1, -1) for row in array]
        return arrays_out, [True] * len(arrays_out), results
      except (requests.RequestException, IOError) as e:
        logger.error(
          f"Request/IO error in _fetch_or_split at depth {depth}, idx_offset {idx_offset}: {e}"
        )
        if len(batch) == 1:
          logger.error(f"Failed to process single-element batch at idx_offset {idx_offset}")
          return [None], [False], None
        mid = len(batch) // 2
        left_arrays, left_mask, left_res = self._fetch_or_split(batch[:mid], idx_offset, depth + 1)
        right_arrays, right_mask, right_res = self._fetch_or_split(
          batch[mid:], idx_offset + mid, depth + 1
        )
        return left_arrays + right_arrays, left_mask + right_mask, right_res or left_res
    except Exception as e:
      logger.error(
        f"Unexpected error in _fetch_or_split at depth {depth}, idx_offset {idx_offset}: {e}"
      )
      raise

  def run(self):
    if not self.enable_cache:
      logger.warning(f"Isaura store operation is disabled globally!")
      return self._run()
    try:
      self.version = self.resolve_version(self.model_id, self.default_bucket)
      df = pd.DataFrame(columns=["input"], data=self.input_data)
      r = IsauraReader(
        model_id=self.model_id,
        model_version=self.version,
        input_csv=None,
        approximate=self.nns,
        bucket=self.default_bucket,
      )
      df = r.read(df=df)
      values = df[df.columns.difference(["key", "input"])].values
      values = values.astype("float")
      cols = df.columns.difference(["key", "input"]).tolist()
      self.schema = fetch_schema_from_github(self.model_id)
      dtype = self.resolve_dtype()
      any_results = {
        "shape": values.shape,
        "dtype": dtype,
        "data": values,
        "dims": cols,
        "inputs": self.input_data,
      }
      return any_results
    except SystemExit as e:
      logger.info(f"Fall back to the api for calculating the descriptors due to SystemExit: {e}")
      return self._run()
    except Exception as e:
      logger.error(f"Unhandled error in run: {e}")
      raise

  def _get_ersilia_df(self, res):
    try:
      keys = [hashlib.md5(inp.encode()).hexdigest() for inp in res["inputs"]]
      return pd.DataFrame({"key": keys, "input": res["inputs"]}).join(
        pd.DataFrame(res["data"], columns=res["dims"], dtype=str)
      )
    except Exception as e:
      logger.error(f"Error creating Ersilia DataFrame: {e}")
      raise

  def _run(self):
    try:
      total_time = 0.0
      n = len(self.input_data)
      per_item_rows = [None] * n
      valid_mask_input = [False] * n
      good_idx = []
      checked_input = []
      for i, s in enumerate(self.input_data):
        ok = self._is_smiles(s)
        valid_mask_input[i] = ok
        if ok:
          good_idx.append(i)
          checked_input.append(s)

      any_results = None
      try:
        for start in range(0, len(checked_input), self.batch_size):
          batch = checked_input[start : start + self.batch_size]
          batch_abs_offset = good_idx[start]
          t0 = time.perf_counter()

          arrays_out, _, results = self._fetch_or_split(batch, idx_offset=batch_abs_offset)
          total_time += time.perf_counter() - t0
          if results is not None:
            any_results = results
          for j, row in enumerate(arrays_out):
            per_item_rows[good_idx[start + j]] = row
      finally:
        self.logger.info(f"Total elapsed: {total_time:.4f}s")
      filled = []
      for row in per_item_rows:
        if isinstance(row, np.ndarray):
          filled.append(row)
        else:
          filled.append(self._placeholder_row())
      stacked = np.vstack(filled)
      if any_results is None:
        any_results = {}
      if "dtype" not in any_results:
        any_results.update({"dtype": self.resolve_dtype()})
      if "dims" not in any_results:
        any_results.update({"dims": self.resolve_dims()})
      any_results.update({
        "shape": stacked.shape,
        "data": stacked,
        "inputs": self.input_data,
      })
      if self.enable_cache:
        df = self._get_ersilia_df(any_results)
        try:
          w = IsauraWriter(
            input_csv=None,
            model_id=self.model_id,
            model_version=self.version,
            bucket=self.project_name,
            access=self.access,
          )
          w.write(df=df)
          if self.contribute_cache:
            c = IsauraCopy(
              model_id=self.model_id, model_version=self.version, bucket=self.project_name
            )
            c.copy()
        except Exception as e:
          logger.error(f"Error writing to Isaura cache or copying data: {e}")
      return any_results
    except Exception as e:
      logger.error(f"Unhandled error in _run: {e}")
      raise
