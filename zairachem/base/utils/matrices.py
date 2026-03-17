import collections, h5py, json, joblib, gc, os, glob
import numpy as np
import pandas as pd
from typing import Iterator, Tuple, List, Optional, Union
from contextlib import contextmanager
from zairachem.base.utils.logging import logger

SNIFF_N = 100000
DEFAULT_CHUNK_SIZE = 10000
_CHUNK_META_FILE = "chunks_meta.json"
_CHUNK_PREFIX = "chunk_"


class Data(object):
  def __init__(self):
    self._is_sparse = None

  def _arbitrary_features(self, n):
    return ["f{0}".format(i) for i in range(n)]

  def set(self, inputs, values, features):
    self._inputs = inputs
    self._values = values
    if features is None:
      self._features = self._arbitrary_features(len(values[0]))
    else:
      self._features = features

  def inputs(self):
    return self._inputs

  def values(self):
    return self._values

  def features(self):
    return self._features

  def is_sparse(self):
    return self._is_sparse

  def save(self, file_name):
    joblib.dump(self, file_name)

  def load(self, file_name):
    return joblib.load(file_name)

  def save_info(self, file_name):
    info = {
      "inputs": len(self._inputs),
      "features": len(self._features),
      "values": np.array(self._values).shape,
      "is_sparse": self._is_sparse,
    }
    with open(file_name, "w") as f:
      json.dump(info, f, indent=4)


class Hdf5(object):
  def __init__(self, file_name):
    self.file_name = file_name

  @contextmanager
  def _open(self, mode="r"):
    f = h5py.File(self.file_name, mode)
    try:
      yield f
    finally:
      f.close()

  def shape(self):
    with self._open("r") as f:
      return f["Values"].shape

  def n_rows(self):
    return self.shape()[0]

  def n_features(self):
    return self.shape()[1]

  def values(self):
    with self._open("r") as f:
      return f["Values"][:]

  def values_slice(self, start, end):
    with self._open("r") as f:
      return f["Values"][start:end]

  def iter_values(self, chunk_size=DEFAULT_CHUNK_SIZE) -> Iterator[np.ndarray]:
    with self._open("r") as f:
      ds = f["Values"]
      n = ds.shape[0]
      for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        logger.debug(f"[h5:iter] {self.file_name} rows {start}-{end}/{n}")
        yield ds[start:end]

  def iter_values_with_indices(
    self, chunk_size=DEFAULT_CHUNK_SIZE
  ) -> Iterator[Tuple[int, int, np.ndarray]]:
    with self._open("r") as f:
      ds = f["Values"]
      n = ds.shape[0]
      for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        logger.debug(f"[h5:iter] {self.file_name} rows {start}-{end}/{n}")
        yield start, end, ds[start:end]

  def iter_inputs(self, chunk_size=DEFAULT_CHUNK_SIZE) -> Iterator[List[str]]:
    with self._open("r") as f:
      ds = f["Inputs"]
      n = ds.shape[0]
      for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = [x.decode("utf-8") for x in ds[start:end]]
        yield chunk

  def iter_all(
    self, chunk_size=DEFAULT_CHUNK_SIZE
  ) -> Iterator[Tuple[int, int, np.ndarray, List[str]]]:
    with self._open("r") as f:
      ds_v = f["Values"]
      ds_i = f["Inputs"]
      n = ds_v.shape[0]
      for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        values = ds_v[start:end]
        inputs = [x.decode("utf-8") for x in ds_i[start:end]]
        logger.debug(f"[h5:iter_all] {self.file_name} rows {start}-{end}/{n}")
        yield start, end, values, inputs

  def inputs(self):
    with self._open("r") as f:
      return [x.decode("utf-8") for x in f["Inputs"][:]]

  def inputs_slice(self, start, end) -> List[str]:
    with self._open("r") as f:
      return [x.decode("utf-8") for x in f["Inputs"][start:end]]

  def features(self):
    with self._open("r") as f:
      return [x.decode("utf-8") for x in f["Features"][:]]

  def _sniff_ravel(self):
    with self._open("r") as f:
      V = f["Values"][:SNIFF_N]
    return V.ravel()

  def is_sparse(self):
    V = self._sniff_ravel()
    n_zeroes = np.sum(V == 0)
    if n_zeroes / len(V) > 0.8:
      return True
    return False

  def is_binary(self):
    V = self._sniff_ravel()
    vals = set(V)
    if len(vals) > 2:
      return False
    return True

  def is_dense(self):
    return not self.is_sparse()

  def load(self):
    data = Data()
    data.set(
      inputs=self.inputs(),
      values=self.values(),
      features=self.features(),
    )
    data._is_sparse = self.is_sparse()
    return data

  def save(self, data):
    with self._open("w") as f:
      f.create_dataset("Values", data=data.values())
      f.create_dataset("Inputs", data=np.array(data.inputs(), h5py.string_dtype()))
      f.create_dataset("Features", data=np.array(data.features(), h5py.string_dtype()))

  def create_empty(self, n_features, features, dtype="float32"):
    str_dt = h5py.string_dtype(encoding="utf-8")
    chunk_rows = min(DEFAULT_CHUNK_SIZE, max(1, n_features))
    with self._open("w") as f:
      f.create_dataset(
        "Values",
        shape=(0, n_features),
        maxshape=(None, n_features),
        dtype=dtype,
        chunks=(chunk_rows, n_features),
      )
      f.create_dataset("Inputs", shape=(0,), maxshape=(None,), dtype=str_dt)
      f.create_dataset("Features", data=np.array(features, dtype=str_dt))
    logger.info(f"[h5:create] {self.file_name} features={n_features}")

  def append(self, values, inputs):
    n_new = values.shape[0]
    if n_new == 0:
      return
    str_dt = h5py.string_dtype(encoding="utf-8")
    with self._open("a") as f:
      ds_v = f["Values"]
      ds_i = f["Inputs"]
      old_n = ds_v.shape[0]
      ds_v.resize(old_n + n_new, axis=0)
      ds_v[old_n:] = values
      ds_i.resize(old_n + n_new, axis=0)
      ds_i[old_n:] = np.array(inputs, dtype=str_dt)
    logger.debug(f"[h5:append] {self.file_name} +{n_new} rows (total={old_n + n_new})")

  def update_slice(self, start, end, values):
    with self._open("a") as f:
      f["Values"][start:end] = values

  def save_summary_as_csv(self):
    file_name = self.file_name.split(".h5")[0] + "_summary.csv"
    n_rows = self.n_rows()
    features = self.features()
    n_features = len(features)
    first_row = self.values_slice(0, 1)[0]
    last_row = self.values_slice(n_rows - 1, n_rows)[0]
    means = np.zeros(n_features)
    stds = np.zeros(n_features)
    counts = np.zeros(n_features)
    for chunk in self.iter_values():
      valid_mask = ~np.isnan(chunk)
      chunk_filled = np.where(valid_mask, chunk, 0)
      means += np.sum(chunk_filled, axis=0)
      counts += np.sum(valid_mask, axis=0)
    means = np.divide(means, counts, out=np.zeros_like(means), where=counts > 0)
    for chunk in self.iter_values():
      valid_mask = ~np.isnan(chunk)
      diff_sq = np.where(valid_mask, (chunk - means) ** 2, 0)
      stds += np.sum(diff_sq, axis=0)
    stds = np.sqrt(np.divide(stds, counts, out=np.zeros_like(stds), where=counts > 0))
    columns = ["keys"] + features
    data = collections.defaultdict(list)
    data["keys"] = ["first", "last", "mean", "std"]
    for i, feat in enumerate(features):
      data[feat] += [float(first_row[i]), float(last_row[i]), means[i], stds[i]]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(file_name, index=False)


class ChunkedH5Processor:
  def __init__(self, input_h5: str, output_h5: str = None, chunk_size: int = DEFAULT_CHUNK_SIZE):
    self.input_h5 = Hdf5(input_h5)
    self.output_h5 = Hdf5(output_h5) if output_h5 else None
    self.chunk_size = chunk_size

  def transform_chunked(
    self,
    transform_fn,
    fit_fn=None,
    n_output_features: int = None,
    output_features: List[str] = None,
    dtype: str = "float32",
  ):
    n_rows = self.input_h5.n_rows()
    n_input_features = self.input_h5.n_features()
    if n_output_features is None:
      n_output_features = n_input_features
    if output_features is None:
      output_features = self.input_h5.features()[:n_output_features]
    logger.info(f"[chunked] Processing {n_rows} rows in chunks of {self.chunk_size}")
    if fit_fn is not None:
      logger.info("[chunked] Running fit pass...")
      for start, end, values in self.input_h5.iter_values_with_indices(self.chunk_size):
        fit_fn(values, is_first=(start == 0), is_last=(end >= n_rows))
    if self.output_h5:
      self.output_h5.create_empty(n_output_features, output_features, dtype=dtype)
      logger.info("[chunked] Running transform pass...")
      for start, end, values, inputs in self.input_h5.iter_all(self.chunk_size):
        transformed = transform_fn(values)
        self.output_h5.append(transformed.astype(dtype), inputs)
        gc.collect()
      logger.info(f"[chunked] Completed. Output shape: {self.output_h5.shape()}")

  def apply_column_filter(self, col_indices: np.ndarray, output_features: List[str] = None):
    if output_features is None:
      input_features = self.input_h5.features()
      output_features = [input_features[i] for i in col_indices]

    def transform_fn(chunk):
      return chunk[:, col_indices]

    self.transform_chunked(
      transform_fn=transform_fn, n_output_features=len(col_indices), output_features=output_features
    )


class IncrementalStats:
  def __init__(self, n_features: int):
    self.n_features = n_features
    self.count = np.zeros(n_features, dtype=np.float64)
    self.mean = np.zeros(n_features, dtype=np.float64)
    self.M2 = np.zeros(n_features, dtype=np.float64)
    self.min_val = np.full(n_features, np.inf, dtype=np.float64)
    self.max_val = np.full(n_features, -np.inf, dtype=np.float64)

  def update(self, chunk: np.ndarray):
    for j in range(self.n_features):
      col = chunk[:, j]
      valid = col[~np.isnan(col)]
      if len(valid) == 0:
        continue
      for x in valid:
        self.count[j] += 1
        delta = x - self.mean[j]
        self.mean[j] += delta / self.count[j]
        delta2 = x - self.mean[j]
        self.M2[j] += delta * delta2
      self.min_val[j] = min(self.min_val[j], np.min(valid))
      self.max_val[j] = max(self.max_val[j], np.max(valid))

  def update_batch(self, chunk: np.ndarray):
    valid_mask = ~np.isnan(chunk)
    for j in range(self.n_features):
      col = chunk[:, j]
      mask = valid_mask[:, j]
      valid = col[mask]
      n = len(valid)
      if n == 0:
        continue
      batch_mean = np.mean(valid)
      batch_var = np.var(valid) if n > 1 else 0.0
      batch_count = n
      if self.count[j] == 0:
        self.mean[j] = batch_mean
        self.M2[j] = batch_var * batch_count
        self.count[j] = batch_count
      else:
        delta = batch_mean - self.mean[j]
        total_count = self.count[j] + batch_count
        self.mean[j] = (self.count[j] * self.mean[j] + batch_count * batch_mean) / total_count
        self.M2[j] += batch_var * batch_count + delta**2 * self.count[j] * batch_count / total_count
        self.count[j] = total_count
      self.min_val[j] = min(self.min_val[j], np.min(valid))
      self.max_val[j] = max(self.max_val[j], np.max(valid))

  @property
  def variance(self) -> np.ndarray:
    return np.divide(self.M2, self.count, out=np.zeros(self.n_features), where=self.count > 1)

  @property
  def std(self) -> np.ndarray:
    return np.sqrt(self.variance)

  @property
  def median_estimate(self) -> np.ndarray:
    return self.mean


class ChunkedNanFilter:
  def __init__(self, max_na_ratio: float = 0.2):
    self.max_na_ratio = max_na_ratio
    self.col_idxs = None
    self._na_counts = None
    self._total_rows = 0

  def fit_chunk(self, chunk: np.ndarray, is_first: bool = False, is_last: bool = False):
    if is_first:
      self._na_counts = np.zeros(chunk.shape[1], dtype=np.int64)
      self._total_rows = 0
    self._na_counts += np.sum(np.isnan(chunk), axis=0)
    self._total_rows += chunk.shape[0]
    if is_last:
      max_na = int(self.max_na_ratio * self._total_rows)
      self.col_idxs = np.where(self._na_counts <= max_na)[0]
      logger.info(f"[nan_filter] Kept {len(self.col_idxs)}/{len(self._na_counts)} columns")

  def transform(self, chunk: np.ndarray) -> np.ndarray:
    return chunk[:, self.col_idxs]


class ChunkedImputer:
  def __init__(self, fallback: float = 0.0):
    self.fallback = fallback
    self.impute_values = None
    self._stats = None

  def fit_chunk(self, chunk: np.ndarray, is_first: bool = False, is_last: bool = False):
    if is_first:
      self._stats = IncrementalStats(chunk.shape[1])
    self._stats.update_batch(chunk)
    if is_last:
      self.impute_values = np.where(
        self._stats.count > 0, self._stats.median_estimate, self.fallback
      )
      logger.info(f"[imputer] Computed impute values for {len(self.impute_values)} features")

  def transform(self, chunk: np.ndarray) -> np.ndarray:
    result = chunk.copy()
    for j in range(result.shape[1]):
      mask = np.isnan(result[:, j])
      result[mask, j] = self.impute_values[j]
    return result


class ChunkedVarianceFilter:
  def __init__(self, threshold: float = 0.0):
    self.threshold = threshold
    self.col_idxs = None
    self._stats = None

  def fit_chunk(self, chunk: np.ndarray, is_first: bool = False, is_last: bool = False):
    if is_first:
      self._stats = IncrementalStats(chunk.shape[1])
    self._stats.update_batch(chunk)
    if is_last:
      variances = self._stats.variance
      self.col_idxs = np.where(variances > self.threshold)[0]
      logger.info(f"[variance_filter] Kept {len(self.col_idxs)}/{len(variances)} columns")

  def transform(self, chunk: np.ndarray) -> np.ndarray:
    return chunk[:, self.col_idxs]


class ChunkedScaler:
  def __init__(self, abs_limit: float = 10.0):
    self.abs_limit = abs_limit
    self.center = None
    self.scale = None
    self.skip = False
    self._stats = None

  def set_skip(self):
    self.skip = True

  def fit_chunk(self, chunk: np.ndarray, is_first: bool = False, is_last: bool = False):
    if self.skip:
      return
    if is_first:
      self._stats = IncrementalStats(chunk.shape[1])
    self._stats.update_batch(chunk)
    if is_last:
      self.center = self._stats.mean
      self.scale = self._stats.std
      self.scale[self.scale == 0] = 1.0
      logger.info(f"[scaler] Fitted on {int(self._stats.count[0])} samples")

  def transform(self, chunk: np.ndarray) -> np.ndarray:
    if self.skip:
      return chunk
    result = (chunk - self.center) / self.scale
    return np.clip(result, -self.abs_limit, self.abs_limit)


def open_h5(path: str):
  store = ChunkedH5Store(path)
  if store.exists():
    return store
  if os.path.exists(path):
    return Hdf5(path)
  return None


class ChunkedH5Store:

  def __init__(self, base_path: str):
    if base_path.endswith(".h5"):
      self.dir = base_path.rsplit(".h5", 1)[0] + "_chunks"
      self.legacy_path = base_path
    else:
      self.dir = base_path
      self.legacy_path = None
    self._meta_path = os.path.join(self.dir, _CHUNK_META_FILE)
    self._chunk_idx = 0

  def _chunk_path(self, idx: int) -> str:
    return os.path.join(self.dir, f"{_CHUNK_PREFIX}{idx:04d}.h5")

  def _read_meta(self) -> dict:
    if os.path.exists(self._meta_path):
      with open(self._meta_path, "r") as f:
        return json.load(f)
    return {"n_chunks": 0, "total_rows": 0, "n_features": 0, "features": []}

  def _write_meta(self, meta: dict):
    with open(self._meta_path, "w") as f:
      json.dump(meta, f, indent=2)

  def exists(self) -> bool:
    return os.path.exists(self._meta_path) and self._read_meta()["n_chunks"] > 0

  def n_chunks(self) -> int:
    return self._read_meta()["n_chunks"]

  def n_rows(self) -> int:
    return self._read_meta()["total_rows"]

  def n_features(self) -> int:
    return self._read_meta()["n_features"]

  def features(self) -> List[str]:
    return self._read_meta()["features"]

  def shape(self) -> Tuple[int, int]:
    meta = self._read_meta()
    return (meta["total_rows"], meta["n_features"])

  def create(self, n_features: int, features: List[str]):
    os.makedirs(self.dir, exist_ok=True)
    meta = {"n_chunks": 0, "total_rows": 0, "n_features": n_features, "features": features}
    self._write_meta(meta)
    self._chunk_idx = 0
    logger.info(f"[h5store:create] {self.dir} features={n_features}")

  def save_chunk(self, values: np.ndarray, inputs: List[str]):
    n_new = values.shape[0]
    if n_new == 0:
      return
    meta = self._read_meta()
    chunk_path = self._chunk_path(meta["n_chunks"])
    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(chunk_path, "w") as f:
      f.create_dataset("Values", data=values)
      f.create_dataset("Inputs", data=np.array(inputs, dtype=str_dt))
      f.create_dataset("Features", data=np.array(meta["features"], dtype=str_dt))
    meta["n_chunks"] += 1
    meta["total_rows"] += n_new
    self._write_meta(meta)
    logger.debug(f"[h5store:save_chunk] {chunk_path} rows={n_new} (total={meta['total_rows']})")

  def _sorted_chunk_paths(self) -> List[str]:
    meta = self._read_meta()
    return [self._chunk_path(i) for i in range(meta["n_chunks"])]

  def iter_chunks(self) -> Iterator[Tuple[np.ndarray, List[str]]]:
    for path in self._sorted_chunk_paths():
      with h5py.File(path, "r") as f:
        values = f["Values"][:]
        inputs = [x.decode("utf-8") for x in f["Inputs"][:]]
      yield values, inputs

  def iter_values(self) -> Iterator[np.ndarray]:
    for path in self._sorted_chunk_paths():
      with h5py.File(path, "r") as f:
        yield f["Values"][:]

  def iter_values_with_indices(self) -> Iterator[Tuple[int, int, np.ndarray]]:
    offset = 0
    for path in self._sorted_chunk_paths():
      with h5py.File(path, "r") as f:
        vals = f["Values"][:]
      end = offset + vals.shape[0]
      yield offset, end, vals
      offset = end

  def iter_all(self) -> Iterator[Tuple[int, int, np.ndarray, List[str]]]:
    offset = 0
    for path in self._sorted_chunk_paths():
      with h5py.File(path, "r") as f:
        vals = f["Values"][:]
        inputs = [x.decode("utf-8") for x in f["Inputs"][:]]
      end = offset + vals.shape[0]
      yield offset, end, vals, inputs
      offset = end

  def values(self) -> np.ndarray:
    parts = list(self.iter_values())
    if not parts:
      return np.empty((0, 0))
    return np.concatenate(parts, axis=0)

  def inputs(self) -> List[str]:
    all_inputs = []
    for _, inp in self.iter_chunks():
      all_inputs.extend(inp)
    return all_inputs

  def values_slice(self, start: int, end: int) -> np.ndarray:
    offset = 0
    parts = []
    for path in self._sorted_chunk_paths():
      with h5py.File(path, "r") as f:
        chunk_n = f["Values"].shape[0]
        chunk_end = offset + chunk_n
        if chunk_end <= start:
          offset = chunk_end
          continue
        if offset >= end:
          break
        local_start = max(0, start - offset)
        local_end = min(chunk_n, end - offset)
        parts.append(f["Values"][local_start:local_end])
        offset = chunk_end
    if not parts:
      return np.empty((0, 0))
    return np.concatenate(parts, axis=0)

  def is_sparse(self) -> bool:
    count = 0
    zeros = 0
    for vals in self.iter_values():
      flat = vals.ravel()
      sample = flat[:SNIFF_N - count] if count + len(flat) > SNIFF_N else flat
      zeros += np.sum(sample == 0)
      count += len(sample)
      if count >= SNIFF_N:
        break
    if count == 0:
      return False
    return zeros / count > 0.8

  def load(self) -> "Data":
    data = Data()
    data.set(
      inputs=self.inputs(),
      values=self.values(),
      features=self.features(),
    )
    data._is_sparse = self.is_sparse()
    return data

  def cleanup(self):
    for path in self._sorted_chunk_paths():
      if os.path.exists(path):
        os.remove(path)
    if os.path.exists(self._meta_path):
      os.remove(self._meta_path)
    if os.path.exists(self.dir) and not os.listdir(self.dir):
      os.rmdir(self.dir)
    logger.info(f"[h5store:cleanup] Removed {self.dir}")
