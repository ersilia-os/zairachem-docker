import h5py, json, joblib, os
import numpy as np
from typing import Iterator, Tuple, List
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
      "inputs": int(len(self._inputs)),
      "features": int(len(self._features)),
      "values": [int(x) for x in np.array(self._values).shape],
      "is_sparse": bool(self._is_sparse) if self._is_sparse is not None else False,
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


def open_h5(path: str):
  store = ChunkedH5Store(path)
  if store.exists():
    logger.debug(f"[open_h5] Using chunked store at {store.dir}")
    return store
  if os.path.exists(path):
    logger.debug(f"[open_h5] Using legacy H5 at {path}")
    return Hdf5(path)
  logger.debug(f"[open_h5] No H5 data found at {path}")
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

  def iter_values(self, chunk_size=None) -> Iterator[np.ndarray]:
    for path in self._sorted_chunk_paths():
      with h5py.File(path, "r") as f:
        yield f["Values"][:]

  def iter_values_with_indices(self, chunk_size=None) -> Iterator[Tuple[int, int, np.ndarray]]:
    offset = 0
    for path in self._sorted_chunk_paths():
      with h5py.File(path, "r") as f:
        vals = f["Values"][:]
      end = offset + vals.shape[0]
      yield offset, end, vals
      offset = end

  def iter_all(self, chunk_size=None) -> Iterator[Tuple[int, int, np.ndarray, List[str]]]:
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
    # Read ONLY the Inputs dataset from each chunk — never touch Values, so collecting the input ids
    # doesn't pull the entire descriptor matrix into memory.
    all_inputs = []
    for path in self._sorted_chunk_paths():
      with h5py.File(path, "r") as f:
        all_inputs.extend(x.decode("utf-8") for x in f["Inputs"][:])
    return all_inputs

  def is_sparse(self) -> bool:
    count = 0
    zeros = 0
    for vals in self.iter_values():
      flat = vals.ravel()
      sample = flat[: SNIFF_N - count] if count + len(flat) > SNIFF_N else flat
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
