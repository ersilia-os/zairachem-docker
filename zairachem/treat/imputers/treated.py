import h5py, joblib, json, os, shutil, tempfile, gc
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold

from zairachem.treat.imputers import DescriptorBase
from zairachem.treat.fpsim2.searcher import SimilaritySearcher
from zairachem.base import ZairaBase
from zairachem.base.utils.matrices import (
  Hdf5,
  ChunkedH5Store,
  open_h5,
  ChunkedNanFilter,
  ChunkedImputer,
  ChunkedVarianceFilter,
  ChunkedScaler,
  DEFAULT_CHUNK_SIZE,
)
from zairachem.base.utils.logging import logger
from zairachem.base.vars import (
  DESCRIPTORS_SUBFOLDER,
  RAW_DESC_FILENAME,
  TREATED_DESC_FILENAME,
)

MAX_NA = 0.2


class NanFilter(object):
  def __init__(self):
    self._name = "nan_filter"
    self.logger = logger

  def __getstate__(self):
    state = self.__dict__.copy()
    state.pop("logger", None)
    return state

  def fit(self, X):
    max_na = int((1 - MAX_NA) * X.shape[0])
    idxs = []
    for j in range(X.shape[1]):
      c = np.sum(np.isnan(X[:, j]))
      if c > max_na:
        continue
      idxs += [j]
    self.col_idxs = idxs
    self.logger.info(
      "Nan filtering, original columns {0}, final columns {1}".format(
        X.shape[1], len(self.col_idxs)
      )
    )

  def transform(self, X):
    return X[:, self.col_idxs]

  def save(self, file_name):
    joblib.dump(self, file_name)

  def load(self, file_name):
    return joblib.load(file_name)


class Scaler(object):
  def __init__(self):
    self._name = "scaler"
    self.abs_limit = 10
    self.skip = False

  def set_skip(self):
    self.skip = True

  def fit(self, X):
    if self.skip:
      return
    self.scaler = RobustScaler()
    self.scaler.fit(X)

  def transform(self, X):
    if self.skip:
      return X
    X = self.scaler.transform(X)
    X = np.clip(X, -self.abs_limit, self.abs_limit)
    return X

  def save(self, file_name):
    joblib.dump(self, file_name)

  def load(self, file_name):
    return joblib.load(file_name)


class FullLineSimilarityImputer(object):
  def __init__(self):
    self._name = "full_line_imputer"
    self._prefix = "fp2sim"
    self._n_hits = 3
    self._sim_cutoff = 0.5

  def _set_filenames(self, basedir):
    self.basedir = basedir
    self.fp_filename = os.path.join(basedir, "{0}.h5".format(self._prefix))
    self.smiles_filename = os.path.join(basedir, "{0}.csv".format(self._prefix))
    self.x_notnan_filename = os.path.join(basedir, "{0}_X_notnan.h5".format(self._prefix))

  def _get_filenames(self):
    return (
      self.basedir,
      self.fp_filename,
      self.smiles_filename,
      self.x_notnan_filename,
    )

  def fit(self, X, smiles_list):
    idxs = []
    for i in range(X.shape[0]):
      if np.all(np.isnan(X[i, :])):
        continue
      idxs += [i]
    X_ = X[idxs]
    smiles_list_ = [smiles_list[i] for i in idxs]
    tmp_dir = tempfile.mkdtemp(prefix="ersilia-")
    self._set_filenames(tmp_dir)
    similarity_searcher = SimilaritySearcher(fp_filename=self.fp_filename)
    similarity_searcher.fit(smiles_list_)
    with h5py.File(self.x_notnan_filename, "w") as f:
      f.create_dataset("Values", data=X_)

  def transform(self, X, smiles_list):
    with h5py.File(self.x_notnan_filename, "r") as f:
      R = f["Values"][:]
    idxs = []
    for i in range(X.shape[0]):
      if np.all(np.isnan(X[i, :])):
        idxs += [i]
      else:
        continue
    similarity_searcher = SimilaritySearcher(fp_filename=self.fp_filename)
    for idx in idxs:
      smi = smiles_list[idx]
      hits = similarity_searcher.search(smi, cutoff=self._sim_cutoff)
      hits = np.array([x[0] for x in hits][: self._n_hits])
      if len(hits) == 0:
        hits = np.random.choice(
          [i for i in range(R.shape[0])],
          min(R.shape[0], self._n_hits),
          replace=False,
        )
      r = np.mean(R[hits, :], axis=0)
      X[idx, :] = r
    return X

  def save(self, file_name):
    _, fp_filename, smiles_filename, x_notnan_filename = self._get_filenames()
    basedir = os.path.dirname(file_name)
    self._set_filenames(basedir)
    shutil.move(fp_filename, self.fp_filename)
    shutil.move(smiles_filename, self.smiles_filename)
    shutil.move(x_notnan_filename, self.x_notnan_filename)
    joblib.dump(self, file_name)

  def load(self, file_name):
    imp = joblib.load(file_name)
    imp._set_filenames(os.path.dirname(file_name))
    return imp


class Imputer(object):
  def __init__(self):
    self._name = "imputer"
    self._fallback = 0

  def fit(self, X):
    ms = []
    for j in range(X.shape[1]):
      vals = X[:, j]
      mask = ~np.isnan(vals)
      vals = vals[mask]
      if len(vals) == 0:
        m = self._fallback
      else:
        m = np.median(vals)
      ms += [m]
    self.impute_values = np.array(ms)

  def transform(self, X):
    for j in range(X.shape[1]):
      mask = np.isnan(X[:, j])
      X[mask, j] = self.impute_values[j]
    return X

  def save(self, file_name):
    joblib.dump(self, file_name)

  def load(self, file_name):
    return joblib.load(file_name)


class VarianceFilter(object):
  def __init__(self):
    self._name = "variance_filter"

  def fit(self, X):
    self.sel = VarianceThreshold()
    self.sel.fit(X)

  def transform(self, X):
    return self.sel.transform(X)

  def save(self, file_name):
    joblib.dump(self, file_name)

  def load(self, file_name):
    return joblib.load(file_name)


class RawLoader(ZairaBase):
  def __init__(self):
    ZairaBase.__init__(self)
    self.path = self.get_output_dir()

  def open(self, eos_id):
    path = os.path.join(self.path, DESCRIPTORS_SUBFOLDER, eos_id, RAW_DESC_FILENAME)
    h5 = open_h5(path)
    if h5 is None:
      logger.warning(f"[RawLoader] No H5 data found at {path}")
    return h5


class TreatedLoader(ZairaBase):
  def __init__(self):
    ZairaBase.__init__(self)
    self.path = self.get_output_dir()

  def open(self, eos_id):
    path = os.path.join(self.path, DESCRIPTORS_SUBFOLDER, eos_id, TREATED_DESC_FILENAME)
    h5 = open_h5(path)
    if h5 is None:
      logger.warning(f"[TreatedLoader] No H5 data found at {path}")
    return h5


class ChunkedTreatedDescriptors(DescriptorBase):
  def __init__(self, chunk_size=DEFAULT_CHUNK_SIZE):
    DescriptorBase.__init__(self)
    self.logger = logger
    self.chunk_size = chunk_size
    self._name = TREATED_DESC_FILENAME
    self._is_predict = self.is_predict()

  def done_eos_iter(self):
    with open(os.path.join(self.path, DESCRIPTORS_SUBFOLDER, "done_eos.json"), "r") as f:
      data = json.load(f)
    for eos_id in data:
      yield eos_id

  def _fit_transformers_chunked(self, h5_path, trained_path):
    h5 = open_h5(h5_path)
    n_rows = h5.n_rows()
    n_features = h5.n_features()
    is_sparse = h5.is_sparse()
    self.logger.info(f"[treat:fit] Starting chunked fit on {n_rows} rows, {n_features} features")
    nan_filter = ChunkedNanFilter(max_na_ratio=MAX_NA)
    for start, end, chunk in h5.iter_values_with_indices(self.chunk_size):
      is_first = start == 0
      is_last = end >= n_rows
      nan_filter.fit_chunk(chunk, is_first=is_first, is_last=is_last)
      del chunk
      gc.collect()
    joblib.dump(nan_filter, os.path.join(trained_path, "nan_filter.joblib"))
    imputer = ChunkedImputer(fallback=0.0)
    for start, end, chunk in h5.iter_values_with_indices(self.chunk_size):
      is_first = start == 0
      is_last = end >= n_rows
      filtered_chunk = nan_filter.transform(chunk)
      imputer.fit_chunk(filtered_chunk, is_first=is_first, is_last=is_last)
      del chunk, filtered_chunk
      gc.collect()
    joblib.dump(imputer, os.path.join(trained_path, "imputer.joblib"))
    var_filter = ChunkedVarianceFilter(threshold=0.0)
    for start, end, chunk in h5.iter_values_with_indices(self.chunk_size):
      is_first = start == 0
      is_last = end >= n_rows
      filtered_chunk = nan_filter.transform(chunk)
      imputed_chunk = imputer.transform(filtered_chunk)
      var_filter.fit_chunk(imputed_chunk, is_first=is_first, is_last=is_last)
      del chunk, filtered_chunk, imputed_chunk
      gc.collect()
    joblib.dump(var_filter, os.path.join(trained_path, "variance_filter.joblib"))
    scaler = ChunkedScaler(abs_limit=10.0)
    if is_sparse:
      self.logger.info("Skipping normalization as data is sparse")
      scaler.set_skip()
    else:
      for start, end, chunk in h5.iter_values_with_indices(self.chunk_size):
        is_first = start == 0
        is_last = end >= n_rows
        filtered_chunk = nan_filter.transform(chunk)
        imputed_chunk = imputer.transform(filtered_chunk)
        var_filtered_chunk = var_filter.transform(imputed_chunk)
        scaler.fit_chunk(var_filtered_chunk, is_first=is_first, is_last=is_last)
        del chunk, filtered_chunk, imputed_chunk, var_filtered_chunk
        gc.collect()
    joblib.dump(scaler, os.path.join(trained_path, "scaler.joblib"))
    return nan_filter, imputer, var_filter, scaler

  def _load_transformers(self, trained_path):
    nan_filter = joblib.load(os.path.join(trained_path, "nan_filter.joblib"))
    imputer = joblib.load(os.path.join(trained_path, "imputer.joblib"))
    var_filter = joblib.load(os.path.join(trained_path, "variance_filter.joblib"))
    scaler = joblib.load(os.path.join(trained_path, "scaler.joblib"))
    return nan_filter, imputer, var_filter, scaler

  def _transform_chunked(self, h5_path, output_path, nan_filter, imputer, var_filter, scaler):
    h5_in = open_h5(h5_path)
    h5_out = ChunkedH5Store(output_path)
    n_rows = h5_in.n_rows()
    n_final_features = len(var_filter.col_idxs)
    input_features = h5_in.features()
    nan_filtered_features = [input_features[i] for i in nan_filter.col_idxs]
    output_features = [nan_filtered_features[i] for i in var_filter.col_idxs]
    h5_out.create(n_final_features, output_features)
    self.logger.info(f"[treat:transform] Processing {n_rows} rows -> {n_final_features} features")
    for start, end, values, inputs in h5_in.iter_all(self.chunk_size):
      x = nan_filter.transform(values)
      x = imputer.transform(x)
      x = var_filter.transform(x)
      x = scaler.transform(x)
      h5_out.save_chunk(x.astype("float32"), inputs)
      self.logger.debug(f"[treat] Processed rows {start}-{end}/{n_rows}")
      del values, x
      gc.collect()
    self.logger.info(f"[treat] Completed. Output shape: {h5_out.shape()}")

  def run(self):
    rl = RawLoader()
    for eos_id in self.done_eos_iter():
      path = os.path.join(self.path, DESCRIPTORS_SUBFOLDER, eos_id)
      raw_h5_path = os.path.join(path, RAW_DESC_FILENAME)
      output_h5_path = os.path.join(path, TREATED_DESC_FILENAME)
      raw_h5 = open_h5(raw_h5_path)
      if raw_h5 is None:
        self.logger.warning(f"[treat] Skipping {eos_id}: no raw data found at {raw_h5_path}")
        continue
      if not self._is_predict:
        trained_path = path
      else:
        trained_path = os.path.join(self.trained_path, DESCRIPTORS_SUBFOLDER, eos_id)
      self.logger.info(f"[treat] Processing {eos_id} ({type(raw_h5).__name__})")
      if not self._is_predict:
        nan_filter, imputer, var_filter, scaler = self._fit_transformers_chunked(
          raw_h5_path, trained_path
        )
      else:
        nan_filter, imputer, var_filter, scaler = self._load_transformers(trained_path)
      self._transform_chunked(raw_h5_path, output_h5_path, nan_filter, imputer, var_filter, scaler)
      h5_out = open_h5(output_h5_path)
      info_path = output_h5_path.replace(".h5", ".json")
      info = {
        "inputs": h5_out.n_rows(),
        "features": h5_out.n_features(),
        "values": list(h5_out.shape()),
        "is_sparse": h5_out.is_sparse(),
      }
      with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
      gc.collect()


class TreatedDescriptors(DescriptorBase):
  def __init__(self, chunk_size=DEFAULT_CHUNK_SIZE):
    DescriptorBase.__init__(self)
    self.logger = logger
    self.chunk_size = chunk_size
    self.common_pipeline = [(0, -1, NanFilter), (1, 0, Imputer), (2, 1, VarianceFilter)]
    self.scaled_pipeline = [(3, 2, Scaler)]
    self._name = TREATED_DESC_FILENAME
    self._is_predict = self.is_predict()

  def done_eos_iter(self):
    with open(os.path.join(self.path, DESCRIPTORS_SUBFOLDER, "done_eos.json"), "r") as f:
      data = json.load(f)
    for eos_id in data:
      yield eos_id

  def _should_use_chunked(self, h5_path):
    h5 = open_h5(h5_path)
    if h5 is None:
      return False
    try:
      n_rows = h5.n_rows()
      return n_rows > self.chunk_size * 2
    except Exception as e:
      self.logger.warning(f"[treat] Could not check file size for {h5_path}: {e}")
      return False

  def run(self):
    rl = RawLoader()
    for eos_id in self.done_eos_iter():
      path = os.path.join(self.path, DESCRIPTORS_SUBFOLDER, eos_id)
      raw_h5_path = os.path.join(path, RAW_DESC_FILENAME)
      raw_h5 = open_h5(raw_h5_path)
      if raw_h5 is None:
        self.logger.warning(f"[treat] Skipping {eos_id}: no raw data found at {raw_h5_path}")
        continue
      if not self._is_predict:
        trained_path = path
      else:
        trained_path = os.path.join(self.trained_path, DESCRIPTORS_SUBFOLDER, eos_id)
      if self._should_use_chunked(raw_h5_path):
        self.logger.info(f"[treat] Using chunked processing for {eos_id}")
        chunked_processor = ChunkedTreatedDescriptors(chunk_size=self.chunk_size)
        chunked_processor.path = self.path
        chunked_processor.trained_path = self.trained_path
        chunked_processor._is_predict = self._is_predict
        output_h5_path = os.path.join(path, TREATED_DESC_FILENAME)
        if not self._is_predict:
          nan_filter, imputer, var_filter, scaler = chunked_processor._fit_transformers_chunked(
            raw_h5_path, trained_path
          )
        else:
          nan_filter, imputer, var_filter, scaler = chunked_processor._load_transformers(
            trained_path
          )
        chunked_processor._transform_chunked(
          raw_h5_path, output_h5_path, nan_filter, imputer, var_filter, scaler
        )
        h5_out = open_h5(output_h5_path)
        info_path = output_h5_path.replace(".h5", ".json")
        info = {
          "inputs": h5_out.n_rows(),
          "features": h5_out.n_features(),
          "values": list(h5_out.shape()),
          "is_sparse": h5_out.is_sparse(),
        }
        with open(info_path, "w") as f:
          json.dump(info, f, indent=4)
      else:
        self.logger.info(f"[treat] Using in-memory processing for {eos_id}")
        h5_data = raw_h5.load()
        X = {}
        X[-1] = h5_data.values()
        for step in self.common_pipeline:
          curr, prev = step[0], step[1]
          algo = step[-1]()
          if not self._is_predict:
            algo.fit(X[prev])
            algo.save(os.path.join(trained_path, algo._name + ".joblib"))
          else:
            algo = algo.load(os.path.join(trained_path, algo._name + ".joblib"))
          X[curr] = algo.transform(X[prev])
        final_idx = len(self.common_pipeline) - 1
        for step in self.scaled_pipeline:
          curr, prev = step[0], step[1]
          algo = step[-1]()
          if algo._name == "scaler":
            if h5_data.is_sparse():
              self.logger.info("Skipping normalization of {0} as it is sparse".format(eos_id))
              algo.set_skip()
            else:
              if not self._is_predict:
                algo.fit(X[prev])
                algo.save(os.path.join(trained_path, algo._name + ".joblib"))
              else:
                algo = algo.load(os.path.join(trained_path, algo._name + ".joblib"))
          X[curr] = algo.transform(X[prev])
        final_idx = len(self.common_pipeline) + len(self.scaled_pipeline) - 1
        file_name = os.path.join(self.path, DESCRIPTORS_SUBFOLDER, eos_id, self._name)
        h5_data._values = X[final_idx]
        Hdf5(file_name).save(h5_data)
        h5_data.save_info(file_name.split(".")[0] + ".json")
      gc.collect()
