import gc, json, os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

from eosframes.scale import transform as eosframes_transform

from zairachem.base import params_path
from zairachem.base.utils.concurrency import io_workers
from zairachem.treat.imputers import DescriptorBase
from zairachem.treat.imputers.reference_transformer import (
  fetch_reference_transformer,
  load_local_transformer,
  local_transformer_name,
  save_local_transformer,
  validate_transformer,
)
from zairachem.base.utils.matrices import (
  Data,
  Hdf5,
  ChunkedH5Store,
  open_h5,
  DEFAULT_CHUNK_SIZE,
)
from zairachem.base.utils.logging import logger
from zairachem.base.utils.model_version import ersilia_model_version
from zairachem.base.utils.progress import LiveTableMonitor, STEP_COLORS
from zairachem.base.vars import (
  DEFAULT_REFERENCE_LIBRARY,
  DESCRIPTORS_SUBFOLDER,
  RAW_DESC_FILENAME,
  TREATED_DESC_FILENAME,
  TRANSFORMERS_SUBFOLDER,
)

# Scaled output dtype. "float32" preserves the eosframes-scaled values (NaNs imputed away).
OUTPUT_DTYPE = "float32"


class TreatMonitor(LiveTableMonitor):
  """Live per-descriptor table for the Treat step's reference-transformer application.

  Columns: Descriptor | Status (queued → treating:<chunk k/n> → done/skipped) | Columns (in→out) |
  Time. Serial loop, so no thread contention.
  """

  item_label = "Descriptor"
  title = "Treating descriptors"
  running_verb = "treating"

  def _columns(self, table):
    table.add_column("Columns", width=22, no_wrap=True)
    table.add_column("Time", justify="right", width=8, no_wrap=True)

  def _row_cells(self, item_id, s):
    return [s["extra"].get("columns", "[dim]—[/]"), self._fmt_time(s)]


class TreatedDescriptors(DescriptorBase):
  """Apply the reference-library eosframes transformer to each featurizer's raw descriptors.

  This is *apply-only*: the transformer is pre-fitted offline on the reference library. At fit time
  it is downloaded and a copy is saved per model; at predict time the saved copy is reused. The
  full descriptor column set is preserved (eosframes maps constants to 0 and imputes NaNs), so the
  feature dimensionality is identical across fit and predict.
  """

  def __init__(self, path, chunk_size=DEFAULT_CHUNK_SIZE):
    DescriptorBase.__init__(self, path)
    self.logger = logger
    self.chunk_size = chunk_size
    self._name = TREATED_DESC_FILENAME
    self._params = self._load_params()
    self._reference_library = self._params.get("reference_library", DEFAULT_REFERENCE_LIBRARY)

  def _load_params(self):
    with open(params_path(self.path), "r") as f:
      return json.load(f)

  def _save_params(self):
    with open(params_path(self.path), "w") as f:
      json.dump(self._params, f, indent=4)

  def _featurizer_version(self, eos_id):
    versions = self._params.get("latest_featurizer_version") or {}
    if eos_id in versions:
      return versions[eos_id]
    # describe records this in parameters.json, but only on some paths (e.g. when a store is used).
    # Resolve it here the same way describe does (ersilia_model_version) and persist it, so the
    # version used for the transformer matches and predict can reuse it.
    version = ersilia_model_version(eos_id)
    self._params.setdefault("latest_featurizer_version", {})[eos_id] = version
    self._save_params()
    return version

  def _transformer_path(self, base, eos_id, version):
    return os.path.join(base, TRANSFORMERS_SUBFOLDER, local_transformer_name(eos_id, version))

  def _get_transformer(self, eos_id):
    """Download (fit) or load from the trained model dir (predict) the transformer for ``eos_id``.

    The local copy lives at ``<model_dir>/transformers/<eos>_<version>_transformer.json.gz``.
    """
    version = self._featurizer_version(eos_id)
    if self._is_predict:
      return load_local_transformer(self._transformer_path(self.trained_path, eos_id, version))
    transformer = fetch_reference_transformer(
      eos_id, version, reference_library=self._reference_library
    )
    save_local_transformer(transformer, self._transformer_path(self.path, eos_id, version))
    return transformer

  def _treat_in_memory(
    self, raw_h5, raw_features, expected_cols, transformer, output_path, substep=None
  ):
    if substep:
      substep("scaling")
    # Single pass over the raw H5, collecting values + inputs together. The separate
    # raw_h5.values() then raw_h5.inputs() calls each traverse the file independently (two full
    # reads for legacy Hdf5; two full chunk-file sweeps for ChunkedH5Store); iter_all yields both per
    # chunk in the same row order, so the concatenation is byte-for-byte what values()/inputs() return.
    value_parts = []
    inputs = []
    for _start, _end, vals, ins in raw_h5.iter_all(self.chunk_size):
      value_parts.append(np.asarray(vals, dtype="float32"))
      inputs.extend(ins)
    values = (
      np.concatenate(value_parts, axis=0)
      if value_parts
      else np.empty((0, len(raw_features)), dtype="float32")
    )
    df = pd.DataFrame(values, columns=raw_features)[expected_cols]
    scaled = eosframes_transform(df, transformer, output_dtype=OUTPUT_DTYPE, impute=True)
    out_values = scaled[expected_cols].to_numpy(dtype="float32")
    data = Data()
    data.set(inputs=inputs, values=out_values, features=expected_cols)
    data._is_sparse = False
    Hdf5(output_path).save(data)
    data.save_info(output_path.replace(".h5", ".json"))

  def _treat_chunked(
    self, raw_h5, n_rows, raw_features, expected_cols, transformer, output_path, substep=None
  ):
    store = ChunkedH5Store(output_path)
    store.create(len(expected_cols), expected_cols)
    n_chunks = max(1, -(-n_rows // self.chunk_size))  # ceil division
    for k, (start, end, values, inputs) in enumerate(raw_h5.iter_all(self.chunk_size), start=1):
      if substep:
        substep(f"chunk {k}/{n_chunks}")
      df = pd.DataFrame(np.array(values, dtype="float32"), columns=raw_features)[expected_cols]
      scaled = eosframes_transform(df, transformer, output_dtype=OUTPUT_DTYPE, impute=True)
      store.save_chunk(scaled[expected_cols].to_numpy(dtype="float32"), inputs)
      self.logger.debug(f"[treat] Processed rows {start}-{end}/{n_rows}")
      del df, scaled, values
      gc.collect()
    info = {
      "inputs": int(store.n_rows()),
      "features": int(store.n_features()),
      "values": [int(x) for x in store.shape()],
      "is_sparse": False,
    }
    with open(output_path.replace(".h5", ".json"), "w") as f:
      json.dump(info, f, indent=4)

  def _should_use_chunked(self, n_rows):
    return n_rows > self.chunk_size * 2

  def _prefetch_transformers(self, eos_ids):
    """Resolve + load every featurizer's transformer up front, downloads in parallel.

    The per-featurizer transformer fetch (a network download at fit; a local load at predict) is
    independent across featurizers and the slow part of treat, so do them concurrently instead of one
    at a time inside the apply loop. Versions are resolved SERIALLY first because that may persist
    ``parameters.json`` (concurrent writes would race); the parallel phase then only downloads/loads
    and writes distinct per-featurizer files. Returns ``{eos_id: transformer | Exception}`` — a
    per-featurizer failure is captured and re-raised when that featurizer is treated, preserving the
    original per-descriptor error behaviour.
    """
    for eos_id in eos_ids:
      self._featurizer_version(eos_id)  # serial: resolve + persist versions before fanning out

    def _one(eos_id):
      try:
        return eos_id, self._get_transformer(eos_id)
      except Exception as e:
        return eos_id, e

    out = {}
    with ThreadPoolExecutor(max_workers=io_workers(len(eos_ids))) as ex:
      for eos_id, value in ex.map(_one, eos_ids):
        out[eos_id] = value
    return out

  def _treat_one(self, eos_id, substep, transformer):
    """Apply the (prefetched) reference transformer to one featurizer. Returns ``(n_in, n_out)``, or
    None if the raw descriptors are missing (the descriptor is skipped)."""
    if isinstance(transformer, Exception):
      raise transformer
    run_eos_path = os.path.join(self.path, DESCRIPTORS_SUBFOLDER, eos_id)
    raw_h5_path = os.path.join(run_eos_path, RAW_DESC_FILENAME)
    output_h5_path = os.path.join(run_eos_path, TREATED_DESC_FILENAME)
    raw_h5 = open_h5(raw_h5_path)
    if raw_h5 is None:
      self.logger.warning(f"[treat] Skipping {eos_id}: no raw data found at {raw_h5_path}")
      return None
    raw_features = raw_h5.features()
    validate_transformer(transformer, raw_features, eos_id)
    # transform() requires the feature columns to match in set AND order; reorder to the
    # transformer's column order (set equality already verified above).
    expected_cols = list(transformer["columns"].keys())
    try:
      n_rows = raw_h5.n_rows()  # read once; drives both the chunked decision and the chunk loop
    except Exception as e:
      self.logger.warning(f"[treat] Could not determine row count for {eos_id}: {e}")
      n_rows = 0
    self.logger.info(
      f"[treat] Applying reference transformer to {eos_id} ({type(raw_h5).__name__})"
    )
    if self._should_use_chunked(n_rows):
      self.logger.info(f"[treat] Using chunked processing for {eos_id}")
      self._treat_chunked(
        raw_h5, n_rows, raw_features, expected_cols, transformer, output_h5_path, substep=substep
      )
    else:
      self.logger.info(f"[treat] Using in-memory processing for {eos_id}")
      self._treat_in_memory(
        raw_h5, raw_features, expected_cols, transformer, output_h5_path, substep=substep
      )
    gc.collect()
    return len(raw_features), len(expected_cols)

  def run(self):
    eos_ids = list(self.done_eos_iter())
    transformers = self._prefetch_transformers(eos_ids)
    monitor = TreatMonitor(eos_ids, color=STEP_COLORS.get("treat", "magenta"))
    with monitor.live():
      for eos_id in eos_ids:
        monitor.start(eos_id)
        try:
          widths = self._treat_one(
            eos_id, lambda t, e=eos_id: monitor.set_substep(e, t), transformers.get(eos_id)
          )
        except Exception:
          monitor.finish(eos_id, ok=False)
          raise
        if widths is None:
          monitor.finish(eos_id, ok=False)
          continue
        n_in, n_out = widths
        monitor.update_fields(eos_id, columns=f"{n_in:,} → {n_out:,}")
        monitor.finish(eos_id, ok=True)
