import contextlib
import csv
import json
import hashlib
import os
import threading
import time
import gc
import requests
import re
import pandas as pd
import numpy as np
from zairachem.base.utils.logging import logger
from zairachem.base.utils.console import console, echo
from zairachem.base import ZairaBase
from zairachem.base.utils.utils import (
  fetch_schema_from_github,
)
from zairachem.base.utils.model_version import ersilia_model_version
from zairachem.base.vars import METADATA_SUBFOLDER, PARAMETERS_FILE
from urllib.parse import urlparse
from rich.progress import (
  Progress,
  SpinnerColumn,
  TextColumn,
  BarColumn,
  TaskProgressColumn,
  MofNCompleteColumn,
  TimeElapsedColumn,
  TimeRemainingColumn,
)

try:
  from isaura.manage import IsauraCopy, IsauraReader, IsauraRemover, IsauraWriter
except Exception as e:
  logger.debug(f"Isaura modules could not be imported: {e}")
  IsauraCopy = None
  IsauraReader = None
  IsauraRemover = None
  IsauraWriter = None

ZAIRATEMP_BUCKET = "zairatemp"

_RING_TOKEN_RE = re.compile(r"%(?:\d{2})|[0-9]")
_ALLOWED_RE = re.compile(
  r"^[A-Za-z0-9@+\-\[\]\(\)=#$:.\\/%,*\.]+(?:\s+[A-Za-z0-9@+\-\[\]\(\)=#$:.\\/%,*\.]+)?$"
)
DEFAULT_BATCH_SIZE = 1000


class BinaryStreamClient(ZairaBase):
  # ALL isaura store access (reads, index lookups AND writes) is serialized across models. Describe
  # featurizes models concurrently, but isaura's DuckDB/S3 layer is not thread-safe — concurrent
  # IsauraReader/IsauraInspect/IsauraWriter from worker threads collide (a shared connection gets
  # "already closed"). Only the store I/O is serialized; the slow part — the model-server HTTP calls —
  # still runs fully in parallel. Shared by every client instance, so it serializes across the whole
  # describe thread pool; uncontended (and free) in the default serial path.
  _store_lock = threading.RLock()

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
      self.read_store = self.params.get("read_store")
      self.contribute_store = self.params.get("contribute_store")
      self._feature_len = None
      # Served-model output modality: try "heavy" (fast binary) first; if it fails the model latches
      # to "simple" (JSON) for the rest of the run — see _try_request.
      self._output_mode = "heavy"
      # Which provenance group this client's results count toward: "featurizers" (default) or
      # "projections" (set by the treat/manifolds step when computing projection models).
      self._provenance_kind = "featurizers"
      # Whether _run shows its own live progress bar. Disabled when several models are featurized
      # concurrently (rich allows only one live display at a time, so concurrent bars would clash).
      self._show_progress = True
      # Optional progress callback ``(model_id, *event)`` injected by a step's live-table monitor
      # (Describe/projections). When set, this client reports ("plan", n_cached, n_compute) and
      # ("batch", done, total) events up to the table instead of owning its own Progress bar / prints.
      self._progress_cb = None
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
      version = ersilia_model_version(model_id)
      self.params["latest_featurizer_version"][model_id] = version
      self._save_params(self.params)
      return version
    except Exception as e:
      logger.error(f"Error resolving version for model {model_id}: {e}")
      raise

  def _save_params(self, params):
    try:
      with open(os.path.join(self.path, METADATA_SUBFOLDER, PARAMETERS_FILE), "w") as f:
        json.dump(params, f, indent=2)
    except Exception as e:
      logger.error(f"Error saving params to disk: {e}")
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

  def _decode_json_records(self, response):
    """Parse a `simple` (JSON `records`) /run response into a (n, n_features) float array.

    Each record is a dict keyed by the model's output columns; values are pulled in `resolve_dims()`
    order so the matrix aligns with the rest of the pipeline. JSON `null` becomes NaN. The companion
    `results` meta is empty — `_run` overwrites dims/dtype/shape/data downstream regardless.
    """
    records = response.json()
    if not isinstance(records, list):
      raise IOError(
        f"Unexpected JSON /run response (expected a list, got {type(records).__name__})"
      )
    dims = self.resolve_dims()
    rows = [[rec.get(d) for d in dims] if isinstance(rec, dict) else list(rec) for rec in records]
    array = np.array(rows, dtype=float)  # None -> NaN
    return array, {}

  def _attempt(self, mode, batch):
    """POST one batch in a given output modality and return (array, results). Raises on any error.

    `mode` is "heavy" (binary stream) or "simple" (JSON records). On a non-OK HTTP status the raised
    error includes the server's response body so the real cause is visible (not just the status).
    """
    self._ensure_ready()
    headers = {"Content-Type": "application/json", "Accept": "*/*", "Connection": "close"}
    payload = json.dumps(batch, separators=(",", ":"))
    response = requests.post(
      self.url,
      params={"output_type": mode},
      data=payload,
      headers=headers,
      stream=(mode == "heavy"),
      timeout=(10, None),
    )
    if not response.ok:
      body = (response.text or "").strip().replace("\n", " ")[:500]
      raise requests.HTTPError(
        f"{response.status_code} {response.reason} from {self.url} (output_type={mode}): {body}",
        response=response,
      )
    if mode == "heavy":
      array, results = self.decode_binary_stream(response)
    else:
      array, results = self._decode_json_records(response)
    if self._feature_len is None:
      arr0 = np.asarray(array[0] if isinstance(array, (list, tuple)) else array)
      self._feature_len = arr0.shape[-1] if arr0.ndim > 1 else arr0.shape[0]
    return array, results

  def _try_request(self, batch):
    """Request one batch, picking the output modality at the model level.

    `heavy` (fast binary) is tried first; the first time it fails, the whole model latches to
    `simple` (JSON) and the batch is retried once. This is "heavy when it works, JSON when it
    doesn't" — a per-model choice, not a per-molecule one. Genuinely unprocessable molecules are
    handled by the caller's bisection (NaN placeholders), independent of the modality.
    """
    try:
      return self._attempt(self._output_mode, batch)
    except Exception as e:
      if self._output_mode == "heavy":
        # Benign transport fallback (results are identical) — log it for --verbose, don't print it
        # to the console on every run.
        self.logger.info(
          f"{self.model_id}: 'heavy' output failed ({e}); using JSON for this model."
        )
        self._output_mode = "simple"
        return self._attempt("simple", batch)
      logger.debug(f"Error in _try_request for batch of size {len(batch)}: {e}")
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
        # Per-split failures are expected while isolating a bad molecule; keep them at debug so the
        # console isn't flooded. The heavy→JSON modality switch happens in _try_request; a
        # fully-failed run is reported once by _run's all-failed guard.
        logger.debug(
          f"Request/IO error in _fetch_or_split at depth {depth}, idx_offset {idx_offset}: {e}"
        )
        if len(batch) == 1:
          logger.debug(f"Failed to process single-element batch at idx_offset {idx_offset}")
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

  def _contribute(self, any_results, batch_size=None):
    if not self.contribute_store:
      return
    is_temp = self.contribute_store == ZAIRATEMP_BUCKET
    write_bucket = ZAIRATEMP_BUCKET if is_temp else self.contribute_store
    try:
      logger.info(f"Writing precalculations to bucket: {write_bucket}")
      w = IsauraWriter(
        input_csv=None,
        model_id=self.model_id,
        model_version=self.version,
        bucket=write_bucket,
      )
      self._write_to_store(w, any_results, batch_size)
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
    except SystemExit as e:
      # isaura raises SystemExit (not Exception) when the project/bucket does not
      # exist; catch it explicitly so a missing bucket is a clear, non-fatal warning
      # instead of silently aborting the whole run.
      logger.warning(
        f"Could not contribute descriptors to isaura bucket '{write_bucket}': {e}. "
        f"Skipping upload and continuing. If the project does not exist, create it with: "
        f"isaura create -pn {write_bucket} --access <public|private>"
      )
    except Exception as e:
      logger.error(f"Error in Isaura contribute workflow: {e}")

  def _ersilia_chunk_df(self, inputs, values, dims):
    keys = [hashlib.md5(str(i).encode()).hexdigest() for i in inputs]
    return pd.DataFrame({"key": keys, "input": list(inputs)}).join(
      pd.DataFrame(values, columns=dims, dtype=str)
    )

  def _write_to_store(self, writer, res, batch_size=None):
    # Stream the upload chunk-by-chunk so the full descriptor matrix (n_molecules x
    # thousands of features) is never materialised in memory at once. Falls back to a
    # single write only when the values are already held in memory (non-streamed path).
    from zairachem.base.vars import DEFAULT_ISAURA_BATCH_SIZE

    bs = batch_size or DEFAULT_ISAURA_BATCH_SIZE
    dims = res["dims"]
    if res.get("data") is None and res.get("h5_file"):
      from zairachem.base.utils.matrices import open_h5

      h5 = open_h5(res["h5_file"])
      inputs = [str(x) for x in h5.inputs()]
      n = 0
      for start, end, chunk in h5.iter_values_with_indices(bs):
        writer.write(df=self._ersilia_chunk_df(inputs[start:end], np.asarray(chunk), dims))
        n += end - start
      logger.info(f"Contributed {n} rows to bucket (streamed in chunks of {bs})")
    else:
      writer.write(df=self._get_ersilia_df(res))

  def _report(self, *event):
    """Forward a progress event ``(model_id, *event)`` to the injected monitor callback, if any."""
    cb = getattr(self, "_progress_cb", None)
    if cb is not None:
      with contextlib.suppress(Exception):
        cb(self.model_id, *event)

  def _announce_plan(self, n_cached, n_compute):
    """Print an attractive one-line per-model sourcing plan (cached vs to-compute).

    When a progress callback is wired, the sourcing plan is reported to the live table instead of
    printed standalone, so the table is the single source of truth.
    """
    if getattr(self, "_progress_cb", None) is not None:
      self._report("plan", n_cached, n_compute)
      return
    ver = getattr(self, "version", None)
    parts = []
    if n_cached:
      parts.append(f"[green]{n_cached:,} cached[/]")
    if n_compute:
      parts.append(f"[yellow]{n_compute:,} to compute[/]")
    if not parts:
      parts.append("[dim]nothing to do[/]")
    from zairachem.base.utils.console import active_color

    c = active_color()
    ver_part = f" [dim]·[/] [{c}]{ver}[/]" if ver and ver != "?" else ""
    console.print(
      f"  [{c}]▪[/] [bold {c}]{self.model_id}[/]{ver_part}   " + " [dim]·[/] ".join(parts)
    )

  def run(self, output_h5=None, isaura_batch_size=None):
    from zairachem.base.vars import DEFAULT_ISAURA_BATCH_SIZE

    if isaura_batch_size is None:
      isaura_batch_size = DEFAULT_ISAURA_BATCH_SIZE
    n_total = len(self.input_data)
    any_results = None
    # Only the rows actually computed this run get written back. When everything is read from the
    # store there is nothing new to cache, so we skip the (otherwise all-duplicate) write entirely.
    to_contribute = None
    if self.contribute_store or self.read_store:
      self.version = self.resolve_version(self.model_id, self.read_store)
    if not self.read_store:
      logger.info("Isaura read store is disabled; computing all descriptors via Ersilia.")
      self._announce_plan(0, n_total)
      any_results = self._run(output_h5=output_h5)
      # Record provenance even without a store so the per-model provenance box always renders
      # (everything computed via Ersilia this run).
      self._record_provenance(n_total, 0, n_total)
      to_contribute = any_results  # all rows are freshly computed
    else:
      try:
        self._computed_results = None
        any_results = self._run_hybrid(output_h5=output_h5, isaura_batch_size=isaura_batch_size)
        # Hybrid contributes only its cache misses (None when everything came from the store).
        to_contribute = self._computed_results
      except SystemExit as e:
        # The project has no data for this model yet: compute everything via Ersilia.
        logger.info(
          f"No precalculations in '{self.read_store}' for {self.model_id}; computing all: {e}"
        )
        self._announce_plan(0, n_total)
        any_results = self._run(output_h5=output_h5)
        self._record_provenance(n_total, 0, n_total)
        to_contribute = any_results
      except Exception as e:
        logger.error(f"Unhandled error in run: {e}")
        raise
    if to_contribute is not None:
      # Serial across the describe pool — only one model touches the store at a time.
      with BinaryStreamClient._store_lock:
        self._contribute(to_contribute, isaura_batch_size)
    # When a monitor callback is wired, the step's live table reports completion (per row); only
    # print the standalone success line for the unmonitored/standalone case.
    if getattr(self, "_progress_cb", None) is None:
      noun = "projections" if self._provenance_kind == "projections" else "descriptors"
      echo(f"[bold green]{self.model_id}[/] ready [dim]·[/] {n_total:,} {noun}", kind="success")
    return any_results

  def _record_provenance(self, n_total, n_from_project, n_computed):
    try:
      from zairachem.base.utils.isaura_report import record_provenance

      record_provenance(
        self.path, self._provenance_kind, self.model_id, n_total, n_from_project, n_computed
      )
    except Exception:
      pass

  def _assemble_results(self, full, cols, dtype, output_h5, isaura_batch_size):
    n_total = full.shape[0]
    if output_h5:
      from zairachem.base.utils.matrices import ChunkedH5Store

      h5_store = ChunkedH5Store(output_h5)
      h5_store.create(len(cols), cols)
      for lo in range(0, n_total, isaura_batch_size):
        hi = min(lo + isaura_batch_size, n_total)
        h5_store.save_chunk(full[lo:hi, :], self.input_data[lo:hi])
      return {
        "shape": (n_total, len(cols)),
        "dtype": dtype,
        "data": None,
        "dims": cols,
        "inputs": self.input_data,
        "h5_file": output_h5,
      }
    return {
      "shape": full.shape,
      "dtype": dtype,
      "data": full,
      "dims": cols,
      "inputs": self.input_data,
    }

  def _stored_inputs(self):
    """Set of input molecules actually stored in the project for this model/version.

    Uses isaura's authoritative index (not the bloom filter, which can over-report). An empty/absent
    model prefix yields an empty set, so everything is treated as a cache miss.
    """
    from isaura.manage import IsauraInspect

    # Serialized with all other store access: isaura's DuckDB layer isn't thread-safe (see _store_lock).
    with BinaryStreamClient._store_lock:
      try:
        insp = IsauraInspect(model_id=self.model_id, model_version=self.version, cloud=False)
        idx = insp.load_index(self.read_store, self.model_id, self.version) or {}
        return set(idx.keys())
      except Exception:
        return set()
      finally:
        # Tear the connection down deterministically while still holding the lock, so its (possibly
        # shared) DuckDB connection can't be closed from this thread after another thread has reopened.
        insp = None
        gc.collect()

  def _read_subset(self, smiles, isaura_batch_size):
    """Read descriptors for a subset of (project-present) molecules, aligned to `smiles` order.

    Runs entirely under :attr:`_store_lock` — isaura's DuckDB/S3 layer is not thread-safe, so the
    parallel describe pool must not open concurrent readers. The reader is built once and the reads
    stay chunked (with the inter-chunk gc) to bound peak memory on large reads."""
    cols = None
    rows = []
    with BinaryStreamClient._store_lock:
      r = IsauraReader(
        model_id=self.model_id,
        model_version=self.version,
        input_csv=None,
        approximate=False,
        bucket=self.read_store,
      )
      try:
        for lo in range(0, len(smiles), isaura_batch_size):
          chunk = smiles[lo : lo + isaura_batch_size]
          rdf = r.read(df=pd.DataFrame({"input": chunk}))
          if "input" in rdf.columns:
            rdf = (
              rdf.drop_duplicates(subset="input").set_index("input").reindex(chunk).reset_index()
            )
          if cols is None:
            cols = rdf.columns.difference(["key", "input"]).tolist()
          rows.append(rdf[cols].values.astype("float32"))
          del rdf
          gc.collect()
      finally:
        # Release the reader (and its DuckDB connection) deterministically before another thread can
        # acquire the lock and open its own — avoids a cross-thread "connection already closed".
        r = None
        gc.collect()
    return (np.vstack(rows) if rows else None), cols

  def _run_hybrid(self, output_h5=None, isaura_batch_size=None):
    """Read the project's stored molecules, compute the rest via Ersilia, and merge in input order.

    Partitions inputs by the project's actual index so neither side is all-or-nothing: present
    molecules are read, absent ones are computed. Records per-model provenance. The caller's
    _contribute() writes the merged result back (deduplicated) so newly computed rows get cached.
    """
    from zairachem.base.vars import DEFAULT_ISAURA_BATCH_SIZE

    if isaura_batch_size is None:
      isaura_batch_size = DEFAULT_ISAURA_BATCH_SIZE
    n_total = len(self.input_data)
    self.schema = fetch_schema_from_github(self.model_id)
    dtype = self.resolve_dtype()
    logger.info(f"Reading precalculations from project: {self.read_store}")

    stored = self._stored_inputs()
    present_idx = [i for i, s in enumerate(self.input_data) if s in stored]
    present_set = set(present_idx)
    absent_idx = [i for i in range(n_total) if i not in present_set]
    n_from_project = len(present_idx)
    n_computed = len(absent_idx)
    self._announce_plan(n_from_project, n_computed)

    cols = None
    read_vals = None
    if present_idx:
      read_vals, cols = self._read_subset(
        [self.input_data[i] for i in present_idx], isaura_batch_size
      )
    comp_vals = None
    self._computed_results = None
    if absent_idx:
      saved = self.input_data
      try:
        self.input_data = [self.input_data[i] for i in absent_idx]
        comp = self._run(output_h5=None)
        comp_vals = np.asarray(comp["data"], dtype="float32")
        if cols is None:
          cols = comp.get("dims") or self.resolve_dims()
        # `comp` holds exactly the cache-miss rows (its "inputs" are the absent molecules), so it's
        # what we contribute back to the store — never the rows we just read from it.
        self._computed_results = comp
      finally:
        self.input_data = saved
    if cols is None:
      cols = self.resolve_dims()

    full = np.full((n_total, len(cols)), np.nan, dtype="float32")
    if read_vals is not None:
      for k, i in enumerate(present_idx):
        full[i, :] = read_vals[k]
    if comp_vals is not None:
      for k, i in enumerate(absent_idx):
        full[i, :] = comp_vals[k]

    self._record_provenance(n_total, n_from_project, n_computed)
    return self._assemble_results(full, cols, dtype, output_h5, isaura_batch_size)

  def _get_ersilia_df(self, res):
    try:
      keys = [hashlib.md5(inp.encode()).hexdigest() for inp in res["inputs"]]
      data = res["data"]
      # When descriptors are streamed to disk, res["data"] is None and the real
      # values live in the h5 file. Load them so we don't upload all-NaN columns.
      if data is None and res.get("h5_file"):
        from zairachem.base.utils.matrices import open_h5

        h5 = open_h5(res["h5_file"])
        data = h5.values()
      return pd.DataFrame({"key": keys, "input": res["inputs"]}).join(
        pd.DataFrame(data, columns=res["dims"], dtype=str)
      )
    except Exception as e:
      logger.error(f"Error creating Ersilia DataFrame: {e}")
      raise

  def _run(self, output_h5=None):
    try:
      from zairachem.base.utils.matrices import ChunkedH5Store

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
      h5_store = None
      cols = None
      n_ok = 0  # rows the server actually returned (vs NaN placeholders for failures)
      if output_h5:
        schema_cols = fetch_schema_from_github(self.model_id)[0]
        h5_store = ChunkedH5Store(output_h5)
        h5_store.create(len(schema_cols), schema_cols)
        cols = schema_cols
      # Live bar only when this model is featurized on its own; suppressed under concurrency
      # (rich permits a single live display at a time) and when a monitor callback is wired (the
      # step's shared live table reports batch progress instead, via ("batch", done, total)).
      show = getattr(self, "_show_progress", True) and getattr(self, "_progress_cb", None) is None
      n_batches = max(1, -(-len(checked_input) // self.batch_size))  # ceil division
      progress = (
        Progress(
          SpinnerColumn(),
          TextColumn("[progress.description]{task.description}"),
          BarColumn(),
          TaskProgressColumn(),
          MofNCompleteColumn(),
          TimeElapsedColumn(),
          TimeRemainingColumn(),
          transient=True,
        )
        if show
        else None
      )
      try:
        with progress if show else contextlib.nullcontext():
          task = (
            progress.add_task(
              f"  [yellow]computing[/] [bold]{self.model_id}[/]", total=len(checked_input)
            )
            if show
            else None
          )
          processed_count = 0
          for batch_i, start in enumerate(range(0, len(checked_input), self.batch_size), start=1):
            batch = checked_input[start : start + self.batch_size]
            batch_abs_offset = good_idx[start]
            t0 = time.perf_counter()
            arrays_out, _, results = self._fetch_or_split(batch, idx_offset=batch_abs_offset)
            total_time += time.perf_counter() - t0
            if results is not None:
              any_results = results
            if h5_store:
              batch_arrays = []
              batch_inputs = []
              for j, row in enumerate(arrays_out):
                if isinstance(row, np.ndarray):
                  batch_arrays.append(row)
                  n_ok += 1
                else:
                  batch_arrays.append(self._placeholder_row())
                batch_inputs.append(checked_input[start + j])
              if batch_arrays:
                stacked_batch = np.vstack(batch_arrays).astype("float32")
                h5_store.save_chunk(stacked_batch, batch_inputs)
                logger.debug(f"[api] Saved chunk with {len(batch_arrays)} rows")
                del stacked_batch, batch_arrays
                gc.collect()
            else:
              for j, row in enumerate(arrays_out):
                per_item_rows[good_idx[start + j]] = row
                if isinstance(row, np.ndarray):
                  n_ok += 1
            processed_count += len(arrays_out)
            if show:
              progress.advance(task, len(arrays_out))
            else:
              self._report("batch", batch_i, n_batches)
      finally:
        self.logger.info(f"Total elapsed: {total_time:.4f}s")
      # A live server that returns nothing for any valid molecule means the model is broken in both
      # modalities — abort loudly rather than train on an all-NaN descriptor matrix.
      if checked_input and n_ok == 0:
        raise RuntimeError(
          f"Model '{self.model_id}' returned no descriptors for any of {len(checked_input):,} "
          f"molecules (server failed in both 'heavy' and JSON modes). Check the model container "
          f"logs (docker logs) for the underlying error."
        )
      if h5_store:
        if any_results is None:
          any_results = {}
        if "dtype" not in any_results:
          any_results.update({"dtype": self.resolve_dtype()})
        if "dims" not in any_results:
          any_results.update({"dims": cols})
        any_results.update({
          "shape": h5_store.shape(),
          "data": None,
          "inputs": checked_input,
          "h5_file": output_h5,
        })
      else:
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
      return any_results
    except Exception as e:
      logger.error(f"Unhandled error in _run: {e}")
      raise
