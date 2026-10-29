import contextlib
import csv
import gc
import hashlib
import json
import math
import os
import re
import threading
import time
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from rich.progress import (
  BarColumn,
  MofNCompleteColumn,
  Progress,
  SpinnerColumn,
  TaskProgressColumn,
  TextColumn,
  TimeElapsedColumn,
  TimeRemainingColumn,
)

from zairachem.base import ZairaBase, _atomic_json
from zairachem.base.utils.console import console, echo
from zairachem.base.utils.logging import logger
from zairachem.base.utils.model_version import ersilia_model_version
from zairachem.base.utils.utils import (
  fetch_schema_from_github,
)
from zairachem.base.vars import (
  DESCRIBE_BATCH_SIZE,
  DESCRIBE_MAX_FAILED_FRACTION,
  DESCRIBE_MIN_FAILED_ALLOWED,
  DESCRIBE_READ_TIMEOUT_PER_MOLECULE_S,
  DESCRIBE_READ_TIMEOUT_S,
  DESCRIBE_SERVER_WORKERS,
  METADATA_SUBFOLDER,
  MIN_DESCRIBE_BATCH_SIZE,
  PARAMETERS_FILE,
  SERVER_CHUNKING_MOLECULES,
  SERVER_CHUNKS_PER_WORKER,
)


def _isaura_classes():
  """``(IsauraReader, IsauraWriter)``, or ``(None, None)`` when isaura cannot be imported.

  isaura is an OPTIONAL dependency, only needed when a run uses a --store, and importing it costs
  ~250 ms (DuckDB, boto3), so it is imported on first use, not with this module. Its CURRENT API is
  used directly (no shim for older releases); a run that needs a store and cannot import it fails
  loudly at the point of use (see _contribute), never silently.
  """
  try:
    from isaura.manage import IsauraReader, IsauraWriter
  except Exception as e:
    logger.debug(f"Isaura could not be imported: {e}")
    return None, None
  return IsauraReader, IsauraWriter


_RING_TOKEN_RE = re.compile(r"%(?:\d{2})|[0-9]")
_ALLOWED_RE = re.compile(
  r"^[A-Za-z0-9@+\-\[\]\(\)=#$:.\\/%,*\.]+(?:\s+[A-Za-z0-9@+\-\[\]\(\)=#$:.\\/%,*\.]+)?$"
)


class DescriptorFailedError(RuntimeError):
  """A model server failed on more molecules than ``DESCRIBE_MAX_FAILED_FRACTION`` allows."""


def _is_balanced(s, open_ch, close_ch):
  """Whether ``open_ch``/``close_ch`` pairs in ``s`` are balanced and never close early.

  Parameters
  ----------
  s : str
      Text to scan.
  open_ch, close_ch : str
      Opening and closing characters.

  Returns
  -------
  bool
      True when every closing character matches an earlier opening one and none is left open.
  """
  c = 0
  for ch in s:
    if ch == open_ch:
      c += 1
    elif ch == close_ch:
      c -= 1
      if c < 0:
        return False
  return c == 0


class BinaryStreamClient(ZairaBase):
  """Featurization client for one Ersilia model server, with optional isaura store reads/writes.

  Reads the run's input SMILES from ``csv_path``, computes descriptors by posting batches to the
  model server at ``url`` and, when the run has a store, reads cached rows from and writes newly
  computed rows to the isaura project.
  """

  #: Response format used for every request, with no learning and no retry in the other one (a
  #: projector sends a single small request, where binary buys nothing); None learns it per image.
  fixed_output_type = None

  # ALL isaura store access (reads, index lookups AND writes) is serialized across models. Describe
  # featurizes models concurrently, but isaura's DuckDB/S3 layer is not thread-safe — concurrent
  # IsauraReader/IsauraInspect/IsauraWriter from worker threads collide (a shared connection gets
  # "already closed"). Only the store I/O is serialized; the slow part — the model-server HTTP calls —
  # still runs fully in parallel. Shared by every client instance, so it serializes across all the
  # models describe runs at once.
  _store_lock = threading.RLock()

  def __init__(self, path, csv_path, model_id=None, url=None, project_name=None, batch_size=None):
    try:
      super(ZairaBase, self).__init__()
      self.logger = logger
      # Molecules per request (DESCRIBE_BATCH_SIZE), halved on a failed request.
      self.batch_size = batch_size or DESCRIBE_BATCH_SIZE
      self._initial_batch_size = self.batch_size
      self._largest_ok = 0  # largest request that has succeeded in this run
      # Set by describe (raw.py): the model's memory profile and a sampler of its container; each
      # request's molecules, peak memory and time are measured into the profile (for the scheduler's
      # memory reservations and the time estimate of later runs).
      self.profile = None
      self.sampler = None
      # Copies of the model the server may run for one request (its max_workers); describe raises
      # it for requests the server cuts into chunks when the VM has room (see raw._server_workers).
      self.server_workers = DESCRIBE_SERVER_WORKERS
      self.csv_path = csv_path
      self.url = url
      self.model_id = model_id
      self.path = path
      self.input_data, self.input_header = self._load_data()
      self.params = self._load_params()
      self.read_store = self.params.get("read_store")
      self.contribute_store = self.params.get("contribute_store")
      self._feature_len = None
      # False for a model that must get all molecules in one request (a projector embeds the
      # molecules of a request together): a failed request then raises instead of being split.
      self.splittable = True
      # Response format without a profile: "heavy" (binary array, no text round trip). With a
      # profile, the format the image supports is learned once and kept there (see _try_request).
      self.output_type = "heavy"
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
      self.schema = None  # (col_names, col_dtypes, width); lazily fetched + cached via _get_schema
    except Exception as e:
      logger.error(f"Error during BinaryStreamClient initialization: {e}")
      raise

  def _get_schema(self):
    """This model's output schema, fetched at most once per client. ``fetch_schema_from_github`` is
    also process-cached, but the resolve/placeholder paths ask for it several times per run, so caching
    on the instance avoids the repeated lookups entirely. Not cached when the fetch fails (returns
    None), so a transient failure can still recover."""
    if self.schema is None:
      self.schema = fetch_schema_from_github(self.model_id)
    return self.schema

  def resolve_dtype(self):
    """Return ``"float"`` if any output column in the model schema is float, else ``"int"``."""
    schema = self._get_schema()
    return "float" if "float" in set(schema[1]) else "int"

  def resolve_dims(self):
    """Return the model's output column names from its schema."""
    return self._get_schema()[0]

  def resolve_version(self, model_id, bucket):
    """Return ``model_id``'s Ersilia version, cached under ``latest_featurizer_version``.

    The version is persisted to the run's ``parameters.json`` on first lookup. ``bucket`` is unused.
    """
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
      _atomic_json(os.path.join(self.path, METADATA_SUBFOLDER, PARAMETERS_FILE), params)
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
    """Decode a "heavy" (binary) model-server response into a numpy array.

    The stream is a JSON header line (``dtype``, ``shape``) followed by the raw array bytes.

    Returns
    -------
    tuple of (numpy.ndarray, dict)
      The decoded array and the parsed header.
    """
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
      return _is_balanced(s, open_ch, close_ch)
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
        self._feature_len = len(self._get_schema()[0])
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

  def _ensure_ready(self, attempts=240, sleep_s=0.5):
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

  def open_below_known_refusals(self):
    """Drop the opening request under a size this image has refused before, if it has.

    Some images cannot serve a request past a size of their own, and the refusal costs its full
    timeout: eos4ex3 refuses 8,192 molecules after about 28 seconds, every run, before the client
    halves and carries on. The profile remembers it per image, so only the first run that meets a
    ceiling pays for it.
    """
    if self.profile is None:
      return
    opening = self.profile.first_request(self.batch_size)
    if opening < self.batch_size:
      logger.info(
        f"{self.model_id}: this image refused {self.profile.failed_at} molecules before, so "
        f"starting at {opening} per request rather than {self.batch_size}."
      )
      self.batch_size = opening
      self._initial_batch_size = opening

  def set_request_size_for_workers(self, workers):
    """Size requests for the number of server workers this model was granted.

    The server cuts a request of SERVER_CHUNKING_MOLECULES or more into chunks and runs them on
    its workers. With several workers that is what makes a large request fast; with one it only
    serialises the chunks and costs their overhead for nothing. Measured on eos3l5f in the binary
    format it actually uses: 9,999 molecules on one worker ran at 635 mol/s and 10,000 at 601 --
    throughput given away for crossing a line the request gets nothing from -- while 10,000 on
    four workers ran at 1,101. The same shape in JSON, where the gap is wider (435 against 391),
    and workers below the line buy nothing at all (8,192 on four workers: 436).

    So the request size and the worker grant are one decision. A model granted one worker stays
    just under the line; one granted more sits on it.
    """
    if workers > 1 or self.batch_size < SERVER_CHUNKING_MOLECULES:
      return
    self.batch_size = SERVER_CHUNKING_MOLECULES - 1
    self._initial_batch_size = self.batch_size
    logger.debug(
      f"[api] {self.model_id}: one server worker, so requests stay at {self.batch_size} "
      "molecules rather than crossing the server's chunking line"
    )

  def _attempt(self, batch, output_type):
    """POST one batch in ``output_type`` ("heavy" or "simple") and return ``(array, results)``.

    Raises on any error. On a non-OK HTTP status the raised error includes the server's response
    body so the real cause is visible.
    """
    self._ensure_ready()
    headers = {"Content-Type": "application/json", "Accept": "*/*", "Connection": "close"}
    payload = json.dumps(batch, separators=(",", ":"))
    response = requests.post(
      self.url,
      params={
        "output_type": output_type,
        "min_workers": 1,
        "max_workers": self.server_workers,
        # No server-side result cache: some server versions cache in redis by default, which would
        # serve stale results after a model's image is updated (redis outlives the containers).
        "fetch_cache": "false",
        "save_cache": "false",
      },
      data=payload,
      headers=headers,
      stream=output_type == "heavy",
      timeout=(10, DESCRIBE_READ_TIMEOUT_S + DESCRIBE_READ_TIMEOUT_PER_MOLECULE_S * len(batch)),
    )
    if not response.ok:
      body = (response.text or "").strip().replace("\n", " ")
      if len(body) > 2000:  # keep the start and the end of a traceback, where the cause is
        body = f"{body[:300]} … {body[-1700:]}"
      raise requests.HTTPError(
        f"{response.status_code} {response.reason} from {self.url} "
        f"(output_type={output_type}): {body}",
        response=response,
      )
    if output_type == "heavy":
      array, results = self.decode_binary_stream(response)
    else:
      array, results = self._decode_json_records(response)
    values = np.asarray(array)  # no copy: a batch of descriptors is large
    if values.dtype.kind == "f" and values.size and np.isnan(values).all():
      # A 200 that holds nothing: an image that cannot read this format answers so (its code reads
      # the input as a CSV, finds no molecules and returns empty rows), and would otherwise pass.
      raise IOError(f"{self.model_id} returned no values in {output_type} output (all empty)")
    if self._feature_len is None:
      arr0 = np.asarray(array[0] if isinstance(array, (list, tuple)) else array)
      self._feature_len = arr0.shape[-1] if arr0.ndim > 1 else arr0.shape[0]
    return array, results

  def _try_request(self, batch):
    """Request one batch in the format this model's image supports; raises on any error.

    The format is learned once per image and kept in the profile. Until known, binary is tried
    first; if the server rejects it or its reply cannot be decoded, and the same request succeeds as
    JSON, the image cannot produce binary output and is recorded as JSON. A timeout or a dropped
    connection says nothing about the format, so it is raised without trying JSON, as is a JSON
    failure (then the problem is not the format, e.g. memory).
    """
    if self.fixed_output_type:
      return self._attempt(batch, self.fixed_output_type)
    if self.profile is None:  # nothing to learn into: the client's own format
      return self._attempt(batch, self.output_type)
    if self.profile.output_type:
      try:
        return self._attempt(batch, self.profile.output_type)
      except (requests.Timeout, requests.ConnectionError):
        raise
      except (requests.RequestException, IOError, ValueError, KeyError, TypeError) as error:
        # A format that worked before and fails now: the image was replaced since it was learned
        # (a deleted and re-fetched image keeps its profile), so learn it again.
        self.logger.info(
          f"{self.model_id}: {self.profile.output_type} output failed ({error}); trying again."
        )
        self.profile.output_type = None
    try:
      out = self._attempt(batch, "heavy")
    except (requests.Timeout, requests.ConnectionError):
      raise
    except (requests.RequestException, IOError, ValueError, KeyError, TypeError) as heavy_error:
      try:
        out = self._attempt(batch, "simple")
      except (requests.RequestException, IOError) as simple_error:
        # Binary always fails on some images, so its error alone hides the real cause.
        raise type(simple_error)(
          f"{simple_error} [binary output had failed first: {heavy_error}]"
        ) from None
      self.logger.info(f"{self.model_id}: binary output not supported by this image; using JSON.")
      self.profile.output_type = "simple"
      return out
    self.profile.output_type = "heavy"
    return out

  def _fetch_batch(self, batch, idx_offset):
    """Request one batch; on failure halve the batch size, or isolate the failing molecules.

    Returns ``(arrays_out, results)``, one row per molecule (``None`` where a molecule failed), or
    ``(None, None)`` after halving ``self.batch_size``: the caller re-sends from the same molecule.
    A failure usually means the server ran out of memory on a request that large, so the smaller
    size is kept for the rest of the run. Only at ``MIN_DESCRIBE_BATCH_SIZE`` is the batch split to
    find molecules the model cannot process at all; the failure was then caused by those molecules,
    not the request size, so the size goes back to the largest one that has worked in this run.
    """
    if self.sampler is not None:
      self.sampler.reset()
    t0 = time.perf_counter()
    if not self.splittable:
      # All molecules in one request, never split: a failure is the model's, and is raised.
      array, results = self._try_request(batch)
      self._record(len(batch), t0)
      return self._rows(array), results
    if len(batch) <= MIN_DESCRIBE_BATCH_SIZE:
      arrays_out, ok, results = self._fetch_or_split(batch, idx_offset)
      if not all(ok):
        self.batch_size = self._largest_ok or self._initial_batch_size
      else:
        self._record(len(batch), t0)
      return arrays_out, results
    try:
      array, results = self._try_request(batch)
    except (requests.RequestException, IOError) as e:
      if self.profile is not None:
        self.profile.record_failure(len(batch))
      self.batch_size = max(MIN_DESCRIBE_BATCH_SIZE, len(batch) // 2)
      logger.warning(
        f"{self.model_id}: a request of {len(batch)} molecules failed; continuing with "
        f"{self.batch_size} per request."
      )
      logger.debug(f"{self.model_id}: {e}")
      return None, None
    self._largest_ok = max(self._largest_ok, len(batch))
    self._record(len(batch), t0)
    return self._rows(array), results

  def _record(self, n, t0):
    """Add a successful request of ``n`` molecules, started at ``t0``, to the model's profile."""
    if self.profile is None:
      return
    peak, cores = self.sampler.take() if self.sampler is not None else (0.0, None)
    seconds, copies = time.perf_counter() - t0, 1
    if n >= SERVER_CHUNKING_MOLECULES:
      # The server ran it as chunks of n / SERVER_CHUNKS_PER_WORKER molecules, one process per
      # worker at a time. Memory is recorded at the chunk size and time at the request's own size,
      # both only with one worker (with several, the peak mixes processes and the time is shorter).
      copies = max(1, self.server_workers)
      if copies == 1:
        self.profile.record(math.ceil(n / SERVER_CHUNKS_PER_WORKER), peak, None)
        self.profile.record(n, 0.0, seconds)
    else:
      self.profile.record(n, peak, seconds)
    if cores:
      # What one copy of the model keeps busy.
      self.profile.cores = max(cores / copies, self.profile.cores or 0.0)

  def _rows(self, array):
    """Split a returned batch into one ``(1, n_features)`` array per molecule."""
    if isinstance(array, np.ndarray):
      return [array[i : i + 1] for i in range(array.shape[0])]
    return [np.asarray(row).reshape(1, -1) for row in array]

  def _fetch_or_split(self, batch, idx_offset, depth=0):
    try:
      try:
        array, results = self._try_request(batch)
        arrays_out = self._rows(array)
        return arrays_out, [True] * len(arrays_out), results
      except (requests.RequestException, IOError) as e:
        # Per-split failures are expected while isolating a bad molecule; keep them at debug so the
        # console isn't flooded; a fully-failed run is reported once by _run's all-failed guard.
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

  def _fail_contribute(self, bucket, err):
    """Abort the run with a clear message when a requested store write fails.

    A store was explicitly requested (``--store``), so a failed contribute is fatal: continuing
    would report the run as successful while nothing was cached. Raises a normal ``Exception`` (not
    ``SystemExit``) so it also propagates out of describe's scheduler threads.
    """
    echo(
      f"Failed to write descriptors for [bold]{self.model_id}[/] to isaura store "
      f"'[bold]{bucket}[/]': {err}",
      kind="error",
    )
    echo(
      "A store was requested with --store but nothing could be cached. Re-run after fixing the "
      "store, or run without --store. If the project bucket does not exist: "
      f"[bold]isaura create -pn {bucket} --access <public|private>[/].",
      kind="info",
    )
    raise RuntimeError(f"isaura contribute failed for {self.model_id} → {bucket}: {err}")

  def _contribute(self, any_results, batch_size=None):
    if not self.contribute_store:
      return
    bucket = self.contribute_store
    # A store was explicitly requested (--store), so a write failure is fatal — otherwise the run
    # would report success while nothing was cached. isaura raises SystemExit (not Exception) for a
    # missing bucket, so catch both and route every failure through _fail_contribute.
    IsauraWriter = _isaura_classes()[1]
    if IsauraWriter is None:
      self._fail_contribute(
        bucket,
        "isaura's writer is unavailable — is 'isaura' installed and 'isaura.manage.IsauraWriter' "
        "importable?",
      )
    try:
      logger.info(f"Writing precalculations to bucket: {bucket}")
      w = IsauraWriter(
        input_csv=None,
        model_id=self.model_id,
        model_version=self.version,
        bucket=bucket,
      )
      self._write_to_store(w, any_results, batch_size)
    except (Exception, SystemExit) as e:
      self._fail_contribute(bucket, e)

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
        # show_progress=False: this runs inside the Describe live table; isaura's own progress bar
        # would draw over our region (see quiet_isaura_reads).
        writer.write(
          df=self._ersilia_chunk_df(inputs[start:end], np.asarray(chunk), dims), show_progress=False
        )
        n += end - start
      logger.info(f"Contributed {n} rows to bucket (streamed in chunks of {bs})")
    else:
      writer.write(df=self._get_ersilia_df(res), show_progress=False)

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
    """Featurize the run's molecules with this model and return the descriptor result.

    Molecules present in the isaura read store are read from it and the rest are computed via the
    model server; only freshly computed rows are written to the contribute store. With ``output_h5``
    the matrix is streamed to that chunked H5 file instead of being kept in memory.

    Returns
    -------
    dict
      ``shape``, ``dtype``, ``dims`` and ``inputs``, plus ``data`` (the matrix, or None when it
      was written to ``h5_file``).
    """
    from zairachem.base.vars import DEFAULT_ISAURA_BATCH_SIZE

    # Wall-clock for this single model's run() (one client per Ersilia model, featurizer or
    # projection), recorded into provenance by _record_provenance for the report's per-model timing.
    self._run_t0 = time.perf_counter()
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

      t0 = getattr(self, "_run_t0", None)
      elapsed = (time.perf_counter() - t0) if t0 is not None else None
      record_provenance(
        self.path,
        self._provenance_kind,
        self.model_id,
        n_total,
        n_from_project,
        n_computed,
        elapsed_seconds=elapsed,
      )
    except Exception as e:  # provenance is a best-effort record
      logger.debug(f"[describe] {self.model_id}: provenance not recorded: {e}")

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
    """Subset of this run's input molecules already stored for this model/version.

    Membership comes from isaura's **Bloom filter** — its authoritative "have we stored this?"
    structure, which is always loaded and appended across writes. We deliberately do NOT use
    ``index.sqlite`` (``IsauraInspect.load_index``): for wide models isaura writes a *partial* index
    on each incremental append (only that write's keys — intentional, see isaura's build_location_index),
    so it under-reports the cache and would force needless recompute + re-append. The Bloom is checked
    against this run's inputs (already standardized, matching what isaura stored). An empty/absent model
    yields an empty set, so everything is treated as a cache miss.
    """
    from isaura.manage import IsauraChecker

    # Serialized with all other store access: isaura's DuckDB/S3 layer isn't thread-safe (see _store_lock).
    with BinaryStreamClient._store_lock:
      try:
        with IsauraChecker(self.read_store, self.model_id, self.version) as checker:
          seen = checker.seen_many(self.input_data)
        return {s for s in self.input_data if seen.get(s, [False, None])[0]}
      except Exception:
        return set()
      finally:
        gc.collect()

  def _read_subset(self, smiles, isaura_batch_size):
    """Read descriptors for a subset of (project-present) molecules, aligned to `smiles` order.

    Runs entirely under :attr:`_store_lock` — isaura's DuckDB/S3 layer is not thread-safe, so
    describe (several models at once) must not open concurrent readers. The reads stay chunked
    (with the inter-chunk gc) to bound peak memory on large reads. Each chunk gets its own reader:
    a reader keeps the first inputs it is given and would return that chunk again (isaura#29)."""
    cols = None
    rows = []
    IsauraReader = _isaura_classes()[0]
    with BinaryStreamClient._store_lock:
      for lo in range(0, len(smiles), isaura_batch_size):
        chunk = smiles[lo : lo + isaura_batch_size]
        r = IsauraReader(
          model_id=self.model_id,
          model_version=self.version,
          input_csv=None,
          approximate=False,
          bucket=self.read_store,
        )
        try:
          rdf = r.read(df=pd.DataFrame({"input": chunk}))
        finally:
          # Release the reader (and its DuckDB connection) deterministically before another thread
          # can acquire the lock and open its own — avoids a cross-thread "connection already closed".
          r = None
        if "input" in rdf.columns:
          rdf = rdf.drop_duplicates(subset="input").set_index("input").reindex(chunk).reset_index()
        if cols is None:
          cols = rdf.columns.difference(["key", "input"]).tolist()
        rows.append(rdf[cols].values.astype("float32"))
        del rdf
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
    self._get_schema()
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
    # Scatter the read + computed blocks into their original row positions via fancy indexing (the
    # index lists align row-for-row with the value blocks), instead of a per-row Python loop.
    if read_vals is not None:
      full[present_idx] = read_vals
    if comp_vals is not None:
      full[absent_idx] = comp_vals

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
      n = len(self.input_data)
      per_item_rows = [None] * n
      good_idx, checked_input = self._valid_inputs()
      h5_store, cols = self._open_h5_store(output_h5)
      any_results, n_ok = self._stream_batches(checked_input, good_idx, h5_store, per_item_rows)
      # A live server that returns nothing for any valid molecule means the model is broken in both
      # modalities — abort loudly rather than train on an all-NaN descriptor matrix.
      if checked_input and n_ok == 0:
        raise RuntimeError(
          f"Model '{self.model_id}' returned no descriptors for any of {len(checked_input):,} "
          "molecules. Check the model container logs (docker logs) for the underlying error."
        )
      if h5_store:
        return self._h5_results(any_results, h5_store, cols, checked_input, output_h5)
      return self._in_memory_results(any_results, per_item_rows)
    except Exception as e:
      logger.error(f"Unhandled error in _run: {e}")
      raise

  def _valid_inputs(self):
    """Split the inputs into the ones that look like SMILES.

    Returns
    -------
    tuple of (list of int, list of str)
        Positions in ``input_data`` of the valid inputs, and the valid inputs themselves.
    """
    good_idx = []
    checked_input = []
    for i, s in enumerate(self.input_data):
      if self._is_smiles(s):
        good_idx.append(i)
        checked_input.append(s)
    return good_idx, checked_input

  def _open_h5_store(self, output_h5):
    """Create the chunked H5 store that descriptors stream into, when ``output_h5`` is set.

    Parameters
    ----------
    output_h5 : str or None
        Path of the H5 file, or None to keep descriptors in memory.

    Returns
    -------
    tuple of (ChunkedH5Store or None, list of str or None)
        The store and its feature columns, or ``(None, None)`` without ``output_h5``.
    """
    from zairachem.base.utils.matrices import ChunkedH5Store

    if not output_h5:
      return None, None
    schema_cols = self._get_schema()[0]
    h5_store = ChunkedH5Store(output_h5)
    h5_store.create(len(schema_cols), schema_cols)
    return h5_store, schema_cols

  def _make_progress(self):
    """Return the rich progress bar for the describe loop."""
    return Progress(
      SpinnerColumn(),
      TextColumn("[progress.description]{task.description}"),
      BarColumn(),
      TaskProgressColumn(),
      MofNCompleteColumn(),
      TimeElapsedColumn(),
      TimeRemainingColumn(),
      transient=True,
    )

  def _stream_batches(self, checked_input, good_idx, h5_store, per_item_rows):
    """Describe ``checked_input`` batch by batch, into ``h5_store`` or ``per_item_rows``.

    Parameters
    ----------
    checked_input : list of str
        Valid inputs to describe.
    good_idx : list of int
        Position in ``input_data`` of each valid input.
    h5_store : ChunkedH5Store or None
        Store to stream rows into; None fills ``per_item_rows`` in place instead.
    per_item_rows : list
        One slot per input in ``input_data``, filled when there is no ``h5_store``.

    Returns
    -------
    tuple of (dict or None, int)
        The last non-None server result, and how many rows the server actually returned.
    """
    total_time = 0.0
    any_results = None
    n_ok = 0  # rows the server actually returned (vs NaN placeholders for failures)
    # Live bar only when this model is featurized on its own; suppressed under concurrency
    # (rich permits a single live display at a time) and when a monitor callback is wired (the
    # step's shared live table reports batch progress instead, via ("batch", done, total)).
    show = getattr(self, "_show_progress", True) and getattr(self, "_progress_cb", None) is None
    progress = self._make_progress() if show else None
    try:
      with progress if show else contextlib.nullcontext():
        task = (
          progress.add_task(
            f"  [yellow]computing[/] [bold]{self.model_id}[/]", total=len(checked_input)
          )
          if show
          else None
        )
        start, batch_i = 0, 0
        # A share of the molecules, but never fewer than a few, so that one molecule the model
        # cannot process does not fail a small input.
        max_failed = max(
          DESCRIBE_MIN_FAILED_ALLOWED, DESCRIBE_MAX_FAILED_FRACTION * len(checked_input)
        )
        if not self.splittable:
          self.batch_size = max(1, len(checked_input))
        while start < len(checked_input):
          batch = checked_input[start : start + self.batch_size]
          t0 = time.perf_counter()
          arrays_out, results = self._fetch_batch(batch, idx_offset=good_idx[start])
          total_time += time.perf_counter() - t0
          if arrays_out is None:
            continue  # the batch size was halved; re-send from the same molecule
          if results is not None:
            any_results = results
          if h5_store:
            n_ok += self._save_h5_batch(h5_store, arrays_out, checked_input, start)
          else:
            n_ok += self._fill_rows(per_item_rows, arrays_out, good_idx, start)
          start += len(batch)
          batch_i += 1
          if start - n_ok > max_failed:
            raise DescriptorFailedError(
              f"Model '{self.model_id}' failed on {start - n_ok:,} of the first {start:,} molecules "
              f"(limit {DESCRIBE_MAX_FAILED_FRACTION:.0%} of {len(checked_input):,}), even when "
              "sent in small requests."
            )
          if show:
            progress.advance(task, len(arrays_out))
          else:
            remaining = -(-(len(checked_input) - start) // self.batch_size)
            self._report("batch", batch_i, batch_i + remaining)
    finally:
      self.logger.info(f"Total elapsed: {total_time:.4f}s")
    return any_results, n_ok

  def _save_h5_batch(self, h5_store, arrays_out, checked_input, start):
    """Write one batch of rows to ``h5_store``, with NaN placeholders for failed molecules.

    Parameters
    ----------
    h5_store : ChunkedH5Store
        Store to append the chunk to.
    arrays_out : list
        One row per batch input: an array, or a non-array for a failure.
    checked_input : list of str
        Valid inputs; the batch starts at ``start``.
    start : int
        Offset of the batch in ``checked_input``.

    Returns
    -------
    int
        Number of rows the server actually returned.
    """
    n_ok = 0
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
    return n_ok

  def _fill_rows(self, per_item_rows, arrays_out, good_idx, start):
    """Place one batch of rows into ``per_item_rows`` at their input positions.

    Parameters
    ----------
    per_item_rows : list
        One slot per input in ``input_data``, updated in place.
    arrays_out : list
        One row per batch input: an array, or a non-array for a failure.
    good_idx : list of int
        Position in ``input_data`` of each valid input.
    start : int
        Offset of the batch among the valid inputs.

    Returns
    -------
    int
        Number of rows the server actually returned.
    """
    n_ok = 0
    for j, row in enumerate(arrays_out):
      per_item_rows[good_idx[start + j]] = row
      if isinstance(row, np.ndarray):
        n_ok += 1
    return n_ok

  def _h5_results(self, any_results, h5_store, cols, checked_input, output_h5):
    """Complete the result dict for descriptors streamed to ``output_h5``.

    Parameters
    ----------
    any_results : dict or None
        Last server result, if any.
    h5_store : ChunkedH5Store
        Store the descriptors were written to.
    cols : list of str
        Feature columns of the store.
    checked_input : list of str
        Valid inputs, in store row order.
    output_h5 : str
        Path of the H5 file.

    Returns
    -------
    dict
        Result with ``dtype``, ``dims``, ``shape``, ``data`` (None), ``inputs`` and ``h5_file``.
    """
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
    return any_results

  def _in_memory_results(self, any_results, per_item_rows):
    """Complete the result dict for descriptors held in memory.

    Parameters
    ----------
    any_results : dict or None
        Last server result, if any.
    per_item_rows : list
        One row per input; non-array entries become NaN placeholder rows.

    Returns
    -------
    dict
        Result with ``dtype``, ``dims``, ``shape``, ``data`` and ``inputs``.
    """
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
