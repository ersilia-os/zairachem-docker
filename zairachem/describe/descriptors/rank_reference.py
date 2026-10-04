"""lazy-qsar's rank reference: one descriptor matrix per featurizer, kept in this repo's ``data/``.

lazy-qsar's ``rank`` is a position against a fixed library of 50,000 molecules, so every descriptor a
fit trains needs that library featurized with it. :func:`prepare` runs at the start of the estimate
step, after pre-screening, so only the descriptors that are actually trained need one.

The matrices live in ``data/rank_reference/<REFERENCE_ID>_n<n>/<eos_id>_<version>.h5`` at the root of
this repository, which is an eosvc repo: ``install.sh`` fills it with ``eosvc download --path
data/rank_reference``, and maintainers publish it with ``eosvc upload``. A trained descriptor whose
matrix is not there is computed on its model server and saved there, so later fits reuse it.

Every matrix has a JSON sidecar naming the reference it was computed against (lazy-qsar's reference
id, size and manifest hash), the featurizer and version, and the file's sha256. A matrix whose
sidecar does not match this install's reference is never used.
"""

import hashlib
import json
import os
import sys

import h5py
import numpy as np

from zairachem.base.utils.logging import logger
from zairachem.base.utils.matrices import open_h5, remove_h5
from zairachem.base.vars import (
  DATA_SUBFOLDER,
  DESCRIPTORS_SUBFOLDER,
  RANK_REFERENCE_DIR,
  RANK_REFERENCE_RAW_FILENAME,
  RANK_REFERENCE_SMILES_FILENAME,
  RANK_REFERENCE_TREATED_FILENAME,
  TRANSFORMERS_SUBFOLDER,
)

_BYTES_PER_VALUE = 4  # float32, as describe writes descriptors
_HASH_CHUNK = 1 << 20


def reference_smiles():
  """lazy-qsar's reference molecules, in its row order, or None when they cannot be fetched.

  lazy-qsar downloads the list with the ``eosvc`` command, found on PATH. This interpreter's own
  ``bin`` goes first, so the ``eosvc`` installed alongside ZairaChem is found even when its
  environment was not activated. A failure is reported and returns None: the fit then goes ahead
  without ``rank``.
  """
  env_bin = os.path.dirname(sys.executable)
  if env_bin not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = os.pathsep.join([env_bin, os.environ.get("PATH", "")])
  try:
    from lazyqsar.reference import reference_smiles as _reference_smiles

    return _reference_smiles()
  except Exception as e:
    logger.warning(f"[rank_reference] lazy-qsar's reference list is unavailable: {e}")
    return None


def reference_identity():
  """What a matrix was computed against: lazy-qsar's reference id, size and manifest hash."""
  from lazyqsar.reference import REFERENCE_ID, default_n
  from lazyqsar.reference.manifest import manifest_sha256

  return {"reference_id": REFERENCE_ID, "n": default_n(), "manifest_sha256": manifest_sha256()}


def library_dir():
  """``data/rank_reference/<REFERENCE_ID>_n<n>``: where this install's reference matrices live."""
  from lazyqsar.reference import REFERENCE_ID, default_n

  return os.path.join(RANK_REFERENCE_DIR, f"{REFERENCE_ID}_n{default_n()}")


def matrix_path(eos_id, version):
  return os.path.join(library_dir(), f"{eos_id}_{version}.h5")


def _sidecar(h5_path):
  return h5_path[: -len(".h5")] + ".json"


def _sha256(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for block in iter(lambda: f.read(_HASH_CHUNK), b""):
      h.update(block)
  return h.hexdigest()


def stored(eos_id, version):
  """The matrix in ``data/`` for ``(eos_id, version)``, or None when absent or for another library."""
  path = matrix_path(eos_id, version)
  if not os.path.exists(path) or not os.path.exists(_sidecar(path)):
    return None
  try:
    with open(_sidecar(path)) as f:
      meta = json.load(f)
  except Exception:
    return None
  expected = {**reference_identity(), "eos_id": eos_id, "version": version}
  if any(meta.get(k) != v for k, v in expected.items()):
    logger.info(f"[rank_reference] Ignoring {path}: computed for another reference library")
    return None
  return path


def write_matrix(src_h5, eos_id, version):
  """Stream the matrix at ``src_h5`` into ``data/`` as one gzip-compressed H5, plus its sidecar.

  Written to a temporary name and moved into place, so a concurrent run never reads a half-written
  file and an interrupted write leaves nothing that looks valid. Returns the stored path.
  """
  src = open_h5(src_h5)
  dest = matrix_path(eos_id, version)
  os.makedirs(os.path.dirname(dest), exist_ok=True)
  tmp = f"{dest}.{os.getpid()}.tmp"
  n_rows, n_features = src.shape()
  str_dt = h5py.string_dtype(encoding="utf-8")
  with h5py.File(tmp, "w") as f:
    values = f.create_dataset(
      "Values",
      shape=(n_rows, n_features),
      dtype="float32",
      chunks=(min(n_rows, 4096), n_features),
      compression="gzip",
    )
    for start, end, chunk in src.iter_values_with_indices():
      values[start:end] = np.asarray(chunk, dtype="float32")
    f.create_dataset("Inputs", data=np.array(src.inputs(), dtype=str_dt))
    f.create_dataset("Features", data=np.array(src.features(), dtype=str_dt))
  meta = {
    **reference_identity(),
    "eos_id": eos_id,
    "version": version,
    "n_features": n_features,
    "sha256": _sha256(tmp),
  }
  os.replace(tmp, dest)
  with open(_sidecar(dest), "w") as f:
    json.dump(meta, f, indent=2)
  return dest


def link_into_run(stored_path, run_h5):
  """Point the run's raw rank-reference path at the stored matrix (a symlink, nothing copied)."""
  remove_h5(run_h5)
  os.symlink(stored_path, run_h5)


def estimate_bytes(eos_ids):
  """Uncompressed size of the reference matrices for ``eos_ids`` (width x library size x float32)."""
  from lazyqsar.reference import default_n

  from zairachem.base.utils.utils import fetch_schema_from_github

  total = 0
  for eos_id in eos_ids:
    schema = fetch_schema_from_github(eos_id)
    if schema:
      total += schema[2] * default_n() * _BYTES_PER_VALUE
  return total


def format_bytes(n):
  for unit in ("B", "KB", "MB", "GB"):
    if n < 1024 or unit == "GB":
      return f"{n:,.0f} {unit}" if unit in ("B", "KB") else f"{n:,.1f} {unit}"
    n /= 1024


def _write_reference_csv(path):
  """lazy-qsar's reference molecules as the run's ``inputs/rank_reference.csv``, or None."""
  csv = os.path.join(path, DATA_SUBFOLDER, RANK_REFERENCE_SMILES_FILENAME)
  if not os.path.exists(csv):
    smiles = reference_smiles()
    if smiles is None:
      return None
    with open(csv, "w") as f:
      f.write("smiles\n" + "\n".join(smiles) + "\n")
  return csv


def _resolve_raw(path, eos_id, version, csv, raw_path, batch_size):
  """Put ``eos_id``'s raw reference matrix at ``raw_path``, from ``data/`` or its model server.

  A computed matrix is saved into ``data/`` for later fits. The model server is only needed then;
  when it is no longer up it is started again.
  """
  from zairachem.describe.descriptors.api import BinaryStreamClient
  from zairachem.describe.descriptors.utils import Hdf5Data, get_model_url

  hit = stored(eos_id, version)
  if hit:
    logger.info(f"[rank_reference] {eos_id} read from {hit}")
    link_into_run(hit, raw_path)
    return "data"
  url = get_model_url(eos_id)
  if url is None:
    from zairachem.describe.descriptors.describe import Describer

    logger.info(f"[rank_reference] {eos_id} server is down; starting the model servers")
    Describer(path=path).setup_model_servers()
    url = get_model_url(eos_id)
  logger.info(f"[rank_reference] {eos_id} computing the 50,000 reference molecules")
  client = BinaryStreamClient(
    path=path, csv_path=csv, model_id=eos_id, url=url, project_name=os.path.basename(path)
  )
  client._provenance_kind = "rank_reference"
  res = client.run(output_h5=raw_path, isaura_batch_size=batch_size)
  if not res.get("h5_file"):
    if res.get("data") is None:
      raise RuntimeError(f"No rank-reference descriptors returned for model {eos_id}")
    Hdf5Data(res).save(raw_path)
  link_into_run(write_matrix(raw_path, eos_id, version), raw_path)
  return "computed, saved to data/"


def prepare(path, eos_ids, batch_size=None):
  """Put the treated rank reference at ``descriptors/<eos>/rank_reference.h5`` for ``eos_ids``.

  Fit only, for the descriptors the run trains (after pre-screening). Each one's raw reference is
  read from ``data/`` (or computed into it), scaled with the transformer the treat step saved for
  that featurizer, and the run's link to the raw matrix is dropped. A descriptor whose reference
  cannot be prepared is reported and skipped: its model then trains without ``rank``.
  """
  import time

  from zairachem.base import params_path
  from zairachem.base.utils.console import echo
  from zairachem.base.utils.matrices import DEFAULT_CHUNK_SIZE
  from zairachem.treat.imputers.reference_transformer import (
    load_local_transformer,
    local_transformer_name,
  )
  from zairachem.treat.imputers.treated import treat_rank_reference

  with open(params_path(path)) as f:
    versions = json.load(f).get("latest_featurizer_version") or {}
  pending = [
    e
    for e in eos_ids
    if not os.path.exists(
      os.path.join(path, DESCRIPTORS_SUBFOLDER, e, RANK_REFERENCE_TREATED_FILENAME)
    )
  ]
  if not pending:
    return
  csv = _write_reference_csv(path)
  if csv is None:
    return
  for eos_id in pending:
    eos_dir = os.path.join(path, DESCRIPTORS_SUBFOLDER, eos_id)
    raw_path = os.path.join(eos_dir, RANK_REFERENCE_RAW_FILENAME)
    t0 = time.perf_counter()
    try:
      version = versions[eos_id]
      source = _resolve_raw(path, eos_id, version, csv, raw_path, batch_size)
      transformer = load_local_transformer(
        os.path.join(path, TRANSFORMERS_SUBFOLDER, local_transformer_name(eos_id, version))
      )
      treat_rank_reference(
        raw_path,
        os.path.join(eos_dir, RANK_REFERENCE_TREATED_FILENAME),
        transformer,
        eos_id,
        chunk_size=batch_size or DEFAULT_CHUNK_SIZE,
      )
      echo(f"Rank reference {eos_id}: {source} · {time.perf_counter() - t0:.1f}s")
    except Exception as e:
      echo(f"Rank reference {eos_id} unavailable, trained without rank: {e}", kind="warning")
    finally:
      remove_h5(raw_path)
