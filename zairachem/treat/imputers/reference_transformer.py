"""Fetch and manage the eosframes transformers pre-fitted on the reference library.

The treat step no longer fits a scaler per run. Instead, for each featurizer (eos_id) it applies a
transformer that was fitted offline on a fixed reference library (e.g. ``ersilia_reference_library_v0``)
and published to a public S3 bucket as a flat object named ``<eos_id>_<version>_transformer.json``.

At fit time the transformer is downloaded and a compact gzipped copy is saved under the model's
``transformers/`` folder (``transformers/<eos_id>_<version>_transformer.json.gz``); at predict time
that saved copy is reused, so predict reproduces the exact same scaling without any network access.
"""

import gzip, json, os

import requests

from zairachem.base.utils.logging import logger
from zairachem.base.vars import (
  DEFAULT_REFERENCE_LIBRARY,
  REFERENCE_LIBRARY_S3_BASE_URL,
  TRANSFORMER_SUFFIX,
)


def transformer_object_name(eos_id, version):
  """Name of the REMOTE object in the reference-library bucket (plain ``.json``)."""
  return f"{eos_id}_{version}_transformer.json"


def local_transformer_name(eos_id, version):
  """Name of the LOCAL per-model copy under ``transformers/`` (gzipped: ``..._transformer.json.gz``)."""
  return f"{eos_id}_{version}{TRANSFORMER_SUFFIX}"


def transformer_url(eos_id, version, reference_library=None, base_url=None):
  base_url = (base_url or REFERENCE_LIBRARY_S3_BASE_URL).rstrip("/")
  reference_library = reference_library or DEFAULT_REFERENCE_LIBRARY
  return f"{base_url}/{reference_library}/{transformer_object_name(eos_id, version)}"


def transformer_exists(eos_id, version, reference_library=None, base_url=None):
  """Return True if the reference transformer object is reachable (HTTP 200 to a HEAD request).

  Used by the setup preflight to confirm — before any descriptors are computed — that every
  featurizer has a transformer in the reference library. Any network error counts as "not available".
  """
  url = transformer_url(eos_id, version, reference_library=reference_library, base_url=base_url)
  try:
    resp = requests.head(url, timeout=30, allow_redirects=True)
  except requests.RequestException:
    return False
  return resp.status_code == 200


def fetch_reference_transformer(eos_id, version, reference_library=None, base_url=None):
  """Download the transformer JSON for ``(reference_library, eos_id, version)``.

  Raises a clear error (hard fail) if the object cannot be retrieved — a transformer fitted on the
  reference library is required for every featurizer in the run.
  """
  reference_library = reference_library or DEFAULT_REFERENCE_LIBRARY
  url = transformer_url(eos_id, version, reference_library=reference_library, base_url=base_url)
  logger.info(
    f"[treat] Fetching reference transformer for {eos_id} ({version}) from "
    f"reference library '{reference_library}': {url}"
  )
  try:
    resp = requests.get(url, timeout=60)
  except requests.RequestException as e:
    raise RuntimeError(
      f"Failed to download reference transformer for featurizer '{eos_id}' version "
      f"'{version}' from {url}: {e}"
    ) from e
  if resp.status_code != 200:
    raise RuntimeError(
      f"No reference transformer found for featurizer '{eos_id}' version '{version}' in "
      f"reference library '{reference_library}' (HTTP {resp.status_code} at {url}). A "
      f"transformer fitted on this reference library is required for the treat step."
    )
  try:
    return resp.json()
  except ValueError as e:
    raise RuntimeError(f"Reference transformer at {url} is not valid JSON: {e}") from e


def save_local_transformer(transformer, path):
  """Save the transformer as compact gzipped JSON (``path`` ends in ``.json.gz``)."""
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with gzip.open(path, "wt", encoding="utf-8") as f:
    json.dump(transformer, f, separators=(",", ":"))


def load_local_transformer(path):
  if not os.path.exists(path):
    raise RuntimeError(
      f"Expected a saved reference transformer at {path} but none was found. The trained model "
      f"may predate the eosframes-based treat step and must be retrained."
    )
  with gzip.open(path, "rt", encoding="utf-8") as f:
    return json.load(f)


def validate_transformer(transformer, feature_names, eos_id):
  """Assert the transformer's fitted columns match the raw descriptor columns (set equality)."""
  columns = transformer.get("columns")
  if not columns:
    raise RuntimeError(f"Reference transformer for '{eos_id}' has no 'columns' entry.")
  expected = set(columns.keys())
  actual = set(feature_names)
  if expected != actual:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    raise RuntimeError(
      f"Reference transformer for '{eos_id}' does not match the raw descriptor columns. "
      f"Missing from descriptors: {missing[:10]}{' ...' if len(missing) > 10 else ''}; "
      f"unexpected in descriptors: {extra[:10]}{' ...' if len(extra) > 10 else ''}."
    )
