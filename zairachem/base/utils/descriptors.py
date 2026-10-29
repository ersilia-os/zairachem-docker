"""Resolve which descriptor models a run should actually train / pool / report on.

``done_eos.json`` (under ``descriptors/``) records every featurizer that was successfully computed.
When ``--max-descriptors`` pre-screening runs, it writes a chosen subset to
``metadata/selected_eos.json``; that subset then overrides ``done_eos.json`` everywhere downstream.
Keeping the two separate means ``done_eos.json`` stays an honest record of what was featurized.
Each held-out fold has its own ``metadata/selected_eos.json`` (its other metadata files link to the
run's): the run's descriptors, or a selection made on the fold's training rows when it was screened.
"""

import json
import os

from zairachem.base.utils.matrices import remove_h5
from zairachem.base.vars import (
  DESCRIPTORS_SUBFOLDER,
  METADATA_SUBFOLDER,
  POOL_SUBFOLDER,
  POOLED_DESCRIPTORS_FILENAME,
  RANK_REFERENCE_RAW_FILENAME,
  RANK_REFERENCE_TREATED_FILENAME,
  SELECTED_EOS_FILENAME,
)


def effective_descriptors(path):
  """The descriptor ids a run should use: ``metadata/selected_eos.json`` if present, else ``done_eos``.

  Parameters
  ----------
  path : str
    A run directory (the production model dir, a predict dir, or a fold workspace).

  Returns
  -------
  list of str
    Ordered descriptor (featurizer) ids.
  """
  selected = os.path.join(path, METADATA_SUBFOLDER, SELECTED_EOS_FILENAME)
  if os.path.exists(selected):
    with open(selected) as f:
      return list(json.load(f))
  with open(os.path.join(path, DESCRIPTORS_SUBFOLDER, "done_eos.json")) as f:
    return list(json.load(f))


def pooled_descriptors(path):
  """The descriptor ids a trained model's pooled model uses (what a prediction computes).

  :func:`effective_descriptors` less those the pooled model left out at fit (lazy-qsar's pruning,
  or a descriptor whose estimator failed), recorded in ``pool/pooled_descriptors.json``.

  Parameters
  ----------
  path : str
    A trained model directory.

  Returns
  -------
  list of str
    Ordered descriptor (featurizer) ids.
  """
  ids = effective_descriptors(path)
  record = os.path.join(path, POOL_SUBFOLDER, POOLED_DESCRIPTORS_FILENAME)
  if not os.path.exists(record):
    return ids
  with open(record) as f:
    recorded = json.load(f)
  pruned = set(recorded.get("pruned", []))
  # What was pooled is exactly what has an estimator: a descriptor skipped at fit (its estimator
  # failed) is neither pooled nor pruned, and must not be computed again at predict.
  pooled = recorded.get("pooled")
  return [i for i in ids if i not in pruned and (pooled is None or i in pooled)]


def drop_rank_references(path):
  """Delete every descriptor's rank-reference matrices once nothing will fit against them again.

  A fitted lazy-qsar model keeps only the knots it scored the reference into, so the matrices are
  dead weight after the last fit -- the estimate step, or the held-out folds when ``--evaluate`` is
  set. Each is 50,000 rows per featurizer, so they are removed even when intermediate data is kept.

  Parameters
  ----------
  path : str
    A run directory. A no-op where there are none (predict runs, or already dropped).
  """
  base = os.path.join(path, DESCRIPTORS_SUBFOLDER)
  if not os.path.isdir(base):
    return
  for eos_id in os.listdir(base):
    eos_dir = os.path.join(base, eos_id)
    if os.path.isdir(eos_dir):
      for fname in (RANK_REFERENCE_RAW_FILENAME, RANK_REFERENCE_TREATED_FILENAME):
        remove_h5(os.path.join(eos_dir, fname))
