"""Build a per-fold workspace that shares the run's descriptors, inputs and metadata.

A fold is executed as a miniature model directory under ``folds/<fold_name>/``: its estimators and
pooler are written there (mirroring the production ``model/`` layout), while the heavy, already-computed
``descriptors/`` and ``inputs/`` are *symlinked* back to the parent run, and ``metadata/`` links the
parent's files but holds the fold's own descriptor selection. Nothing is recomputed or duplicated;
pointing an estimator/pooler at this directory reads the shared artifacts and writes only the fold's
own results.
"""

import json
import os

from zairachem.base.vars import (
  DATA_SUBFOLDER,
  DESCRIPTORS_SUBFOLDER,
  ESTIMATORS_SUBFOLDER,
  FOLDS_SUBFOLDER,
  METADATA_SUBFOLDER,
  POOL_SUBFOLDER,
  PROXY_SCORES_FILENAME,
  SELECTED_EOS_FILENAME,
)
from zairachem.estimate.estimators.lazy_qsar import ESTIMATORS_FAMILY_SUBFOLDER

# Shared, read-only dirs symlinked from each fold back to the parent run (all single-segment).
_SHARED = (DATA_SUBFOLDER, DESCRIPTORS_SUBFOLDER)


def build_fold_workspace(model_dir, fold_name, descriptors, proxy_scores=None):
  """Create ``<model_dir>/folds/<fold_name>/`` with shared dirs symlinked in, and return its path.

  ``metadata/`` is a folder of links to the parent's files except the screening's: its
  ``selected_eos.json`` holds the fold's own ``descriptors`` and its ``proxy_scores.json`` the
  fold's own ``proxy_scores`` (when the fold was screened). Every step that reads them (estimate,
  pool) then uses the fold's, which were chosen without its held-out molecules.

  Parameters
  ----------
  model_dir : str
    The run's top-level directory (holds the shared ``descriptors/``, ``inputs/``, ``metadata/``).
  fold_name : str
    Fold identifier (e.g. ``scaffold_det`` or ``random_00``).
  descriptors : list of str
    Descriptor ids the fold trains and pools.
  proxy_scores : dict, optional
    The fold's screening proxy AUROC per descriptor.

  Returns
  -------
  str
    Absolute path to the fold workspace.
  """
  fold_path = os.path.join(model_dir, FOLDS_SUBFOLDER, fold_name)
  os.makedirs(fold_path, exist_ok=True)
  for shared in _SHARED:
    link = os.path.join(fold_path, shared)
    if os.path.islink(link) or os.path.exists(link):
      continue
    target = os.path.join(model_dir, shared)
    os.symlink(os.path.relpath(target, fold_path), link)
  meta = os.path.join(fold_path, METADATA_SUBFOLDER)
  if os.path.islink(meta):  # a fold built before metadata/ became per-file
    os.remove(meta)
  os.makedirs(meta, exist_ok=True)
  parent_meta = os.path.join(model_dir, METADATA_SUBFOLDER)
  own = (SELECTED_EOS_FILENAME, PROXY_SCORES_FILENAME)
  for name in os.listdir(parent_meta):
    link = os.path.join(meta, name)
    if name not in own and not os.path.lexists(link):
      os.symlink(os.path.relpath(os.path.join(parent_meta, name), meta), link)
  for name, content in zip(own, (list(descriptors), proxy_scores)):
    path = os.path.join(meta, name)
    if os.path.lexists(path):
      os.remove(path)
    if content is not None:
      with open(path, "w") as f:
        json.dump(content, f, indent=4)
  # The fold's own outputs (mirror the production model/ layout).
  os.makedirs(
    os.path.join(fold_path, ESTIMATORS_SUBFOLDER, ESTIMATORS_FAMILY_SUBFOLDER), exist_ok=True
  )
  os.makedirs(os.path.join(fold_path, POOL_SUBFOLDER), exist_ok=True)
  return fold_path
