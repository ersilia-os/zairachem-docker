"""Build a per-fold workspace that shares the run's descriptors, inputs and metadata.

A fold is executed as a miniature model directory under ``folds/<fold_name>/``: its estimators and
pooler are written there (mirroring the production ``model/`` layout), while the heavy, already-computed
``descriptors/`` and the ``inputs/``/``metadata/`` are *symlinked* back to the parent run so nothing is
recomputed or duplicated. Pointing an estimator/pooler at this directory therefore reads the shared
artifacts and writes only the fold's own results.
"""

import os

from zairachem.base.vars import (
  DATA_SUBFOLDER,
  DESCRIPTORS_SUBFOLDER,
  ESTIMATORS_SUBFOLDER,
  FOLDS_SUBFOLDER,
  METADATA_SUBFOLDER,
  POOL_SUBFOLDER,
)
from zairachem.estimate.estimators.lazy_qsar import ESTIMATORS_FAMILY_SUBFOLDER

# Shared, read-only dirs symlinked from each fold back to the parent run (all single-segment).
_SHARED = (DATA_SUBFOLDER, METADATA_SUBFOLDER, DESCRIPTORS_SUBFOLDER)


def build_fold_workspace(model_dir, fold_name):
  """Create ``<model_dir>/folds/<fold_name>/`` with shared dirs symlinked in, and return its path.

  Parameters
  ----------
  model_dir : str
    The run's top-level directory (holds the shared ``descriptors/``, ``inputs/``, ``metadata/``).
  fold_name : str
    Fold identifier (e.g. ``scaffold_det`` or ``random_00``).

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
  # The fold's own outputs (mirror the production model/ layout).
  os.makedirs(
    os.path.join(fold_path, ESTIMATORS_SUBFOLDER, ESTIMATORS_FAMILY_SUBFOLDER), exist_ok=True
  )
  os.makedirs(os.path.join(fold_path, POOL_SUBFOLDER), exist_ok=True)
  return fold_path
