"""Cheap proxy scoring of descriptor matrices for pre-screening.

Each descriptor is scored by its mean **held-out** AUROC across the chemistry-aware evaluate folds
(random / scaffold / Butina): for every fold a shallow Random Forest is fit on the fold's train slice
and scored on its test slice, and the per-fold AUROCs are averaged. This is a fast, generalization-
focused stand-in for the full per-descriptor autoML fit — good enough to rank descriptors and keep the
promising ones before training. Pure functions (matrices + labels + folds in, scores/selection out).
"""

import os
import numpy as np

from zairachem.base.utils.logging import logger
from zairachem.base.utils.matrices import ChunkedH5Store, open_h5
from zairachem.base.vars import RANDOM_SEED, TREATED_DESC_FILENAME

#: Descriptors scoring below this mean held-out AUROC are dropped (unless none clear it — then the
#: single best is kept, so a run always has at least one descriptor).
PROXY_FLOOR = 0.55
PROXY_MAX_DEPTH = 3
PROXY_N_ESTIMATORS = 100


def _load_treated(descriptors_dir, eos_id):
  """Load a descriptor's full treated matrix, or None if unavailable. Mirrors the estimator's reader."""
  h5 = open_h5(os.path.join(descriptors_dir, eos_id, TREATED_DESC_FILENAME))
  if h5 is None:
    return None
  shape = h5.shape()
  if isinstance(h5, ChunkedH5Store):
    X = np.empty(shape, dtype=np.float32)
    for start, end, chunk in h5.iter_values_with_indices():
      X[start:end] = chunk
    return X
  return h5.values()


def _fold_auroc(clf_cls, X, y, train_idx, test_idx, seed):
  """AUROC of a fresh shallow RF fit on the fold's train slice, scored on its test slice (or None)."""
  from sklearn.metrics import roc_auc_score

  tr, te = np.asarray(train_idx), np.asarray(test_idx)
  ytr, yte = y[tr], y[te]
  if len(set(ytr.tolist())) < 2 or len(set(yte.tolist())) < 2:
    return None  # a single-class train or test slice yields no usable AUROC
  clf = clf_cls(
    n_estimators=PROXY_N_ESTIMATORS, max_depth=PROXY_MAX_DEPTH, random_state=seed, n_jobs=-1
  )
  clf.fit(X[tr], ytr)
  return float(roc_auc_score(yte, clf.predict_proba(X[te])[:, 1]))


def descriptor_score(X, y, folds, seed=RANDOM_SEED):
  """Mean held-out AUROC of a descriptor matrix across ``folds`` (list of ``(train_idx, test_idx)``)."""
  from sklearn.ensemble import RandomForestClassifier

  y = np.asarray(y)
  aucs = [
    a
    for train_idx, test_idx in folds
    if (a := _fold_auroc(RandomForestClassifier, X, y, train_idx, test_idx, seed)) is not None
  ]
  return float(np.mean(aucs)) if aucs else 0.5


def select_descriptors(descriptors_dir, eos_ids, y, folds, k, floor=PROXY_FLOOR):
  """Score each descriptor over ``folds`` and pick the promising ones to fully train.

  Parameters
  ----------
  descriptors_dir : str
    Directory holding ``<eos_id>/treated.h5`` per descriptor (a run's ``descriptors/``).
  eos_ids : list of str
    Candidate descriptor ids (typically the full ``done_eos.json``).
  y : array-like
    Binary labels, row-aligned with the descriptor matrices.
  folds : list of tuple
    ``(train_idx, test_idx)`` pairs (the evaluate splits) the proxy is scored on.
  k : int
    Keep at most this many descriptors (the top-scoring ones clearing ``floor``).
  floor : float, optional
    Minimum mean held-out AUROC to keep a descriptor (default 0.55).

  Returns
  -------
  tuple(list of str, dict)
    ``(selected_ordered, scores)`` — selected ids best-first, and ``{eos_id: mean_held_out_auroc}`` for
    all candidates. If none clear ``floor``, the single best descriptor is kept so a run is never empty.
  """
  scores = {}
  for eos_id in eos_ids:
    X = _load_treated(descriptors_dir, eos_id)
    if X is None:
      logger.warning(f"[screen] {eos_id}: no treated matrix; scoring 0.5")
      scores[eos_id] = 0.5
      continue
    scores[eos_id] = descriptor_score(X, y, folds)
  ranked = sorted(eos_ids, key=lambda e: scores[e], reverse=True)
  passing = [e for e in ranked if scores[e] >= floor]
  selected = passing[: max(1, int(k))] if passing else ranked[:1]
  kept = ", ".join(f"{e}={scores[e]:.3f}" for e in selected)
  logger.info(
    f"[screen] kept {len(selected)}/{len(eos_ids)} descriptors (k={k}, floor={floor:.2f}): {kept}"
  )
  return selected, scores
