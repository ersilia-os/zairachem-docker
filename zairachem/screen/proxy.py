"""Cheap proxy scoring + ensemble-aware selection of descriptors for pre-screening.

Each descriptor is scored with a shallow Random Forest under the chemistry-aware evaluate folds
(random / scaffold / Butina): for every fold the RF is fit on the fold's train slice and predicts its
test slice. From those held-out predictions we get a **solo** AUROC per descriptor *and* keep the
prediction vectors so selection can be **ensemble-aware**: descriptors are chosen by greedy forward
selection — a candidate is kept only if it *adds* predictive value to the pool (lifts the mean-across-
folds ensemble AUROC), where the ensemble is the arithmetic mean of the selected descriptors' held-out
probabilities (a cheap proxy for the reliability pooler, mirroring lazyqsar's DescriptorPortfolio).
Pure functions (matrices + labels + folds in, scores/selection out).
"""

import os
import numpy as np

from zairachem.base.utils.logging import logger
from zairachem.base.utils.matrices import ChunkedH5Store, open_h5
from zairachem.base.vars import RANDOM_SEED, TREATED_DESC_FILENAME

#: A candidate must clear this solo held-out AUROC to enter the greedy pool (else it can't be the seed
#: either — but the single best descriptor is always kept, so a run is never empty).
PROXY_FLOOR = 0.55
#: Minimum ensemble-AUROC gain for a descriptor to be added to the pool (strict improvement; guards
#: against adding redundant descriptors on noise).
PROXY_MARGIN = 1e-3
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


def _fold_pred(X, y, train_idx, test_idx, seed):
  """Held-out probabilities of a shallow RF (fit on the fold's train slice) over its test slice.

  Returns ``None`` when the fold's train or test slice is single-class (no usable signal).
  """
  from sklearn.ensemble import RandomForestClassifier

  tr, te = np.asarray(train_idx), np.asarray(test_idx)
  if len(set(y[tr].tolist())) < 2 or len(set(y[te].tolist())) < 2:
    return None
  clf = RandomForestClassifier(
    n_estimators=PROXY_N_ESTIMATORS, max_depth=PROXY_MAX_DEPTH, random_state=seed, n_jobs=-1
  )
  clf.fit(X[tr], y[tr])
  return clf.predict_proba(X[te])[:, 1].astype(np.float64)


def _ensemble_auc(members, preds, y, folds):
  """Mean-across-folds AUROC of the mean-of-probabilities ensemble over ``members`` (descriptor ids)."""
  from sklearn.metrics import roc_auc_score

  aucs = []
  for fi, (_, test_idx) in enumerate(folds):
    cols = [preds[m][fi] for m in members if preds[m][fi] is not None]
    yte = y[np.asarray(test_idx)]
    if not cols or len(set(yte.tolist())) < 2:
      continue
    aucs.append(roc_auc_score(yte, np.mean(cols, axis=0)))
  return float(np.mean(aucs)) if aucs else 0.5


def select_descriptors(
  descriptors_dir, eos_ids, y, folds, k, floor=PROXY_FLOOR, margin=PROXY_MARGIN
):
  """Greedy ensemble-aware descriptor selection over the evaluate ``folds``.

  Seeds the pool with the best solo descriptor, then walks the rest in descending solo AUROC, adding a
  candidate only if it lifts the ensemble held-out AUROC by ≥ ``margin``. Redundant descriptors (high
  solo score but no marginal pool gain) are skipped, so fewer than ``k`` may be kept.

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
    Maximum number of descriptors to keep.
  floor : float, optional
    Minimum solo held-out AUROC to be eligible (default 0.55).
  margin : float, optional
    Minimum ensemble-AUROC gain to add a descriptor (default 1e-3).

  Returns
  -------
  tuple(list of str, dict)
    ``(selected_ordered, scores)`` — selected ids in add order, and ``{eos_id: solo_held_out_auroc}``
    for every candidate. Never empty (the single best descriptor is kept when nothing clears ``floor``).
  """
  y = np.asarray(y)
  preds, scores = {}, {}
  for eos_id in eos_ids:
    X = _load_treated(descriptors_dir, eos_id)
    if X is None:
      logger.warning(f"[screen] {eos_id}: no treated matrix; scoring 0.5")
      preds[eos_id], scores[eos_id] = None, 0.5
      continue
    fold_preds = [_fold_pred(X, y, tr, te, RANDOM_SEED) for tr, te in folds]
    preds[eos_id] = fold_preds
    scores[eos_id] = _ensemble_auc([eos_id], preds, y, folds)  # solo AUROC = single-member ensemble

  usable = [e for e in eos_ids if preds[e] is not None]
  if not usable:
    return [eos_ids[0]] if eos_ids else [], scores
  eligible = [e for e in usable if scores[e] >= floor]
  order = sorted(eligible or usable, key=lambda e: scores[e], reverse=True)

  selected = [order[0]]  # seed with the best solo descriptor
  ens_auc = scores[order[0]]
  trace = [f"{order[0]}=seed({ens_auc:.3f})"]
  for cand in order[1:]:
    if len(selected) >= k:
      break
    cand_auc = _ensemble_auc(selected + [cand], preds, y, folds)
    if cand_auc >= ens_auc + margin:
      selected.append(cand)
      trace.append(f"+{cand}(ens {ens_auc:.3f}→{cand_auc:.3f})")
      ens_auc = cand_auc
    else:
      trace.append(f"–{cand}(solo {scores[cand]:.3f}, no gain)")
  logger.info(
    f"[screen] greedy kept {len(selected)}/{len(eos_ids)} descriptors "
    f"(k={k}, ensemble AUROC {ens_auc:.3f}): {' '.join(trace)}"
  )
  return selected, scores
