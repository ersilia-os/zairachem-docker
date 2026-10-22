"""Cheap proxy scoring + ensemble-aware selection of descriptors for pre-screening.

Each descriptor is scored with a shallow Random Forest under the chemistry-aware evaluate folds
(random / scaffold / Butina): for every fold the RF is fit on the fold's train slice and predicts its
test slice. From those held-out predictions we get a **solo** skill per descriptor -- AUROC for
classification, Spearman's rho for regression -- *and* keep the prediction vectors so selection can
be **ensemble-aware**: descriptors are chosen by greedy forward selection, a candidate being kept
only if it *adds* predictive value to the pool (lifts the mean-across-folds ensemble skill), where
the ensemble is the arithmetic mean of the selected descriptors' held-out predictions (a cheap proxy
for the reliability pooler, mirroring lazyqsar's DescriptorPortfolio). Pure functions (matrices +
labels + folds in, scores/selection out).
"""

import os

import numpy as np

from zairachem.base.utils.logging import logger
from zairachem.base.utils.matrices import ChunkedH5Store, open_h5
from zairachem.base.vars import RANDOM_SEED, TREATED_DESC_FILENAME

#: A candidate must clear this solo held-out skill to enter the greedy pool (else it can't be the
#: seed either — but the single best descriptor is always kept, so a run is never empty). The
#: regression floor is the classification one on Spearman rho, as lazy-qsar scales it: AUROC - 0.5
#: is about rho / 2.
PROXY_FLOOR = {"classification": 0.55, "regression": 0.10}
#: Skill of a descriptor that carries no signal: chance for AUROC, no correlation for rho.
PROXY_CHANCE = {"classification": 0.5, "regression": 0.0}
#: Minimum ensemble-skill gain for a descriptor to be added to the pool (strict improvement; guards
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


def _fold_pred(X, y, train_idx, test_idx, seed, task):
  """Held-out predictions of a shallow random forest (fit on the fold's train slice).

  Probabilities for classification, values for regression. Returns ``None`` when the fold carries
  no usable signal: a single-class slice (classification) or a constant one (regression).
  """
  from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

  tr, te = np.asarray(train_idx), np.asarray(test_idx)
  if task == "classification":
    if len(set(y[tr].tolist())) < 2 or len(set(y[te].tolist())) < 2:
      return None
  elif np.ptp(y[tr]) == 0 or np.ptp(y[te]) == 0:
    return None
  forest = RandomForestClassifier if task == "classification" else RandomForestRegressor
  model = forest(
    n_estimators=PROXY_N_ESTIMATORS, max_depth=PROXY_MAX_DEPTH, random_state=seed, n_jobs=-1
  )
  model.fit(X[tr], y[tr])
  if task == "classification":
    return model.predict_proba(X[te])[:, 1].astype(np.float64)
  return model.predict(X[te]).astype(np.float64)


def _ensemble_skill(members, preds, y, folds, task):
  """Mean-across-folds skill of the mean-of-predictions ensemble over ``members``.

  AUROC for classification; Spearman's rho for regression, because what a regression model is
  asked for in a screen is the right ordering of compounds, not the right units.
  """
  from scipy.stats import spearmanr
  from sklearn.metrics import roc_auc_score

  chance = PROXY_CHANCE[task]
  scores = []
  for fi, (_, test_idx) in enumerate(folds):
    cols = [preds[m][fi] for m in members if preds[m][fi] is not None]
    yte = y[np.asarray(test_idx)]
    if not cols:
      continue
    pooled = np.mean(cols, axis=0)
    if task == "classification":
      if len(set(yte.tolist())) < 2:
        continue
      scores.append(roc_auc_score(yte, pooled))
    else:
      if np.ptp(yte) == 0 or np.ptp(pooled) == 0:
        continue
      rho = spearmanr(yte, pooled).statistic
      scores.append(chance if np.isnan(rho) else rho)
  return float(np.mean(scores)) if scores else chance


def select_descriptors(
  descriptors_dir, eos_ids, y, folds, k, task="classification", floor=None, margin=PROXY_MARGIN
):
  """Greedy ensemble-aware descriptor selection over the evaluate ``folds``.

  Seeds the pool with the best solo descriptor, then walks the rest in descending solo skill, adding
  a candidate only if it lifts the ensemble held-out skill by ≥ ``margin``. Redundant descriptors
  (high solo score but no marginal pool gain) are skipped, so fewer than ``k`` may be kept. The
  skill is AUROC for classification and Spearman's rho for regression.

  Parameters
  ----------
  descriptors_dir : str
    Directory holding ``<eos_id>/treated.h5`` per descriptor (a run's ``descriptors/``).
  eos_ids : list of str
    Candidate descriptor ids (typically the full ``done_eos.json``).
  y : array-like
    Labels (binary) or target values (continuous), row-aligned with the descriptor matrices.
  folds : list of tuple
    ``(train_idx, test_idx)`` pairs (the evaluate splits) the proxy is scored on.
  k : int
    Maximum number of descriptors to keep.
  task : str
    ``"classification"`` or ``"regression"``.
  floor : float, optional
    Minimum solo held-out skill to be eligible (default: the task's ``PROXY_FLOOR``).
  margin : float, optional
    Minimum ensemble-skill gain to add a descriptor (default 1e-3).

  Returns
  -------
  tuple(list of str, dict)
    ``(selected_ordered, scores)`` — selected ids in add order, and ``{eos_id: solo_held_out_skill}``
    for every candidate. Never empty: the best descriptor is kept when nothing clears ``floor``, and
    every candidate when none could be scored.
  """
  y = np.asarray(y)
  floor = PROXY_FLOOR[task] if floor is None else floor
  chance = PROXY_CHANCE[task]
  preds, scores = {}, {}
  for eos_id in eos_ids:
    try:
      X = _load_treated(descriptors_dir, eos_id)
      if X is None:
        raise ValueError("no treated matrix")
      preds[eos_id] = [_fold_pred(X, y, tr, te, RANDOM_SEED, task) for tr, te in folds]
      # Solo skill: a one-member ensemble.
      scores[eos_id] = _ensemble_skill([eos_id], preds, y, folds, task)
    except Exception as e:  # one unreadable descriptor must not stop the screen
      logger.warning(f"[screen] {eos_id}: not scored ({e}); left out of the selection")
      preds[eos_id], scores[eos_id] = None, chance

  usable = [e for e in eos_ids if preds[e] is not None]
  if not usable:
    # Nothing could be scored, so there is no basis for choosing: keep every descriptor and let the
    # estimate step train (or skip) each one.
    logger.warning("[screen] no descriptor could be scored; keeping all of them")
    return list(eos_ids), scores
  eligible = [e for e in usable if scores[e] >= floor]
  order = sorted(eligible or usable, key=lambda e: scores[e], reverse=True)

  selected = [order[0]]  # seed with the best solo descriptor
  ens_skill = scores[order[0]]
  trace = [f"{order[0]}=seed({ens_skill:.3f})"]
  for cand in order[1:]:
    if len(selected) >= k:
      break
    cand_skill = _ensemble_skill(selected + [cand], preds, y, folds, task)
    if cand_skill >= ens_skill + margin:
      selected.append(cand)
      trace.append(f"+{cand}(ens {ens_skill:.3f}→{cand_skill:.3f})")
      ens_skill = cand_skill
    else:
      trace.append(f"–{cand}(solo {scores[cand]:.3f}, no gain)")
  measure = "AUROC" if task == "classification" else "rho"
  logger.info(
    f"[screen] greedy kept {len(selected)}/{len(eos_ids)} descriptors "
    f"(k={k}, ensemble {measure} {ens_skill:.3f}): {' '.join(trace)}"
  )
  return selected, scores
