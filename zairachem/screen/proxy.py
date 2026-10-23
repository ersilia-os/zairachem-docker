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
from zairachem.base.utils.matrices import open_h5
from zairachem.base.vars import CV_SKILL, RANDOM_SEED, SKILL_FLOOR, TREATED_DESC_FILENAME

#: A candidate must clear this solo held-out skill to enter the greedy pool (else it can't be the
#: seed either — but the single best descriptor is always kept, so a run is never empty). The same
#: floor the pooler prunes on, from :data:`zairachem.base.vars.SKILL_FLOOR`.
PROXY_FLOOR = SKILL_FLOOR
#: Skill of a descriptor that carries no signal: chance for AUROC, no correlation for rho. The
#: same value the inner cross-validation reads at chance, from :data:`zairachem.base.vars.CV_SKILL`.
PROXY_CHANCE = {task: skill[-1] for task, skill in CV_SKILL.items()}
#: Minimum ensemble-skill gain for a descriptor to be added to the pool (strict improvement; guards
#: against adding redundant descriptors on noise).
PROXY_MARGIN = 1e-3
PROXY_MAX_DEPTH = 3
PROXY_N_ESTIMATORS = 100
#: Most rows the proxy is scored on. Its job is to rank descriptors against one another, and a
#: shallow forest's ranking settles long before a large set is exhausted. Scoring every row would
#: instead hold a descriptor's whole matrix -- 8 GB at a million rows and 2,048 features -- and
#: copy most of it again for each fold's fit. At or below this, every row is used.
PROXY_MAX_ROWS = 25_000


def _load_rows(descriptors_dir, eos_id, rows):
  """A descriptor's treated matrix restricted to ``rows``, or None if unavailable."""
  h5 = open_h5(os.path.join(descriptors_dir, eos_id, TREATED_DESC_FILENAME))
  if h5 is None:
    return None
  return np.asarray(h5.take_rows(rows), dtype=np.float32)


def _stratified_sample(y_rows, task, limit, seed):
  """Positions of at most ``limit`` rows, keeping the target's shape.

  Taking the first rows, or a flat random draw, can leave a fold without both classes or without
  a spread to correlate against, and a proxy scored on that ranks nothing. The strata are the
  pipeline's own -- the class for a binary target, a quantile bin for a continuous one.
  """
  from zairachem.holdout.splits import strata

  bins = strata(y_rows)
  rng = np.random.default_rng(seed)
  picked = []
  for b in np.unique(bins):
    where = np.flatnonzero(bins == b)
    take = min(where.size, max(1, int(round(limit * where.size / bins.size))))
    picked.append(rng.choice(where, take, replace=False))
  sel = np.concatenate(picked)
  if sel.size > limit:
    sel = rng.choice(sel, limit, replace=False)
  return np.sort(sel)


def _compact_folds(folds, y, task, limit, seed):
  """The rows the proxy will actually read, the folds renumbered into them, and their targets.

  Below ``limit`` this is every row the folds mention, renumbered to itself: the same rows, the
  same folds, the same scores. Above it the folds are thinned to a stratified sample, because the
  alternative is holding the whole matrix for every descriptor.
  """
  used = np.unique(
    np.concatenate([np.concatenate((np.asarray(tr), np.asarray(te))) for tr, te in folds])
  )
  if used.size > limit:
    used = used[_stratified_sample(np.asarray(y)[used], task, limit, seed)]
    logger.info(
      f"[screen] scoring the proxy on {used.size:,} of {len(y):,} rows: the ranking it produces "
      "settles well before that, and reading every row would hold the whole matrix per descriptor"
    )

  def renumber(side):
    side = np.asarray(side)
    return np.searchsorted(used, side[np.isin(side, used)])

  return used, [(renumber(tr), renumber(te)) for tr, te in folds], np.asarray(y)[used]


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
  tuple(list of str, dict, dict)
    ``(selected_ordered, scores, gains)`` — selected ids in add order, ``{eos_id:
    solo_held_out_skill}`` for every candidate, and ``{eos_id: ensemble gain when it was added}``
    for the selected ones (``None`` for the seed, which has nothing to add to). The gain is what
    actually decided a descriptor's fate, and a solo score cannot be read for it: a strong
    descriptor that merely repeats the seed is dropped, and a mediocre one that complements it is
    kept. Never empty: the best descriptor is kept when nothing clears ``floor``, and every
    candidate when none could be scored.
  """
  y = np.asarray(y)
  floor = PROXY_FLOOR[task] if floor is None else floor
  chance = PROXY_CHANCE[task]
  rows, folds, y = _compact_folds(folds, y, task, PROXY_MAX_ROWS, RANDOM_SEED)
  preds, scores = {}, {}
  for eos_id in eos_ids:
    try:
      X = _load_rows(descriptors_dir, eos_id, rows)
      if X is None:
        raise ValueError("no treated matrix")
      preds[eos_id] = [_fold_pred(X, y, tr, te, RANDOM_SEED, task) for tr, te in folds]
      # Solo skill: a one-member ensemble.
      scores[eos_id] = _ensemble_skill([eos_id], preds, y, folds, task)
    except Exception as e:  # one unreadable descriptor must not stop the screen
      logger.warning(f"[screen] {eos_id}: not scored ({e}); left out of the selection")
      preds[eos_id], scores[eos_id] = None, chance

  # A descriptor is unusable when it could not be read at all (None) and equally when every
  # fold declined to score it (a list of Nones) — a single-class or constant slice each time.
  # Only the first was being excluded, so the second could be picked as the greedy seed on its
  # chance score and keep descriptors that had real held-out predictions out of the run.
  usable = [e for e in eos_ids if preds[e] is not None and any(p is not None for p in preds[e])]
  if not usable:
    # Nothing could be scored, so there is no basis for choosing: keep every descriptor and let the
    # estimate step train (or skip) each one.
    logger.warning("[screen] no descriptor could be scored; keeping all of them")
    return list(eos_ids), scores, {}
  eligible = [e for e in usable if scores[e] >= floor]
  order = sorted(eligible or usable, key=lambda e: scores[e], reverse=True)

  selected = [order[0]]  # seed with the best solo descriptor
  ens_skill = scores[order[0]]
  gains = {order[0]: None}  # the seed is what the others are measured against
  trace = [f"{order[0]}=seed({ens_skill:.3f})"]
  for cand in order[1:]:
    if len(selected) >= k:
      break
    cand_skill = _ensemble_skill(selected + [cand], preds, y, folds, task)
    if cand_skill >= ens_skill + margin:
      selected.append(cand)
      gains[cand] = cand_skill - ens_skill
      trace.append(f"+{cand}(ens {ens_skill:.3f}→{cand_skill:.3f})")
      ens_skill = cand_skill
    else:
      trace.append(f"–{cand}(solo {scores[cand]:.3f}, no gain)")
  measure = "AUROC" if task == "classification" else "rho"
  logger.info(
    f"[screen] greedy kept {len(selected)}/{len(eos_ids)} descriptors "
    f"(k={k}, ensemble {measure} {ens_skill:.3f}): {' '.join(trace)}"
  )
  return selected, scores, gains
