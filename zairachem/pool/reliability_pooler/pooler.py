"""Pooling of the per-descriptor models through lazy-qsar's own ensemble.

Every output comes from :func:`lazyqsar.ensemble.combine.combine`, so the pooled model is
lazy-qsar's by construction, for either task:

- **classification** (``LazyClassifierQSAR``): probability, rank against the reference library,
  the binary call and the pre-calibration score, combined in logit space;
- **regression** (``LazyRegressorQSAR``): the predicted value in the target's own units, a weighted
  mean, with conformal prediction intervals. No reference library, rank or cutoff exists.

What this module adds is what lazy-qsar's ensemble does around that call at fit: the skill term
(quality, which penalises overfitting), the pruning of weak descriptors, the pooled reference rank
with its anchors and the score map (classification), and the NNLS stacking weights and interval
model (regression).

One input differs on purpose: each descriptor's rank-to-error reliability curve is built on its
out-of-fold predictions (lazy-qsar builds it on in-sample ones), an honest estimate of the error at
no extra cost.
"""

import dataclasses
import os

import numpy as np

# Imported at module load, i.e. before the CLI re-asserts its loguru sinks, which importing
# lazy-qsar clears.
from lazyqsar.ensemble.combine import EnsembleSpec, combine
from lazyqsar.ensemble.intervals import ad_summary, fit_intervals
from lazyqsar.ensemble.reference import build_pooled_score_knots
from lazyqsar.poolers.regression.inner_pooler import stacking_weights
from lazyqsar.qsar import _quality_weight  # private, pinned with lazyqsar==3.8.0
from lazyqsar.qsar_regressor import _quality_rho  # private, pinned with lazyqsar==3.8.0
from lazyqsar.utils.ranking import (
  DECISION_RANK,
  prepare_knots,
  proba_from_reference_rank,
  rank_from_reference,
  reference_anchor_table,
  subsample_knots,
)

from zairachem.base.vars import DEFAULT_CUTOFF, SKILL_FLOOR

#: Saved weighting of a trained model (in its pool folder). One artefact serves both tasks.
ARTIFACT_NAME = "reliability_pool.joblib"
#: What that file was called before regression existed. Read, never written: a model trained by an
#: earlier ZairaChem carries it, and the contents are the same.
LEGACY_ARTIFACT_NAME = "reliability_pool_clf.joblib"
#: Version of the saved weighting; models saved before it carry none (see :func:`load_params`).
ARTIFACT_VERSION = 2


def artifact_path(trained_path, write=False):
  """Path of the saved weighting inside ``trained_path``.

  Parameters
  ----------
  trained_path : str
    The trained model's ``pool/`` folder.
  write : bool
    True when the caller is about to save: the current name is always the one written.

  Returns
  -------
  str
    :data:`ARTIFACT_NAME` under ``trained_path``, or the legacy ``_clf`` name when reading a model
    trained before regression existed and only that file is present.
  """
  path = os.path.join(trained_path, ARTIFACT_NAME)
  if write or os.path.exists(path):
    return path
  legacy = os.path.join(trained_path, LEGACY_ARTIFACT_NAME)
  return legacy if os.path.exists(legacy) else path


#: lazy-qsar's pruning: a descriptor is left out when its out-of-fold skill is below the floor
#: (:data:`zairachem.base.vars.SKILL_FLOOR`) or more than this gap below the best one. The
#: regression gap is the classification one mapped onto Spearman rho, since ``AUC - 0.5`` is about
#: ``rho / 2``.
PRUNE_GAP = {"classification": 0.10, "regression": 0.20}
CLASSIFICATION = "classification"
REGRESSION = "regression"


def active_mask(skills, task=CLASSIFICATION):
  """Which descriptors the pooled model keeps, by lazy-qsar's pruning rule.

  ``skills`` are out-of-fold AUROCs (classification) or Spearman rhos (regression). An unknown
  skill keeps its descriptor (the measurement failed, which says nothing about it); all are kept
  when none would survive.
  """
  floor, gap = SKILL_FLOOR[task], PRUNE_GAP[task]
  known = [a for a in skills if a is not None]
  best = max(known) if known else 0.0
  mask = [a is None or (a >= floor and a >= best - gap) for a in skills]
  return mask if any(mask) else [True] * len(skills)


def build_spec(
  names, oof_aucs, train_aucs, proxy_aucs, curves, ad_cutoffs, prior, task=CLASSIFICATION
):
  """lazy-qsar's :class:`EnsembleSpec` for the descriptors ``names`` (all of them active).

  Parameters
  ----------
  names : list of str
      Descriptor (prediction column) names, in column order.
  oof_aucs, train_aucs : list of float or None
      Out-of-fold and in-sample skill per descriptor — AUROC for classification, Spearman rho for
      regression. The skill term uses their quality, ``2 * oof - train`` clipped, which penalises a
      descriptor that fits its training rows far better than its held-out ones.
  proxy_aucs : list of float or None, or None
      Screening proxy skill per descriptor, averaged into the skill term when known.
  curves : list of (ndarray, ndarray) or None
      Rank-to-error reliability curve per descriptor.
  ad_cutoffs : list of float or None, or None
      Applicability-domain veto floor per descriptor; None when there is no AD signal.
  prior : float
      Fraction of actives in the training rows (classification only).
  task : str
      ``"classification"`` or ``"regression"``.
  """
  quality_of = _quality_weight if task == CLASSIFICATION else _quality_rho
  quality = tuple(quality_of(o, t) for o, t in zip(oof_aucs, train_aucs))
  proxies = tuple(proxy_aucs) if proxy_aucs and any(p is not None for p in proxy_aucs) else None
  cuts = None
  if ad_cutoffs is not None:
    cuts = tuple(float(c) if c is not None else float("-inf") for c in ad_cutoffs)
  return EnsembleSpec(
    task=task,
    descriptor_names=tuple(names),
    oof_aucs=quality,
    proxy_aucs=proxies,
    rank_error_curves=tuple(curves) if curves is not None else None,
    ad_hard_cutoffs=cuts,
    population_prior=float(prior),
  )


def _outputs(task):
  """The outputs to ask ``combine`` for when fitting."""
  return ("value",) if task == REGRESSION else ("proba",)


def _ranks(R):
  # A descriptor whose percentile could not be computed for a row counts as neither reliable nor
  # unreliable there (lazy-qsar's own channels are never NaN).
  return None if R is None else np.where(np.isfinite(R), R, 0.5)


def pool(P, R, S, A, spec, outputs=("proba",)):
  """``CombineResult`` of lazy-qsar's ``combine`` for per-descriptor matrices aligned to ``spec``."""
  if spec.task == REGRESSION:
    return combine(P, _ranks(R), None, A, spec=spec, outputs=outputs)
  return combine(P, _ranks(R), S, A, spec=spec, outputs=outputs)


def _pooled(result, spec):
  """The pooled prediction: the positive-class probability, or the predicted value."""
  if spec.task == REGRESSION:
    return result.values["value"]
  return result.values["proba"][:, 1]


def _proba(result):
  return result.values["proba"][:, 1]


def _anchors(p, y):
  """lazy-qsar's rank anchors: p5 of the inactives (recorded) and p95 of the actives, which pins
  the top of the rank scale when it lies above the reference's 99.9th percentile."""
  p, y = np.asarray(p), np.asarray(y)
  act, inact = p[y == 1], p[y == 0]
  return (
    float(np.percentile(inact, 5)) if inact.size else None,
    float(np.percentile(act, 95)) if act.size else None,
  )


def rank_of(proba, knots, anchors):
  """A probability's rank against reference ``knots`` (lazy-qsar's ``rank``), or None.

  The same map ``combine`` applies to its pooled probability, so a cutoff and the scores it is
  compared with sit on one scale.
  """
  if proba is None or knots is None or not len(knots):
    return None
  value = rank_from_reference(
    np.array([float(proba)]), prepared=prepare_knots(np.asarray(knots)), anchors=anchors
  )
  return float(value[0])


def _descriptor_ranks(cutoffs, P, y, reference_P):
  """Each descriptor's ``cutoffs`` (probability, or None) on its own reference rank scale.

  As lazy-qsar ranks a single descriptor's probability: against that descriptor's scores of the
  rank-reference library, anchored on its out-of-fold training scores.
  """
  if reference_P is None:
    return [None] * len(cutoffs)
  ranks = []
  for j, cut in enumerate(cutoffs):
    ref = reference_P[:, j][np.isfinite(reference_P[:, j])]
    knots = subsample_knots(np.sort(ref)) if ref.size else None
    ranks.append(rank_of(cut, knots, _anchors(P[:, j], y)))
  return ranks


def _fit_regression(spec, P, R, S, A, y, pruned_columns):
  """Finish a regression fit from the pruned matrices; return ``(params, result)``.

  Two things happen here that have no classification counterpart. The descriptor weights are
  lazy-qsar's 3.8 default -- one fixed weight each, fitted by non-negative least squares on the
  out-of-fold predictions -- which replaces the skill and reliability terms while leaving the
  per-compound applicability-domain veto in force. And the pooled out-of-fold residuals calibrate
  split-conformal prediction intervals, binned by applicability domain so a compound unlike the
  training set is handed a wider one.

  There is no reference library, so no rank, no screening cutoff and no cutoffs of any kind: the
  ``cutoffs`` and ``cutoff_ranks`` a classification model carries are empty by construction.
  """
  weights = stacking_weights(np.asarray(P, dtype=np.float64), np.asarray(y, dtype=np.float64))
  spec = dataclasses.replace(spec, descriptor_weights=tuple(float(w) for w in weights))
  p_train = _pooled(pool(P, R, S, A, spec, _outputs(REGRESSION)), spec)
  intervals = fit_intervals(np.asarray(y, dtype=np.float64), p_train, ad_summary(A))
  if intervals:
    spec = dataclasses.replace(spec, intervals=intervals)
  params = {
    "version": ARTIFACT_VERSION,
    "task": REGRESSION,
    "descriptor_columns": list(spec.descriptor_names),
    "pruned_columns": pruned_columns,
    "spec": spec,
    "has_rank": R is not None,
    "has_ad": A is not None,
    "has_score": False,
    "cutoffs": {},
    "cutoff_ranks": {},
  }
  return params, predict(params, P, R, S, A)


def fit(P, R, S, A, y, names, meta, reference=None, kappa=None, task=CLASSIFICATION):
  """Fit the pooled model on the training rows; return ``(params, result)``.

  ``meta`` holds the per-descriptor ``oof_aucs``, ``train_aucs``, ``proxy_aucs``, ``curves`` and
  ``ad_cutoffs`` (aligned to ``names``). Weak descriptors are pruned first and the matrices are
  pooled over the active ones only; from there the two tasks part company.

  Classification also pools the rank-reference library's ``(P, R, A)`` given in ``reference``,
  which is what gives the pooled probability its rank and the screening cutoff, and ``kappa``
  (``(y, p) -> cutoff``) sets the kappa cutoff on the pooled training scores. ``result`` then holds
  the training rows' ``proba`` and ``binary``, plus ``rank``/``score`` where available.

  Regression has no reference library and no cutoff of any kind (see :func:`_fit_regression`), so
  ``reference`` and ``kappa`` are ignored and ``result`` holds ``value`` with its interval.
  """
  mask = active_mask(meta["oof_aucs"], task)
  keep = [j for j, m in enumerate(mask) if m]

  def cols(M):
    return None if M is None else np.asarray(M, dtype=np.float64)[:, keep]

  def per(key):
    values = meta.get(key)
    return None if values is None else [values[j] for j in keep]

  P_a, R_a, S_a, A_a = cols(P), cols(R), cols(S), cols(A)
  spec = build_spec(
    [names[j] for j in keep],
    per("oof_aucs"),
    per("train_aucs"),
    per("proxy_aucs"),
    per("curves") if R_a is not None else None,
    per("ad_cutoffs") if A_a is not None else None,
    float(np.mean(y)) if len(y) and task == CLASSIFICATION else 0.5,
    task=task,
  )
  pruned_columns = [n for n, m in zip(names, mask) if not m]
  if task == REGRESSION:
    return _fit_regression(spec, P_a, R_a, S_a, A_a, y, pruned_columns)
  p_train = _pooled(pool(P_a, R_a, S_a, A_a, spec, _outputs(task)), spec)

  updates, info = {}, {}
  if S_a is not None:
    updates["pooled_score_knots"] = build_pooled_score_knots(P_a, _ranks(R_a), S_a, A_a, spec)
  screening = None
  if reference is not None and not (A_a is not None and reference[2] is None):
    # (a reference without AD scores would be pooled with different weights from the queries)
    P_ref, R_ref, A_ref = reference
    p_ref = _proba(pool(P_ref[:, keep], cols(R_ref), None, cols(A_ref), spec))
    p_ref = p_ref[np.isfinite(p_ref)]
    if p_ref.size:
      knots = subsample_knots(np.sort(p_ref))
      prepared = prepare_knots(knots)
      anchors = _anchors(p_train, y)
      screening = float(
        proba_from_reference_rank(DECISION_RANK, prepared=prepared, anchor_high=anchors[1])
      )
      updates.update(
        pooled_rank_knots=knots, pooled_rank_anchors=anchors, decision_cutoff=screening
      )
      _, _, high_used = reference_anchor_table(prepared=prepared, anchor_high=anchors[1])
      info = {"n_reference": int(p_ref.size), "anchor_high_used": bool(high_used)}
  spec = dataclasses.replace(spec, **updates)
  cutoffs = {
    "screening": screening,
    "kappa": kappa(np.asarray(y), p_train) if kappa is not None else None,
    "default": DEFAULT_CUTOFF,
  }
  balanced = meta.get("balanced") or [None] * len(names)
  ranks = {
    name: rank_of(value, spec.pooled_rank_knots, spec.pooled_rank_anchors)
    for name, value in cutoffs.items()
  }
  reference_P = None if reference is None else np.asarray(reference[0], dtype=np.float64)
  P_all = np.asarray(P, dtype=np.float64)
  ranks["balanced"] = dict(zip(names, _descriptor_ranks(balanced, P_all, y, reference_P)))
  ranks["default_by_descriptor"] = dict(
    zip(names, _descriptor_ranks([DEFAULT_CUTOFF] * len(names), P_all, y, reference_P))
  )
  params = {
    "version": ARTIFACT_VERSION,
    "descriptor_columns": list(spec.descriptor_names),
    "pruned_columns": pruned_columns,
    "spec": spec,
    "has_rank": R_a is not None,
    "has_ad": A_a is not None,
    "has_score": S_a is not None,
    "cutoffs": cutoffs,
    # Every cutoff as a rank, the scale of the score: pooled ones against the pooled reference,
    # each descriptor's balanced cutoff against its own (see _descriptor_ranks).
    "cutoff_ranks": ranks,
    **info,
  }
  return params, predict(params, P_a, R_a, S_a, A_a)


def predict(params, P, R, S, A):
  """Pooled outputs (``CombineResult``) for matrices aligned to ``params["descriptor_columns"]``.

  Regression gives the predicted value, plus the conformal interval when the model carries one.
  For classification, ``binary`` is the screening call when the model has a rank reference, else
  ``p >= 0.5``.
  """
  spec = params["spec"]
  if spec.task == REGRESSION:
    outputs = ["value"] + (["lower", "upper"] if spec.intervals else [])
  else:
    outputs = ["proba", "binary"]
    if spec.pooled_rank_knots is not None:
      outputs.append("rank")
    if S is not None and params.get("has_score"):
      outputs.append("score")
  return pool(
    P, R if params["has_rank"] else None, S, A if params["has_ad"] else None, spec, outputs
  )


def restrict(params, keep):
  """``params`` for a subset of its descriptors (``keep``: column names, saved order).

  Everything fitted over the whole trained set is dropped, because it does not describe a subset:
  the reference rank and the screening cutoff (classification), whose call falls back to
  ``p >= 0.5``; and the stacking weights and the interval model (regression), so the weighting
  falls back to skill and reliability and no interval is reported.
  """
  saved = params["descriptor_columns"]
  if list(keep) == list(saved):
    return params
  pos = [saved.index(c) for c in keep]
  spec = params["spec"]

  def sub(values):
    return None if values is None else tuple(values[p] for p in pos)

  spec = dataclasses.replace(
    spec,
    descriptor_names=tuple(keep),
    oof_aucs=sub(spec.oof_aucs),
    proxy_aucs=sub(spec.proxy_aucs),
    rank_error_curves=sub(spec.rank_error_curves),
    ad_hard_cutoffs=sub(spec.ad_hard_cutoffs),
    pooled_rank_knots=None,
    pooled_rank_anchors=None,
    decision_cutoff=DEFAULT_CUTOFF,
    descriptor_weights=None,
    intervals=None,
  )
  return {**without_rank({**params, "spec": spec}), "descriptor_columns": list(keep)}


def _pooled_rankless(ranks):
  """Cutoff ranks without the pooled ones (each descriptor's balanced rank does not depend on the
  pooled reference)."""
  ranks = ranks or {}
  return {k: dict(ranks.get(k) or {}) for k in ("balanced", "default_by_descriptor")}


def without_rank(params):
  """``params`` without the reference rank and the screening cutoff (call at ``p >= 0.5``).

  A no-op for a regression model, which has neither.
  """
  if params["spec"].task == REGRESSION:
    return params
  spec = dataclasses.replace(
    params["spec"], pooled_rank_knots=None, pooled_rank_anchors=None, decision_cutoff=DEFAULT_CUTOFF
  )
  return {
    **params,
    "spec": spec,
    "cutoffs": {**params["cutoffs"], "screening": None},
    "cutoff_ranks": _pooled_rankless(params.get("cutoff_ranks")),
  }


def to_artifact(params):
  """``params`` as saved: the spec as a plain dict, so a saved model does not depend on how
  lazy-qsar lays out :class:`EnsembleSpec`."""
  return {**params, "spec": dataclasses.asdict(params["spec"])}


def load_params(raw):
  """The saved weighting as current ``params``; models saved before ``ARTIFACT_VERSION`` convert.

  Those stored ``base_scores = max(0, oof - 0.5)``, which lazy-qsar's skill term reproduces from
  ``oof = base + 0.5``; their weights only differ when the model had no applicability domain
  (lazy-qsar then weights uniformly). They have no pruning, score map or kappa cutoff, and no rank
  for the balanced cutoffs (their per-descriptor reference scores are not kept with the pooled
  model).
  """
  if raw.get("version") == ARTIFACT_VERSION:
    return {**raw, "spec": EnsembleSpec(**raw["spec"])}

  names = list(raw["descriptor_columns"])
  high = raw.get("pooled_rank_anchor_high")
  knots = raw.get("pooled_rank_knots")
  screening = raw.get("decision_cutoff_proba") if knots is not None else None
  spec = EnsembleSpec(
    descriptor_names=tuple(names),
    oof_aucs=tuple(b + 0.5 for b in raw["base_scores"]),
    rank_error_curves=tuple(raw["rank_error_curves"]) if raw.get("rank_error_curves") else None,
    ad_hard_cutoffs=tuple(raw["ad_hard_cutoffs"]) if raw.get("ad_hard_cutoffs") else None,
    pooled_rank_knots=knots,
    pooled_rank_anchors=(None, high) if high is not None else None,
    decision_cutoff=screening if screening is not None else DEFAULT_CUTOFF,
  )
  return {
    "version": ARTIFACT_VERSION,
    "descriptor_columns": names,
    "pruned_columns": [],
    "spec": spec,
    "has_rank": bool(raw.get("has_rank")),
    "has_ad": bool(raw.get("has_ad")),
    "has_score": False,
    "cutoffs": {"screening": screening, "kappa": None, "default": DEFAULT_CUTOFF},
    "cutoff_ranks": {
      "screening": rank_of(screening, spec.pooled_rank_knots, spec.pooled_rank_anchors),
      "kappa": None,
      "default": rank_of(DEFAULT_CUTOFF, spec.pooled_rank_knots, spec.pooled_rank_anchors),
      "balanced": {},
      "default_by_descriptor": {},
    },
  }
