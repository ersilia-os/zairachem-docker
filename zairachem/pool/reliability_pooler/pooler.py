"""Per-sample, per-descriptor reliability pooler.

A port of LazyQSAR's outer multi-descriptor pooler (``lazyqsar/qsar.py::_build_weight_matrix`` and
its logit-space combination) adapted to zairachem's pool step. It replaces the bagger's global,
probability-space weighting with per-sample weights that blend each descriptor's global skill with
its per-sample reliability (a rank→error curve), apply an applicability-domain (AD) hard veto, and
combine the calibrated probabilities in logit space.

The math is task-neutral and pure-numpy; persistence is a single joblib dict so inference reproduces
the exact weighting decided at fit time. Inputs (rank, AD, curves, skill) are all optional — missing
signals degrade gracefully (see ``WeightMatrixBuilder.build``).
"""

import os
import joblib
import numpy as np

EPS = 1e-7


class WeightMatrixBuilder(object):
  @staticmethod
  def build(P, R, A, base_scores, rank_error_curves, ad_hard_cutoffs):
    """Return ``(W, base)`` — a row-normalized (B, D) per-sample weight matrix.

    Parameters
    ----------
    P : (B, D) calibrated predictions per descriptor (used only for shape here).
    R : (B, D) rank quantiles in [0, 1], or None.
    A : (B, D) applicability-domain scores, or None when no AD signal exists.
    base_scores : (D,) global skill per descriptor (e.g. max(0, oof_auc − 0.5)).
    rank_error_curves : list len D of (r_knots, e_knots) or None.
    ad_hard_cutoffs : (D,) veto thresholds (use -inf to disable a descriptor's veto), or None.
    """
    B, D = P.shape
    base = np.array(base_scores, dtype=np.float64)
    if base.size != D:
      base = np.ones(D, dtype=np.float64)
    if not np.isfinite(base).all():
      base = np.where(np.isfinite(base), base, 0.0)
    if base.sum() <= 0:
      base = np.ones(D, dtype=np.float64)

    if A is not None:
      A = np.asarray(A, dtype=np.float64)
      if R is not None:
        R = np.asarray(R, dtype=np.float64)
        # NaN rank → neutral 0.5 (neither reliable nor unreliable).
        R = np.where(np.isfinite(R), R, 0.5)
        curves_ok = rank_error_curves is not None and all(c is not None for c in rank_error_curves)
        if curves_ok:
          reliability = np.zeros((B, D), dtype=np.float64)
          for j in range(D):
            r_knots, e_knots = rank_error_curves[j]
            reliability[:, j] = 1.0 - np.interp(R[:, j], r_knots, e_knots)
          reliability = np.clip(reliability, 0.0, 1.0)
        else:
          # Fallback reliability: extreme ranks (far from 0.5) are assumed more confident.
          reliability = np.abs(R - 0.5) * 2.0
        W = 0.5 * base[np.newaxis, :] + 0.5 * reliability
      else:
        W = np.tile(base, (B, 1))

      # AD hard veto: zero a descriptor's weight where its AD score falls below the cutoff.
      # NaN AD passes (no veto) so a half-built upstream never zeroes all weights.
      if ad_hard_cutoffs is not None:
        cut = np.array(ad_hard_cutoffs, dtype=np.float64)
        for j in range(D):
          if np.isfinite(cut[j]):
            vetoed = np.isfinite(A[:, j]) & (A[:, j] < cut[j])
            W[vetoed, j] = 0.0

      # All-OOD rows (every descriptor vetoed) → restore the global skill weights.
      all_ood = W.sum(axis=1) == 0
      if all_ood.any():
        W[all_ood] = base

      W = W / W.sum(axis=1, keepdims=True)
    else:
      # No AD signal: weight purely by global skill (uniform when no skill info is available,
      # since base was reset to ones above).
      W = np.tile(base, (B, 1))
      W = W / W.sum(axis=1, keepdims=True)

    return W, base


class ReliabilityClassifierPooler(object):
  """Per-sample weighted, logit-space classification pooler."""

  ARTIFACT_NAME = "reliability_pool_clf.joblib"

  def __init__(self, path):
    self.path = path
    if not os.path.exists(self.path):
      os.makedirs(self.path, exist_ok=True)
    self._params = None

  def _artifact_path(self):
    return os.path.join(self.path, self.ARTIFACT_NAME)

  @staticmethod
  def _base_scores(oof_aucs):
    return [max(0.0, float(a) - 0.5) if a is not None else 0.0 for a in oof_aucs]

  def fit(self, P, R, A, y, oof_aucs, rank_error_curves, ad_cutoffs, descriptor_columns):
    P = np.asarray(P, dtype=np.float64)
    D = P.shape[1]
    base_scores = self._base_scores(oof_aucs)
    has_ad = A is not None
    has_rank = R is not None
    # None cutoffs → -inf (that descriptor is never vetoed); only meaningful when AD is present.
    ad_hard_cutoffs = None
    if has_ad:
      ad_hard_cutoffs = [
        float(c) if c is not None else float("-inf") for c in (ad_cutoffs or [None] * D)
      ]
    self._params = {
      "descriptor_columns": list(descriptor_columns),
      "base_scores": base_scores,
      "rank_error_curves": rank_error_curves,
      "ad_hard_cutoffs": ad_hard_cutoffs,
      "has_rank": has_rank,
      "has_ad": has_ad,
      "eps": EPS,
    }
    joblib.dump(self._params, self._artifact_path())
    return self._combine(P, R, A, self._params)

  def _load(self):
    if self._params is None:
      self._params = joblib.load(self._artifact_path())
    return self._params

  def predict(self, P, R, A):
    params = self._load()
    P = np.asarray(P, dtype=np.float64)
    # Honor the tier decided at fit: if AD/rank were absent then, ignore them now.
    if not params.get("has_ad", False):
      A = None
    if not params.get("has_rank", False):
      R = None
    return self._combine(P, R, A, params)

  @staticmethod
  def _combine_verbose(P, R, A, params):
    """Return ``(p1, W)``: pooled probabilities and the per-sample weight matrix used."""
    P = np.asarray(P, dtype=np.float64)
    W, _ = WeightMatrixBuilder.build(
      P,
      R,
      A,
      params["base_scores"],
      params.get("rank_error_curves"),
      params.get("ad_hard_cutoffs"),
    )
    eps = params.get("eps", EPS)
    Pc = np.clip(P, eps, 1 - eps)
    logits = np.log(Pc / (1 - Pc))
    p1 = 1.0 / (1.0 + np.exp(-(W * logits).sum(axis=1)))
    return p1, W

  @staticmethod
  def _combine(P, R, A, params):
    p1, _ = ReliabilityClassifierPooler._combine_verbose(P, R, A, params)
    return p1
