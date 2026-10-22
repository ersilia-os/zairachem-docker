import json
import os

import joblib
import numpy as np
import pandas as pd

# Imported at module load, i.e. before the CLI re-asserts its loguru sinks, which importing
# lazy-qsar clears.
from lazyqsar.utils.ranking import (
  DECISION_RANK,
  prepare_knots,
  proba_from_reference_rank,
  rank_from_reference,
  reference_anchor_table,
  subsample_knots,
)

from zairachem.base import ZairaBase
from zairachem.base.utils.logging import logger
from zairachem.base.utils.matrices import DEFAULT_CHUNK_SIZE
from zairachem.base.vars import DATA_FILENAME, DATA_SUBFOLDER, POOL_SUBFOLDER, SMILES_COLUMN

from ..base import BasePooler
from .pooler import ReliabilityClassifierPooler

#: Summary written by the pooler and rendered under the Pool step banner (see progress.py).
SUMMARY_FILENAME = "reliability_summary.json"


def _check_regression(task):
  if task == "regression":
    raise NotImplementedError(
      "Regression is not supported yet: lazyqsar has no regressor, so there are no per-descriptor "
      "regression predictions to pool."
    )


def _prettify(col):
  """Display name for a prediction column ('<family>-<model_id>-clf' → '<model_id>')."""
  for suf in ("-clf", "-reg"):
    if col.endswith(suf):
      col = col[: -len(suf)]
      break
  return col.split("-")[-1] if "-" in col else col


def _tier(has_rank, has_ad, base_scores):
  if has_rank and has_ad:
    return "full (percentile reliability + applicability domain)"
  if has_ad:
    return "applicability-domain weighted"
  if has_rank:
    return "percentile-reliability weighted"
  return "skill weighted" if any(b > 0 for b in (base_scores or [])) else "uniform mean"


def _build_reference_rank(params, ref, p_valid, y_valid):
  """Fit the pooled reference rank into ``params`` (in place), mirroring lazy-qsar's ensemble.

  ``ref`` is the rank-reference library's per-descriptor ``(P, R, A)``. It goes through the same
  fitted weighting a query does, so its pooled probabilities are on the query scale; their knots are
  what ``rank`` is a position against, as in lazy-qsar's ensemble (``LazyClassifierQSAR``).
  The top of the scale is pinned on p95 of the validation actives' pooled probability, and the
  decision cutoff is :data:`~lazyqsar.utils.ranking.DECISION_RANK` (top 1% of the library) mapped
  back to a probability. Leaves ``params`` without a rank when there is no usable reference.
  """
  params["pooled_rank_knots"] = None
  if ref is None:
    logger.warning("[reliability_pool:fit] no rank-reference scores; `clf_rank` is unavailable")
    return
  P_ref, R_ref, A_ref = ref
  if params.get("has_ad") and A_ref is None:
    logger.warning("[reliability_pool:fit] rank reference has no AD scores; `clf_rank` unavailable")
    return
  p_ref, _ = ReliabilityClassifierPooler._combine_verbose(
    P_ref,
    R_ref if params.get("has_rank") else None,
    A_ref if params.get("has_ad") else None,
    params,
  )
  p_ref = np.asarray(p_ref)[np.isfinite(p_ref)]
  if p_ref.size == 0:
    return
  knots = subsample_knots(np.sort(p_ref))
  prepared = prepare_knots(knots)
  actives = np.asarray(p_valid)[np.asarray(y_valid) == 1]
  high = float(np.percentile(actives, 95)) if actives.size else None
  _, _, used = reference_anchor_table(prepared=prepared, anchor_high=high)
  params["pooled_rank_knots"] = knots
  params["pooled_rank_anchor_high"] = high if used else None
  params["decision_cutoff_proba"] = float(
    proba_from_reference_rank(DECISION_RANK, prepared=prepared, anchor_high=high if used else None)
  )
  logger.info(
    f"[reliability_pool:fit] rank reference: {p_ref.size:,} molecules · decision cutoff "
    f"rank {DECISION_RANK} = probability {params['decision_cutoff_proba']:.4f}"
  )


def _rank_outputs(p, params):
  """``(clf_bin, clf_rank)`` for pooled probabilities ``p``; ``clf_rank`` is None without a reference.

  With a reference the call is ``rank >= DECISION_RANK`` (top 1% of drug-like space, as lazy-qsar's
  ``binary``); without one it falls back to ``p > 0.5``.
  """
  p = np.asarray(p, dtype=np.float64)
  knots = params.get("pooled_rank_knots")
  if knots is None or not len(knots):
    return (p > 0.5).astype(int), None
  high = params.get("pooled_rank_anchor_high")
  rank = rank_from_reference(
    p,
    prepared=prepare_knots(np.asarray(knots, dtype=np.float64)),
    anchors=(None, high) if high is not None else None,
  )
  return (p >= params["decision_cutoff_proba"]).astype(int), rank


def _veto_stats(A, ad_hard_cutoffs):
  """(per-descriptor vetoed counts, n_fully_out_of_domain) from AD scores vs per-descriptor cutoffs."""
  if A is None or ad_hard_cutoffs is None:
    return None, 0
  A = np.asarray(A, dtype=np.float64)
  cut = np.asarray(ad_hard_cutoffs, dtype=np.float64)
  B, D = A.shape
  vetoed = np.zeros((B, D), dtype=bool)
  for j in range(D):
    if np.isfinite(cut[j]):
      vetoed[:, j] = np.isfinite(A[:, j]) & (A[:, j] < cut[j])
  per = vetoed.sum(axis=0).astype(int).tolist()
  has_cut = np.isfinite(cut)
  n_fully = int(vetoed[:, has_cut].all(axis=1).sum()) if has_cut.any() else 0
  return per, n_fully


def _write_summary(out_path, mode, pred_cols, params, P, R, A, W):
  """Persist a compact, human-readable summary of how the pooler combined the descriptors.

  Rendered on-screen under the Pool step banner (progress.py) — at predict time this is where the
  user learns how many query compounds fell outside the applicability domain. Best-effort.
  """
  try:
    B, D = P.shape
    names = [_prettify(c) for c in pred_cols]
    has_rank = bool(params.get("has_rank"))
    has_ad = bool(params.get("has_ad"))
    base_scores = list(params.get("base_scores") or [])
    mean_w = W.mean(axis=0)
    summary = {
      "pooler": "reliability",
      "mode": mode,
      "n_compounds": int(B),
      "n_descriptors": int(D),
      "descriptors": names,
      "tier": _tier(has_rank, has_ad, base_scores),
      "has_rank": has_rank,
      "has_ad": has_ad,
      "mean_weights": {n: round(float(w), 4) for n, w in zip(names, mean_w)},
    }
    if has_ad:
      per, n_fully = _veto_stats(A, params.get("ad_hard_cutoffs"))
      summary["applicability"] = {
        "n_out_of_domain": int(n_fully),
        "frac_out_of_domain": round(float(n_fully) / B, 4) if B else 0.0,
        "per_descriptor_out_of_domain": (
          {n: int(c) for n, c in zip(names, per)} if per is not None else {}
        ),
      }
    pool_dir = os.path.join(out_path, POOL_SUBFOLDER)
    os.makedirs(pool_dir, exist_ok=True)
    with open(os.path.join(pool_dir, SUMMARY_FILENAME), "w") as f:
      json.dump(summary, f, indent=2)
    # File-log line (also visible on-screen with --verbose); the durable on-screen surface is the
    # Pool step detail block rendered from this artifact.
    ad_txt = ""
    if has_ad and "applicability" in summary:
      ad_txt = f" | out-of-domain {summary['applicability']['n_out_of_domain']}/{B}"
    logger.info(
      f"[reliability_pool:{mode}] {D} descriptors · {B} compounds · {summary['tier']}{ad_txt}"
    )
  except Exception as e:
    logger.debug(f"[reliability_pool:{mode}] could not write summary: {e}")


class Fitter(BasePooler):
  """Fit-time reliability pooler that learns the per-descriptor weighting on validation rows."""

  def __init__(self, path, batch_size=None):
    BasePooler.__init__(self, path=path, batch_size=batch_size)
    self.trained_path = os.path.join(self.get_output_dir(), POOL_SUBFOLDER)

  def run(self):
    """Fit the pooler on the validation rows and return their pooled predictions.

    Builds the weighting from OOF skill, rank and AD signals, adds the pooled reference rank,
    saves the params under the run's pool folder and writes the pooling summary. Returns a
    dataframe with ``compound_id``, ``clf``, ``clf_bin`` and, when available, ``clf_raw``
    (pooled uncalibrated score) and ``clf_rank``.
    """
    self.reset_time()
    _check_regression(self.task)
    valid_idxs = self.get_validation_indices(path=self.path)
    cids = self._get_compound_ids()
    df_X = self._get_X()
    y = np.asarray(self._get_y())

    pred_cols = self._get_pred_columns(df_X, self.task)
    if len(pred_cols) == 0:
      raise RuntimeError("[reliability_pool] No per-descriptor prediction columns found to pool.")

    P = np.asarray(df_X[pred_cols], dtype=np.float64)
    R, _ = self._get_rank_matrix(df_X, pred_cols)
    A, _ = self._get_ad_matrix(df_X, pred_cols)
    oof_aucs, curves, cutoffs = self._get_descriptor_meta(pred_cols, self.task)
    logger.info(
      f"[reliability_pool:fit] {len(pred_cols)} descriptors | rank={'y' if R is not None else 'n'} "
      f"ad={'y' if A is not None else 'n'}"
    )

    # Slice every aligned matrix by the SAME validation indices so rows stay paired.
    idx = np.asarray(valid_idxs)
    P_v = P[idx]
    R_v = R[idx] if R is not None else None
    A_v = A[idx] if A is not None else None
    y_v = y[idx]
    cids_v = [cids[i] for i in idx]

    pooler = ReliabilityClassifierPooler(self.trained_path)
    y_hat = pooler.fit(P_v, R_v, A_v, y_v, oof_aucs, curves, cutoffs, pred_cols)
    _, W = ReliabilityClassifierPooler._combine_verbose(P_v, R_v, A_v, pooler._params)
    _write_summary(self.path, "fit", pred_cols, pooler._params, P_v, R_v, A_v, W)
    _build_reference_rank(
      pooler._params, self._get_reference_signals(pred_cols, self.task), y_hat, y_v
    )
    pooler.save()
    b_hat, rank = _rank_outputs(y_hat, pooler._params)

    results = pd.DataFrame({"clf": y_hat, "clf_bin": b_hat})
    # Pooled UNCALIBRATED score (report raw lens): combine the per-descriptor clf_raw columns with the
    # SAME weights (same R/A/params → same W). Only when every descriptor carries a raw column (fit
    # time with lazy-qsar ≥ 3.4.2); absent otherwise.
    raw_cols = [c + "_raw" for c in pred_cols]
    if all(c in df_X.columns for c in raw_cols):
      P_raw_v = np.asarray(df_X[raw_cols], dtype=np.float64)[idx]
      y_hat_raw, _ = ReliabilityClassifierPooler._combine_verbose(P_raw_v, R_v, A_v, pooler._params)
      results["clf_raw"] = np.asarray(y_hat_raw)
    if rank is not None:
      results["clf_rank"] = rank
    columns = results.columns.tolist()
    results["compound_id"] = cids_v
    results = results[["compound_id"] + columns]
    self.update_elapsed_time()
    return results


class Predictor(BasePooler):
  """Predict-time reliability pooler that applies the weighting saved in the trained model dir."""

  def __init__(self, path, batch_size=None):
    BasePooler.__init__(self, path=path, batch_size=batch_size)
    self.trained_path = os.path.join(self.get_trained_dir(), POOL_SUBFOLDER)

  def run(self):
    """Pool every compound's per-descriptor predictions with the saved weighting.

    Descriptors missing now are dropped and weights renormalised over the rest (``clf_rank``
    is then disabled). Returns a dataframe with ``compound_id``, ``clf``, ``clf_bin``,
    ``train_similarity`` (Tanimoto similarity to the training set), ``descriptor_std`` (spread of
    the per-descriptor probabilities) and, when available, ``clf_rank``.
    """
    self.reset_time()
    _check_regression(self.task)
    df = self._get_X()
    cids = self._get_compound_ids()

    pred_cols = self._get_pred_columns(df, self.task)
    params = joblib.load(os.path.join(self.trained_path, ReliabilityClassifierPooler.ARTIFACT_NAME))
    saved = params["descriptor_columns"]
    # Reconcile descriptor drift: keep saved-order columns that are present now; renormalize over
    # them. A descriptor present at fit but absent now is dropped (no base score to apply).
    keep = [c for c in saved if c in pred_cols]
    if len(keep) == 0:
      raise RuntimeError(
        "[reliability_pool] None of the trained descriptors are available at predict."
      )
    if keep != saved:
      logger.warning(
        f"[reliability_pool:predict] descriptor drift: using {len(keep)}/{len(saved)} trained "
        "descriptors; `clf_rank` is unavailable (its reference was pooled over all of them)"
      )
    pos = [saved.index(c) for c in keep]

    P = np.asarray(df[keep], dtype=np.float64)
    R, _ = self._get_rank_matrix(df, keep) if params.get("has_rank") else (None, False)
    A, _ = self._get_ad_matrix(df, keep) if params.get("has_ad") else (None, False)

    reduced = dict(params)
    reduced["descriptor_columns"] = keep
    reduced["base_scores"] = [params["base_scores"][p] for p in pos]
    if keep != saved:
      reduced["pooled_rank_knots"] = None
    missing = [
      name
      for name, fitted, now in (
        ("rank", params.get("has_rank"), R),
        ("AD", params.get("has_ad"), A),
      )
      if fitted and now is None
    ]
    if missing and reduced.get("pooled_rank_knots") is not None:
      # The reference was pooled with these signals; without them the queries are weighted
      # differently, so a position against the reference would not mean what it does at fit.
      logger.warning(
        f"[reliability_pool:predict] no {' or '.join(missing)} signals for these molecules; "
        "`clf_rank` is unavailable (the reference was pooled with them)"
      )
      reduced["pooled_rank_knots"] = None
    if params.get("rank_error_curves") is not None:
      reduced["rank_error_curves"] = [params["rank_error_curves"][p] for p in pos]
    if params.get("ad_hard_cutoffs") is not None:
      reduced["ad_hard_cutoffs"] = [params["ad_hard_cutoffs"][p] for p in pos]

    y_hat, W = ReliabilityClassifierPooler._combine_verbose(P, R, A, reduced)
    _write_summary(self.path, "predict", keep, reduced, P, R, A, W)
    if keep == saved and params.get("pooled_rank_knots") is None:
      logger.warning(
        "[reliability_pool:predict] model has no pooled rank reference (refit to get `clf_rank`)"
      )
    b_hat, rank = _rank_outputs(y_hat, reduced)

    results = pd.DataFrame({"clf": y_hat, "clf_bin": b_hat})
    # Per-compound signals for the predict report, appended AFTER clf/clf_bin so the first-match
    # "clf" column scanners still resolve to the base prediction. The applicability domain still
    # vetoes descriptors inside the weighting above; it is not reported.
    results["train_similarity"] = self._train_similarity()
    # Disagreement between the descriptor models: std of the probabilities the pooler combined.
    results["descriptor_std"] = np.nanstd(P, axis=1)
    if rank is not None:
      # Position of the pooled probability against the rank-reference library (lazy-qsar's `rank`:
      # 0.50 = top 10%, 0.65 = top 1% and the binary cutoff, 0.75 = top 0.1%).
      results["clf_rank"] = rank
    columns = results.columns.tolist()
    results["compound_id"] = cids
    results = results[["compound_id"] + columns]
    self.update_elapsed_time()
    return results

  def _train_similarity(self):
    """Return each compound's Tanimoto similarity to the trained model's training molecules."""
    from zairachem.base.utils.similarity import training_similarity

    def smiles(folder):
      return list(pd.read_csv(os.path.join(folder, DATA_SUBFOLDER, DATA_FILENAME))[SMILES_COLUMN])

    return training_similarity(
      smiles(self.path), smiles(self.get_trained_dir()), os.path.join(self.path, POOL_SUBFOLDER)
    )


class ReliabilityPooler(ZairaBase):
  """Reliability pool step, dispatching to the fitter or the predictor by run mode."""

  def __init__(self, path=None, batch_size=None):
    ZairaBase.__init__(self)
    self.batch_size = batch_size or DEFAULT_CHUNK_SIZE
    self.path = path
    if not self.is_predict():
      self.estimator = Fitter(path=self.path, batch_size=self.batch_size)
    else:
      self.estimator = Predictor(path=self.path, batch_size=self.batch_size)

  def run(self):
    """Run the fitter (fit mode) or the predictor (predict mode); return the pooled results."""
    return self.estimator.run()
