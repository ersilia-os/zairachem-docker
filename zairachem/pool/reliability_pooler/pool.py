import json
import os

import joblib
import numpy as np
import pandas as pd

from zairachem.base import ZairaBase
from zairachem.base.utils.logging import logger
from zairachem.base.utils.matrices import DEFAULT_CHUNK_SIZE
from zairachem.base.vars import (
  CUTOFF_SCREENING,
  DATA_FILENAME,
  DATA_SUBFOLDER,
  POOL_SUBFOLDER,
  POOLED_DESCRIPTORS_FILENAME,
  SMILES_COLUMN,
)

from ..base import BasePooler
from ..cutoffs import write_cutoffs
from . import pooler

#: Summary written by the pooler and rendered under the Pool step banner (see progress.py).
SUMMARY_FILENAME = "reliability_summary.json"


def _prettify(col):
  """Display name for a prediction column ('<family>-<model_id>-clf' → '<model_id>')."""
  for suf in ("-clf", "-reg"):
    if col.endswith(suf):
      col = col[: -len(suf)]
      break
  return col.split("-")[-1] if "-" in col else col


def _tier(has_rank, has_ad, stacked=False):
  # As lazy-qsar weights. Regression uses one fixed NNLS stacking weight per descriptor, which
  # replaces the skill and reliability terms; the applicability-domain veto still applies.
  if stacked:
    return "stacking weights" + (" + applicability domain" if has_ad else "")
  if not has_ad:
    return "uniform (no applicability domain)"
  if has_rank:
    return "full (skill + percentile reliability + applicability domain)"
  return "skill + applicability domain"


def _kappa_cutoff(y, p):
  """The kappa cutoff (GHOST, Cohen's kappa) of pooled scores ``p`` for labels ``y``."""
  from zairachem.estimate.tools.ghost.ghost import GhostLight

  return float(GhostLight().get_threshold(np.asarray(y), np.asarray(p)))


def _write_summary(out_path, mode, params, result, A):
  """Persist a compact, human-readable summary of how the pooler combined the descriptors.

  Rendered on-screen under the Pool step banner (progress.py); at predict time this is where the
  user learns how many query compounds fell outside the applicability domain. Best-effort.
  """
  try:
    names = [_prettify(c) for c in params["descriptor_columns"]]
    W = result.weights
    B = W.shape[0]
    summary = {
      "pooler": "reliability",
      "mode": mode,
      "n_compounds": int(B),
      "n_descriptors": len(names),
      "descriptors": names,
      "pruned": [_prettify(c) for c in params.get("pruned_columns", [])],
      "tier": _tier(
        params["has_rank"],
        params["has_ad"],
        stacked=params["spec"].descriptor_weights is not None,
      ),
      "has_rank": params["has_rank"],
      "has_ad": params["has_ad"],
      "mean_weights": {n: round(float(w), 4) for n, w in zip(names, W.mean(axis=0))},
    }
    if params["has_ad"] and A is not None:
      vetoed = W == 0
      summary["applicability"] = {
        "n_out_of_domain": int(vetoed.all(axis=1).sum()),
        "frac_out_of_domain": round(float(vetoed.all(axis=1).mean()), 4) if B else 0.0,
        "per_descriptor_out_of_domain": {n: int(c) for n, c in zip(names, vetoed.sum(axis=0))},
      }
    pool_dir = os.path.join(out_path, POOL_SUBFOLDER)
    os.makedirs(pool_dir, exist_ok=True)
    with open(os.path.join(pool_dir, SUMMARY_FILENAME), "w") as f:
      json.dump(summary, f, indent=2)
    ad_txt = ""
    if "applicability" in summary:
      ad_txt = f" | out-of-domain {summary['applicability']['n_out_of_domain']}/{B}"
    pruned = f" | pruned {', '.join(summary['pruned'])}" if summary["pruned"] else ""
    logger.info(
      f"[reliability_pool:{mode}] {len(names)} descriptors · {B} compounds · {summary['tier']}"
      f"{pruned}{ad_txt}"
    )
  except Exception as e:
    logger.debug(f"[reliability_pool:{mode}] could not write summary: {e}")


def _results(cids, result, task):
  """The pooled results dataframe, in the shape the task produces.

  Regression: ``reg`` (the predicted value in the target's own units) and, when the model carries
  an interval model, ``reg_lower`` / ``reg_upper``. Classification: ``clf``, ``clf_bin`` and, when
  available, ``clf_raw`` (pooled pre-calibration score) and ``clf_rank`` (position against the
  reference library).
  """
  v = result.values
  if task == "regression":
    results = pd.DataFrame({"reg": v["value"]})
    if "lower" in v:
      results["reg_lower"] = v["lower"]
      results["reg_upper"] = v["upper"]
  else:
    results = pd.DataFrame({"clf": v["proba"][:, 1], "clf_bin": v["binary"]})
    if "score" in v:
      results["clf_raw"] = v["score"][:, 1]
    if "rank" in v:
      # lazy-qsar's rank: 0.50 = top 10%, 0.65 = top 1% (the screening call), 0.75 = top 0.1%.
      results["clf_rank"] = v["rank"][:, 1]
  results.insert(0, "compound_id", cids)
  return results


def _balanced(meta, columns):
  """``{descriptor: balanced cutoff}`` for the prediction ``columns``."""
  return {_prettify(c): b for c, b in zip(columns, meta["balanced"])}


def _ranks_by_id(ranks):
  """Cutoff ranks with the balanced ones keyed by descriptor id (as in ``cutoffs.json``)."""
  per_descriptor = ("balanced", "default_by_descriptor")
  out = {k: v for k, v in ranks.items() if k not in per_descriptor}
  for k in per_descriptor:
    out[k] = {_prettify(c): r for c, r in (ranks.get(k) or {}).items()}
  return out


class Fitter(BasePooler):
  """Fit-time pooler: lazy-qsar's ensemble over the per-descriptor out-of-fold signals."""

  def __init__(self, path, batch_size=None):
    BasePooler.__init__(self, path=path, batch_size=batch_size)
    self.trained_path = os.path.join(self.get_output_dir(), POOL_SUBFOLDER)

  def run(self):
    """Fit the pooled model on the training rows and return their pooled predictions.

    Prunes weak descriptors, weights the rest from their out-of-fold skill, rank and AD signals,
    builds the pooled reference rank and the named cutoffs, saves the params under the run's pool
    folder and writes the pooling summary and ``pool/cutoffs.json``.
    """
    self.reset_time()
    valid_idxs = self.get_validation_indices(path=self.path)
    cids = self._get_compound_ids()
    df_X = self._get_X()
    y = np.asarray(self._get_y())

    pred_cols = self._get_pred_columns(df_X, self.task)
    if len(pred_cols) == 0:
      raise RuntimeError("[reliability_pool] No per-descriptor prediction columns found to pool.")
    clf = self.task == "classification"
    P = np.asarray(df_X[pred_cols], dtype=np.float64)
    R, _ = self._get_rank_matrix(df_X, pred_cols)
    A, _ = self._get_ad_matrix(df_X, pred_cols)
    S, _ = self._get_signal_matrix(df_X, pred_cols, "_raw") if clf else (None, False)
    meta = self._get_descriptor_meta(pred_cols, self.task)

    # Every aligned matrix sliced by the SAME indices, so rows stay paired.
    idx = np.asarray(valid_idxs)

    def rows(M):
      return None if M is None else M[idx]

    params, result = pooler.fit(
      P[idx],
      rows(R),
      rows(S),
      rows(A),
      y[idx],
      pred_cols,
      meta,
      reference=self._get_reference_signals(pred_cols, self.task) if clf else None,
      kappa=_kappa_cutoff if clf else None,
      task=self.task,
    )
    if params["pruned_columns"]:
      logger.warning(
        f"[reliability_pool:fit] left out of the pooled model (out-of-fold AUROC below "
        f"{pooler.PRUNE_FLOOR} or {pooler.PRUNE_GAP} below the best): "
        f"{', '.join(_prettify(c) for c in params['pruned_columns'])}"
      )
    if not clf:
      pass
    elif params["spec"].pooled_rank_knots is None:
      logger.warning("[reliability_pool:fit] no usable rank reference; `clf_rank` is unavailable")
    else:
      logger.info(
        f"[reliability_pool:fit] rank reference: {params['n_reference']:,} molecules · "
        f"screening cutoff = probability {params['cutoffs'][CUTOFF_SCREENING]:.4f}"
      )
    os.makedirs(self.trained_path, exist_ok=True)
    joblib.dump(pooler.to_artifact(params), os.path.join(self.trained_path, pooler.ARTIFACT_NAME))
    with open(os.path.join(self.trained_path, POOLED_DESCRIPTORS_FILENAME), "w") as f:
      json.dump(
        {
          "pooled": [_prettify(c) for c in params["descriptor_columns"]],
          "pruned": [_prettify(c) for c in params["pruned_columns"]],
        },
        f,
        indent=2,
      )
    _write_summary(self.path, "fit", params, result, rows(A))
    if clf:
      # The named cutoffs are classification concepts: a regression run writes no cutoffs.json.
      write_cutoffs(
        self.path,
        params["cutoffs"],
        _balanced(meta, pred_cols),
        _ranks_by_id(params["cutoff_ranks"]),
      )
    out = _results([cids[i] for i in idx], result, self.task)
    self.update_elapsed_time()
    return out


class Predictor(BasePooler):
  """Predict-time pooler: applies the trained model's pooled weighting."""

  def __init__(self, path, batch_size=None):
    BasePooler.__init__(self, path=path, batch_size=batch_size)
    self.trained_path = os.path.join(self.get_trained_dir(), POOL_SUBFOLDER)

  def run(self):
    """Pool every compound's per-descriptor predictions with the trained weighting.

    Descriptors missing now are dropped and the weights renormalised over the rest; the rank and
    the screening call are then unavailable (they describe the pooled probability of all of them),
    as they are when a signal used at fit is missing. Returns ``compound_id``, ``clf``, ``clf_bin``,
    ``train_similarity`` (Tanimoto similarity to the training set), ``descriptor_std`` (spread of
    the per-descriptor probabilities) and, when available, ``clf_raw`` and ``clf_rank``.
    """
    self.reset_time()
    df = self._get_X()
    cids = self._get_compound_ids()

    pred_cols = self._get_pred_columns(df, self.task)
    params = pooler.load_params(joblib.load(os.path.join(self.trained_path, pooler.ARTIFACT_NAME)))
    saved = params["descriptor_columns"]
    keep = [c for c in saved if c in pred_cols]
    if not keep:
      raise RuntimeError(
        "[reliability_pool] None of the trained descriptors are available at predict."
      )
    clf = self.task == "classification"
    if keep != saved:
      lost = "`clf_rank` and the screening call are" if clf else "the prediction intervals are"
      logger.warning(
        f"[reliability_pool:predict] descriptor drift: using {len(keep)}/{len(saved)} trained "
        f"descriptors; {lost} unavailable"
      )
      params = pooler.restrict(params, keep)

    P = np.asarray(df[keep], dtype=np.float64)
    R, _ = self._get_rank_matrix(df, keep)
    A, _ = self._get_ad_matrix(df, keep)
    S, _ = self._get_signal_matrix(df, keep, "_raw") if clf else (None, False)
    missing = [
      name
      for name, fitted, now in (("rank", params["has_rank"], R), ("AD", params["has_ad"], A))
      if fitted and now is None
    ]
    if missing and clf and params["spec"].pooled_rank_knots is not None:
      # The reference was pooled with these signals; without them the queries are weighted
      # differently, so a position against the reference would not mean what it does at fit.
      logger.warning(
        f"[reliability_pool:predict] no {' or '.join(missing)} signals for these molecules; "
        "`clf_rank` and the screening call are unavailable"
      )
      params = pooler.without_rank(params)
    if missing:
      params = {**params, **{f"has_{n.lower()}": False for n in missing}}

    result = pooler.predict(params, P, R, S, A)
    _write_summary(self.path, "predict", params, result, A)
    if clf:
      trained_meta = self._get_descriptor_meta(keep, self.task)
      write_cutoffs(
        self.path,
        params["cutoffs"],
        _balanced(trained_meta, keep),
        _ranks_by_id(params["cutoff_ranks"]),
      )
    results = _results(cids, result, self.task)
    # Per-compound signals for the predict report, after the pooled columns.
    results["train_similarity"] = self._train_similarity()
    # Disagreement between the descriptor models: the spread of what they predicted (probabilities,
    # or values in the target's own units).
    results["descriptor_std"] = np.nanstd(P, axis=1)
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
