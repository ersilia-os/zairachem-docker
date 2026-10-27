"""Per-step result summaries and detail blocks for the ZairaChem CLI.

Reads the run's artifacts (parameters, data.csv, estimator/pool outputs) to produce the one-line
``✓`` summary per step (:data:`SUMMARIES`), the calm detail rows printed under each step banner
(:func:`_detail_rows`) and the closing run-summary panel (:func:`final_summary_panel`). Pure
read-from-disk + formatting; depends only on the console renderer, not on the live widgets or the
tracker (which it touches lazily, only inside :func:`final_summary_panel`).

Re-exported from :mod:`zairachem.base.utils.progress` for backwards compatibility.
"""

import csv
import json
import math
import os

from zairachem.base.utils.console import summary_panel
from zairachem.base.utils.logging import logger
from zairachem.base.vars import CV_SKILL, SKILL_FLOOR

#: Steps that render a persistent live per-item table (see :class:`..live.LiveTableMonitor`). That
#: table stays on screen as the step's record, so these steps intentionally print no separate detail
#: block (``_detail_rows`` returns [] for them).
_LIVE_TABLE_STEPS = {"describe", "projections", "treat", "estimate"}


def _resolve_output_dir(output_dir=None):
  return os.path.abspath(output_dir) if output_dir else None


def _is_predict(output_dir):
  """True if the run in ``output_dir`` is a prediction run (so step text can say 'apply', not
  'train'). Reads the run's own session.json — no global pointer."""
  from zairachem.base.vars import SESSION_FILE

  try:
    with open(os.path.join(output_dir, SESSION_FILE)) as f:
      return json.load(f).get("mode") == "predict"
  except Exception:
    return False


#: Step descriptions that differ at predict (the run applies a trained model rather than building one).
_PREDICT_STEP_DESC = {
  "estimate": "Apply the trained models to score each molecule",
  "treat": "Apply the fitted transformers to the descriptor matrix",
}


def _load_params(output_dir):
  from zairachem.base.vars import METADATA_SUBFOLDER, PARAMETERS_FILE

  try:
    with open(os.path.join(output_dir, METADATA_SUBFOLDER, PARAMETERS_FILE)) as f:
      return json.load(f)
  except Exception:
    return {}


def _n_compounds(output_dir):
  from zairachem.base.vars import DATA_FILENAME, DATA_SUBFOLDER

  try:
    import pandas as pd

    return len(pd.read_csv(os.path.join(output_dir, DATA_SUBFOLDER, DATA_FILENAME)))
  except Exception:
    return None


def _raw_h5_path(base):
  """Path to a descriptor's ``raw.h5`` (or its first raw chunk) under ``base``; None if absent."""
  h5 = os.path.join(base, "raw.h5")
  if not os.path.exists(h5):
    chunks = os.path.join(base, "raw_chunks", "chunk_0000.h5")
    h5 = chunks if os.path.exists(chunks) else None
  return h5


def _h5_width(h5py, h5):
  """Column count of a descriptor H5 file (``Features`` or ``Values``); None if neither exists."""
  with h5py.File(h5, "r") as f:
    if "Features" in f:
      return int(f["Features"].shape[0])
    if "Values" in f:
      return int(f["Values"].shape[1])
    return None


def _descriptor_feature_width(output_dir, featurizer_ids):
  """Total descriptor columns across all featurizers (best-effort; None if unreadable)."""
  from zairachem.base.vars import DESCRIPTORS_SUBFOLDER

  total = 0
  try:
    import h5py

    for eos in featurizer_ids:
      h5 = _raw_h5_path(os.path.join(output_dir, DESCRIPTORS_SUBFOLDER, eos))
      if not h5:
        return None
      width = _h5_width(h5py, h5)
      if width is None:
        return None
      total += width
    return total or None
  except Exception:
    return None


def _onnx_algorithm(filename):
  """Algorithm name of an exported ``.onnx`` file; None for other files and preprocessor/pooler."""
  if not filename.endswith(".onnx"):
    return None
  name = filename[:-5]
  return None if name in ("preprocessor", "pooler") else name


def _estimator_algorithms(output_dir):
  """Algorithm names (from exported .onnx bundles), excluding preprocessor/pooler."""
  from zairachem.base.vars import ESTIMATORS_SUBFOLDER

  algos = set()
  try:
    for root, _dirs, files in os.walk(os.path.join(output_dir, ESTIMATORS_SUBFOLDER)):
      algos.update(name for name in map(_onnx_algorithm, files) if name is not None)
  except OSError as e:
    logger.debug(f"[summary] could not list estimator algorithms: {e}")
  return sorted(algos)


def _count_plots(output_dir):
  from zairachem.base.vars import REPORT_SUBFOLDER

  try:
    p = os.path.join(output_dir, REPORT_SUBFOLDER, "png")
    return sum(1 for f in os.listdir(p) if f.endswith(".png"))
  except Exception:
    return None


def _collapse(path):
  home = os.path.expanduser("~")
  return "~" + path[len(home) :] if path and path.startswith(home) else path


def _plurals(n, word):
  return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _reliability_summary(output_dir):
  """The reliability pooler's run summary (pool/reliability_summary.json), or None."""
  from zairachem.base.vars import POOL_SUBFOLDER

  try:
    with open(os.path.join(output_dir, POOL_SUBFOLDER, "reliability_summary.json")) as f:
      return json.load(f)
  except Exception:
    return None


def _pooled_metrics(output_dir):
  """The pooled model's performance-table row at the run's call cutoff (screening, else default).

  A regression table has one row per model and no ``cutoff`` column at all, so the lookup falls
  back to the row's own name and matches on the model alone -- which is the whole selection there.
  """
  try:
    from zairachem.base.vars import PERFORMANCE_TABLE_FILENAME, RESULTS_SUBFOLDER
    from zairachem.pool.cutoffs import call_cutoff

    name = call_cutoff(output_dir)[0]
    with open(os.path.join(output_dir, RESULTS_SUBFOLDER, PERFORMANCE_TABLE_FILENAME)) as f:
      for row in csv.DictReader(f):
        # `.get("cutoff", name)` makes a cutoff-less (regression) row match unconditionally.
        if row.get("model") == "pooled" and row.get("cutoff", name) == name:
          return row
  except (OSError, csv.Error) as e:
    logger.debug(f"[summary] no pooled metrics: {e}")
  return None


# --- One-line summaries (the ✓ result line per step) -------------------------------------------


def _featurizers(output_dir):
  """Descriptor ids to report for this run.

  At predict only the selected (effective) descriptors are computed and scored, so the pre-screened-
  out ones are omitted from every count/summary; at fit, all requested featurizers are returned."""
  params = _load_params(output_dir)
  all_desc = params.get("featurizer_ids", []) or []
  if _is_predict(output_dir):
    try:
      from zairachem.base.utils.descriptors import effective_descriptors

      eff = effective_descriptors(output_dir)
      if eff:
        return list(eff)
    except Exception as e:  # reads several trained-model artefacts
      logger.debug(f"[summary] effective descriptors unavailable, reporting all: {e}")
  return all_desc


def summarize_setup(output_dir=None):
  """Summarize the setup step for the pipeline panel; empty string when its outputs are missing."""
  d = _resolve_output_dir(output_dir)
  if not d:
    return ""
  params = _load_params(d)
  parts = []
  n = _n_compounds(d)
  if n is not None:
    parts.append(f"{n:,} compounds")
  parts.append(params.get("task", "?"))
  parts.append(_plurals(len(_featurizers(d)), "descriptor"))
  return " · ".join(parts)


def summarize_describe(output_dir=None):
  """Summarize the describe step for the pipeline panel; empty string when its outputs are missing."""
  d = _resolve_output_dir(output_dir)
  if not d:
    return ""
  featurizers = _featurizers(d)
  n = _n_compounds(d)
  width = _descriptor_feature_width(d, featurizers)
  if n is not None and width is not None:
    head = f"{n:,} × {width:,}"
  elif n is not None:
    head = f"{n:,} compounds"
  else:
    head = _plurals(len(featurizers), "descriptor")
  return f"{head} · {_plurals(len(featurizers), 'descriptor')}"


def summarize_projections(output_dir=None):
  """Summarize the projections step for the pipeline panel; empty string when its outputs are missing."""
  d = _resolve_output_dir(output_dir)
  if not d:
    return ""
  params = _load_params(d)
  extra = params.get("projection_ids") or []
  return "MW/LogP" + (f" + {_plurals(len(extra), 'model')}" if extra else " (built-in)")


def summarize_treat(output_dir=None):
  """Summarize the treat step for the pipeline panel; empty string when its outputs are missing."""
  d = _resolve_output_dir(output_dir)
  if not d:
    return ""
  n = _n_compounds(d)
  featurizers = _featurizers(d)
  width = _descriptor_feature_width(d, featurizers)
  if n is None or width is None:
    return "descriptors imputed"
  # Each descriptor keeps its own matrix; the feature count is the total over all of them.
  return (
    f"{_plurals(n, 'compound')} × {_plurals(len(featurizers), 'descriptor')} ({width:,} features)"
  )


def _cv_stats(output_dir):
  """Per-descriptor lazy-qsar CV reports (estimators/*/*/cv_report.json)."""
  import glob

  from zairachem.base.vars import ESTIMATORS_SUBFOLDER

  stats = []
  for f in glob.glob(os.path.join(output_dir, ESTIMATORS_SUBFOLDER, "*", "*", "cv_report.json")):
    try:
      with open(f) as fh:
        r = json.load(fh)
      r["descriptor"] = os.path.basename(os.path.dirname(f))
      stats.append(r)
    except Exception:
      continue
  return stats


def _cv_headline(cv):
  """``(label, mean_oof, best_descriptor, best_oof)`` or None from a list of cv reports.

  The skill is AUROC for a classification run and Spearman's rho for a regression one, so the
  label travels with the numbers rather than being assumed by the caller.
  """
  for _task, (oof_key, _train_key, label, _chance) in CV_SKILL.items():
    skills = [s[oof_key] for s in cv if s.get(oof_key) is not None]
    if not skills:
      continue
    best = max(cv, key=lambda s: s.get(oof_key) if s.get(oof_key) is not None else -float("inf"))
    return label, sum(skills) / len(skills), best.get("descriptor"), best.get(oof_key)
  return None


def summarize_estimate(output_dir=None):
  """Summarize the estimate step for the pipeline panel; empty string when its outputs are missing."""
  d = _resolve_output_dir(output_dir)
  if not d:
    return ""
  if _is_predict(d):
    # Predict applies the trained models — there are no freshly trained estimators or CV stats here.
    n = len(_featurizers(d))
    return _plurals(n, "descriptor") + " scored" if n else "models applied"
  algos = _estimator_algorithms(d)
  base = _plurals(len(algos), "algorithm") + " trained" if algos else "estimators trained"
  head = _cv_headline(_cv_stats(d))
  if head:
    label, mean, best, best_skill = head
    base += f" · CV {label} {mean:.2f} (best {best} {best_skill:.2f})"
  return base


def summarize_pool(output_dir=None):
  """Summarize the pool step for the pipeline panel; empty string when its outputs are missing."""
  d = _resolve_output_dir(output_dir)
  if not d:
    return ""
  rel = _reliability_summary(d)
  if rel:
    parts = [_plurals(rel.get("n_descriptors", 0), "descriptor")]
    ad = rel.get("applicability")
    if ad and ad.get("n_out_of_domain"):
      parts.append(f"{ad['n_out_of_domain']:,} out-of-domain")
    return " · ".join(parts)
  algos = _estimator_algorithms(d)
  return f"consensus of {_plurals(len(algos), 'algorithm')}" if algos else "predictions pooled"


def summarize_screen(output_dir=None):
  """Summarize the screen step for the pipeline panel; empty string when its outputs are missing."""
  d = _resolve_output_dir(output_dir)
  if not d:
    return ""
  import json

  sel = os.path.join(d, "metadata", "selected_eos.json")
  scores = os.path.join(d, "metadata", "proxy_scores.json")
  if not os.path.exists(sel):
    done = os.path.join(d, "descriptors", "done_eos.json")
    try:
      with open(done) as f:
        n = len(json.load(f))
    except (OSError, ValueError):
      return ""
    return f"not needed · all {_plurals(n, 'descriptor')} kept"
  try:
    with open(sel) as f:
      n_sel = len(json.load(f))
    n_all = n_sel
    if os.path.exists(scores):
      with open(scores) as f:
        n_all = len(json.load(f))
  except Exception:
    return ""
  return f"kept {n_sel} of {n_all} descriptors"


def summarize_holdout(output_dir=None):
  """Summarize held-out validation (folds scored and failed, scaffold skill); empty when missing."""
  d = _resolve_output_dir(output_dir)
  if not d:
    return ""
  import json

  path = os.path.join(d, "report", "holdout_summary.json")
  if not os.path.exists(path):
    return ""
  try:
    with open(path) as f:
      summary = json.load(f)
  except Exception:
    return ""
  n = summary.get("n_folds_run", 0)
  failed = summary.get("n_folds_failed", 0)
  if failed:
    total = summary.get("n_folds_defined", n + failed)
    if not n:
      return f"0/{total} folds (all failed)"
    n_text = f"{n}/{total} folds ({failed} failed)"
  else:
    n_text = _plurals(n, "fold")
  # The scaffold split is the honest headline: whichever metric the folds reported for it.
  scaffold = summary.get("strategies", {}).get("scaffold", {}) or {}
  parts = [n_text]
  for metric, label in (("auroc", "AUROC"), ("r2", "R\u00b2")):
    mean = (scaffold.get(metric) or {}).get("mean")
    if mean is not None:
      parts.append(f"scaffold {label} {mean:.2f}")
      break
  return " · ".join(parts)


def summarize_report(output_dir=None):
  """Summarize the report step for the pipeline panel; empty string when its outputs are missing."""
  d = _resolve_output_dir(output_dir)
  if not d:
    return ""
  n = _count_plots(d)
  return f"{_plurals(n, 'plot')} · HTML report" if n else "report written"


def summarize_finish(output_dir=None):
  """Summarize the finish step for the pipeline panel; empty string when its outputs are missing."""
  d = _resolve_output_dir(output_dir)
  if not d:
    return ""
  return "outputs written"


SUMMARIES = {
  "setup": summarize_setup,
  "describe": summarize_describe,
  "projections": summarize_projections,
  "treat": summarize_treat,
  "screen": summarize_screen,
  "estimate": summarize_estimate,
  "pool": summarize_pool,
  "holdout": summarize_holdout,
  "report": summarize_report,
  "finish": summarize_finish,
}


# --- Per-step detail content (rich, always-on), printed borderless under the banner -------------


def _active_inactive(output_dir):
  """(n_active, n_inactive) from data.csv's binary column, or None."""
  from zairachem.base.vars import DATA_FILENAME, DATA_SUBFOLDER

  try:
    import pandas as pd

    df = pd.read_csv(os.path.join(output_dir, DATA_SUBFOLDER, DATA_FILENAME))
    if "bin" in df.columns:
      n_act = int(df["bin"].sum())
      return n_act, len(df) - n_act
  except (OSError, ValueError, KeyError) as e:
    logger.debug(f"[summary] class counts unavailable: {e}")
  return None


def _plots_by_category(output_dir):
  """Group report PNGs into coarse categories by filename keyword. Returns {category: count}.

  The keywords are matched in the order written, so the more specific ones come first. The list
  spans both tasks: a regression run renders none of the classification panels and vice versa, and
  what neither names lands in ``other``.
  """
  from zairachem.base.vars import REPORT_SUBFOLDER

  buckets = {
    "heldout": "held-out validation",
    "roc": "ROC",
    "auroc": "ROC",
    "pr-curve": "PR",
    "calib": "calibration",
    "dist": "distributions",
    "violin": "distributions",
    "strip": "distributions",
    "proj": "projections",
    "cv": "cross-validation",
    "confusion": "confusion",
    "r2": "regression",
    "obs": "regression",
    "regression": "regression",
    "value": "regression",
    "interval": "intervals",
    "enrich": "enrichment",
    "potency": "enrichment",
    "hist": "histograms",
    "oof-score": "out-of-fold scores",
    "class-": "class balance",
    "actives": "class balance",
    "property": "properties",
    "descriptor": "descriptors",
    "topk": "descriptors",
    "threshold": "thresholds",
    "timing": "compute",
    "time": "compute",
    "resource": "compute",
    "provenance": "compute",
    "rank": "rank",
    "similarity": "similarity",
  }
  out = {}
  try:
    p = os.path.join(output_dir, REPORT_SUBFOLDER, "png")
    for fn in os.listdir(p):
      if not fn.endswith(".png"):
        continue
      low = fn.lower()
      cat = next((label for kw, label in buckets.items() if kw in low), "other")
      out[cat] = out.get(cat, 0) + 1
  except Exception:
    return {}
  return out


def _dir_size_mb(output_dir):
  total = 0
  try:
    for root, _dirs, files in os.walk(output_dir):
      for f in files:
        try:
          total += os.path.getsize(os.path.join(root, f))
        except OSError:
          continue
    return total / (1024 * 1024)
  except Exception:
    return None


def _setup_rows(d, params):
  """Detail rows for the setup step."""
  rows = []
  n = _n_compounds(d)
  if params.get("task") == "classification" and _active_inactive(d):
    a, i = _active_inactive(d)
    rows.append(("compounds", f"{n:,} [dim]·[/] {a:,} active [dim]·[/] {i:,} inactive"))
  elif n is not None:
    rows.append(("compounds", f"{n:,}"))
  feats = _featurizers(d)
  if feats:
    rows.append(("featurizers", "  ".join(feats)))
  projs = params.get("projection_ids", []) or []
  rows.append((
    "projection",
    "MW/LogP" + (f"  +  {'  '.join(projs)}" if projs else " [dim](built-in)[/]"),
  ))
  store = params.get("contribute_store")
  rows.append(("store", f"on [dim]· project {store}[/]" if store else "[dim]off[/]"))
  return rows


def _finite_or_none(value):
  """``value`` as a float, or None when it is missing or could not be computed (NaN).

  Named apart from ``estimate``'s ``_float_or_none``, which keeps a NaN: here a metric that could
  not be computed must not be read as a number, because every caller compares it to a threshold.
  """
  try:
    number = float(value)
  except (TypeError, ValueError):
    return None
  return None if number != number else number  # NaN is unknown, not zero


def _distinct(value, other, places=2, limit=6):
  """``value`` with just enough decimals that it cannot print the same as ``other``.

  The two numbers here are a metric and the floor it fell below, and at two decimals a near miss
  prints as "0.10 is below 0.10" -- a contradiction that makes a reader doubt the rest of the
  panel.
  """
  while places < limit and f"{value:.{places}f}" == f"{other:.{places}f}":
    places += 1
  return f"{value:.{places}f}"


def _replicate_sd(output_dir):
  """How much repeat measurements of one compound disagreed, or None when none were repeated."""
  from zairachem.setup.prep.files import read_replicates

  return _finite_or_none(read_replicates(output_dir).get("pooled_sd"))


#: An RMSE within this multiple of the assay's own spread counts as being at the limit of the data.
#: An observed value is the truth plus measurement noise of standard deviation ``s``, so a model
#: whose own error against the truth is ``m`` scores ``RMSE^2 = m^2 + s^2``: the RMSE can never go
#: below ``s``, however good the model. At ``1.25 s`` the model's own error is ``0.75 s`` — smaller
#: than the noise in the thing it is being scored against, which is as far as the data can show.
#: It also keeps the note's claim true: the model's error is the smaller half of the total only
#: while the RMSE stays under ``sqrt(2) s``, and 1.25 is inside that.
NOISE_FLOOR_FACTOR = 1.25


def _note_if_at_the_noise_floor(rows, rmse, floor):
  """Say when the error is as small as the data allows, which is praise, not a warning.

  An RMSE is only meaningful against something, and the one thing the pipeline can measure is how
  far apart two measurements of the same compound were. A model predicting within that spread has
  extracted what the data holds; the way to improve it is more or better measurements, not more
  modelling.

  The note also separates the two. ``RMSE^2 = m^2 + s^2`` inverts, so the model's own error is
  ``sqrt(RMSE^2 - s^2)`` — and saying it matters, because the headline number invites the opposite
  reading. An RMSE of 0.548 against a spread of 0.481 looks like a model barely better than noise;
  it is in fact a model whose own error is 0.263, about half the noise it is being graded through.
  It is an estimate, resting on the measurement error being independent of the model's, which is
  the assumption that makes the identity hold in the first place.
  """
  error, spread = _finite_or_none(rmse), _finite_or_none(floor)
  if error is None or spread is None or spread <= 0:
    return
  if error > spread * NOISE_FLOOR_FACTOR:
    return
  own = math.sqrt(max(0.0, error**2 - spread**2))
  rows.append((
    "",
    f"[green]At the limit of this data[/] [dim]— repeat measurements of the same compound differ "
    f"by {spread:.3f}, so an RMSE of {error:.3f} is as small as this data can show. Most of it is "
    f"the assay: the model's own error is about {own:.3f}.[/]",
  ))


def _note_training_overlap(rows, output_dir):
  """Append a caution when molecules being predicted were in the model's training set.

  Their predicted value is a recollection of the label the model was fitted on, so including them
  in a summary of how the model does on new chemistry flatters it.
  """
  from zairachem.base.vars import DATA_SUBFOLDER, TRAIN_OVERLAP_FILENAME

  path = os.path.join(output_dir, DATA_SUBFOLDER, TRAIN_OVERLAP_FILENAME)
  try:
    with open(path) as f:
      overlap = json.load(f)
  except Exception:
    return
  seen, total = overlap.get("n_in_training", 0), overlap.get("n_query", 0)
  if not seen or not total:
    return
  rows.append((
    "",
    f"[yellow]Already seen[/] [dim]— {seen:,} of {total:,} of these molecules were in the "
    "training set, so the model is recalling those labels rather than predicting them.[/]",
  ))


def _note_contradicted_labels(rows, output_dir):
  """Say how many compounds the input called both active and inactive.

  The classification counterpart of the regression noise floor, and it bites differently.
  Deduplication keeps one label per compound — the rounded median — so a model that predicts that
  label is scored as correct and the measured accuracy is not capped at all. What is wrong sits
  underneath the score: the kept label is arbitrary, so for those compounds the model is fitted on
  a coin flip and then graded against the same coin flip. Neither the training signal nor the
  score on them carries information.
  """
  from zairachem.setup.prep.files import read_replicates

  summary = read_replicates(output_dir)
  contradicted = summary.get("disagreeing")
  repeated = summary.get("n_repeated") or 0
  if not contradicted:
    return
  rows.append((
    "",
    f"[yellow]{contradicted:,} of {repeated:,} repeated compounds[/] [dim]were given both labels "
    "in the input, so the one kept for each is arbitrary: there the model is fitted on a coin "
    "flip and scored against it.[/]",
  ))


def _note_held_out_skill(rows, output_dir, task, pooled):
  """Append the scaffold held-out skill, and a caution when the headline is well above it.

  The Performance row above is out-of-fold: lazy-qsar's inner cross-validation, whose folds are
  drawn at random. On a set built from congeneric series — which is most medicinal-chemistry data
  — a random fold leaves close analogues of every test compound in training, so the number it
  reports is what the model does on more of the same chemistry, not on new chemistry. A run given
  ``--evaluate`` has already measured the second thing by holding out whole scaffolds, and the
  answer sits in a CSV while the optimistic number is what the panel says.

  Only the scaffold strategy is reported. A random held-out fold asks the same question as the
  out-of-fold estimate, so repeating it would add a number without adding information.

  The drop that earns a caution is ``PRUNE_GAP``, the pooler's. That constant was written for a
  different pair of numbers — train against out-of-fold, where it decides a descriptor has stopped
  being trustworthy — but it answers the same question in the same units: how much skill can fall
  away between an optimistic estimate and an honest one before the optimistic one stops describing
  the model. Keeping the two together means a run cannot call a gap acceptable in one place and
  disqualifying in the other.
  """
  from zairachem.pool.reliability_pooler.pooler import PRUNE_GAP

  summary_path = os.path.join(output_dir, "report", "holdout_summary.json")
  try:
    with open(summary_path) as f:
      scaffold = (json.load(f).get("strategies") or {}).get("scaffold") or {}
  except Exception:
    return
  key, label = ("auroc", "AUROC") if task == "classification" else ("r2", "R\u00b2")
  held = _finite_or_none((scaffold.get(key) or {}).get("mean"))
  n_folds = scaffold.get("n_folds")
  if held is None or not n_folds:
    return
  rows.append((
    "Held out",
    f"{label} [bold green]{held:.3f}[/]  [dim](scaffold, {_plurals(n_folds, 'fold')})[/]",
  ))
  headline = _finite_or_none(pooled.get(key))
  gap = PRUNE_GAP.get(task)
  if headline is None or gap is None or headline - held <= gap:
    return
  rows.append((
    "",
    f"[yellow]Weaker on new chemistry[/] [dim]— {label} falls to {_distinct(held, headline)} "
    "when whole scaffolds are held out, so on compounds unlike the training set this model is "
    "worse than the headline above says.[/]",
  ))


#: A descriptor is called out when it took at least this share of the time spent training, …
_EXPENSIVE_TIME_SHARE = 0.6
#: … and that share is at least this many times the share of the pooled weight it earned. A plain
#: weight floor will not do: with two descriptors one of them almost always takes over half the
#: time, and being the slower of two is not by itself worth a warning.
_EXPENSIVE_DISPROPORTION = 3.0


def _human_seconds(seconds):
  """``seconds`` as a short duration: ``"43m"``, ``"2.4h"``, ``"18s"``."""
  if seconds >= 3600:
    return f"{seconds / 3600:.1f}h"
  if seconds >= 60:
    return f"{seconds / 60:.0f}m"
  return f"{seconds:.0f}s"


def _note_expensive_descriptors(rows, output_dir):
  """Append a note when one descriptor took most of the fit and contributed little to the model.

  Both halves are measured, not assumed: the time is what this machine actually spent on each
  descriptor (``fit_seconds`` in its cv report) and the contribution is the weight the pooling
  gave it. Which descriptors are slow is a property of the model, the hardware and the dataset
  together, so there is nothing to tabulate in advance — a run can only find it out by doing it,
  and the next run is the one that can act on it.

  Pre-screening cannot catch this. It ranks descriptors on a shallow forest that costs about the
  same for all of them, so two descriptors that differ by an order of magnitude in training cost
  look equally cheap to it and it keeps both.
  """
  rel = _reliability_summary(output_dir) or {}
  weights = rel.get("mean_weights") or {}
  times = {
    s["descriptor"]: t
    for s in _cv_stats(output_dir)
    if s.get("descriptor") and (t := _finite_or_none(s.get("fit_seconds"))) is not None
  }
  total_time = sum(times.values())
  total_weight = sum(v for v in weights.values() if _finite_or_none(v) is not None)
  if len(times) < 2 or total_time <= 0 or total_weight <= 0:
    return
  eos_id = max(times, key=lambda e: times[e])
  time_share = times[eos_id] / total_time
  weight = _finite_or_none(weights.get(eos_id))
  if weight is None or time_share < _EXPENSIVE_TIME_SHARE:
    return
  weight_share = weight / total_weight
  if time_share < _EXPENSIVE_DISPROPORTION * weight_share:
    return
  rows.append((
    "",
    f"[yellow]Most of the fit for little of the model[/] [dim]— {eos_id} took "
    f"{_human_seconds(times[eos_id])} of the {_human_seconds(total_time)} spent training "
    f"({time_share:.0%}) for {weight_share:.0%} of the pooled weight; leaving it out of `-f` "
    "would save most of the next fit.[/]",
  ))


def _warn_if_no_signal(rows, task, skill, r2=None):
  """Append at most one caution row when the pooled model is not worth shipping.

  Two ways to fail, and the first that applies is the one reported. Its held-out skill is below
  the floor the pooler prunes individual descriptors on: both measures are rank-based (AUROC and
  Spearman's rho), so the claim is precisely that the model does not order compounds better than
  chance. Or -- regression only -- its R2 is at or below zero, which is not a threshold anyone
  chose but the point where predicting the training mean for everything would have done as well.

  Said plainly here because the number alone does not stop anyone shipping the model.

  Parameters
  ----------
  rows : list
      Summary-panel rows, appended to in place.
  task : str
      ``"classification"`` or ``"regression"``.
  skill : float or None
      The pooled model's held-out AUROC or Spearman rho.
  r2 : float or None, optional
      The pooled model's R2 (regression runs only).
  """

  def caution(text):
    rows.append(("", f"[yellow]No usable signal[/] [dim]— {text}[/]"))

  value, floor = _finite_or_none(skill), SKILL_FLOOR.get(task)
  if value is not None and floor is not None and value < floor:
    measure, orders = (
      ("AUROC", "separate the classes")
      if task == "classification"
      else ("Spearman rho", "order compounds")
    )
    caution(
      f"held-out {measure} {_distinct(value, floor)} is below {_distinct(floor, value)}: "
      f"this model does not {orders} better than chance."
    )
    return
  # A model can order compounds acceptably and still predict useless values.
  explained = _finite_or_none(r2)
  if explained is not None and explained <= 0:
    caution(
      f"R² {explained:.2f}: these predicted values are no better than the training average, "
      "whatever their ordering."
    )


def _pooled_row(params, m):
  """The ``pooled`` metrics detail row from a performance-table row."""

  def fmt(k):
    try:
      return f"{float(m[k]):.3f}"
    except Exception:
      return "—"

  if params.get("task") == "classification":
    return (
      "pooled",
      f"AUROC {fmt('auroc')} [dim]·[/] MCC {fmt('mcc')} [dim]at the {m.get('cutoff', '?')} "
      "cutoff[/]",
    )
  return ("pooled", f"R² {fmt('r2')} [dim]·[/] RMSE {fmt('rmse')} [dim]·[/] rho {fmt('spearman')}")


def _applicability_rows(rel, ad):
  """Out-of-domain detail rows from the reliability summary's applicability block."""
  rows = []
  n_ood = ad.get("n_out_of_domain", 0)
  n_tot = rel.get("n_compounds", 0) or 0
  pct = 100.0 * ad.get("frac_out_of_domain", 0.0)
  col = "yellow" if n_ood else "dim"
  rows.append((
    "applicability",
    f"[{col}]{n_ood:,}[/]/{n_tot:,} compounds out-of-domain [dim]({pct:.0f}%)[/]",
  ))
  flagged = [
    f"{name} {cnt:,}"
    for name, cnt in sorted(
      ad.get("per_descriptor_out_of_domain", {}).items(), key=lambda kv: -kv[1]
    )
    if cnt
  ][:4]
  if flagged:
    rows.append(("ood by descriptor", "[dim]" + "  ".join(flagged) + "[/]"))
  return rows


def _reliability_rows(d, params, rel):
  """Detail rows for the pool step when the per-sample reliability pooler ran."""
  rows = []
  # Per-sample reliability pooler: show how descriptors were combined. At predict time (no
  # labels → no pooled metrics) these rows are the user's window into what the step did.
  rows.append(("method", "lazy-qsar ensemble [dim]· per-sample weighted (logit space)[/]"))
  rows.append((
    "weighting",
    f"{rel.get('tier', '?')} [dim]· {_plurals(rel.get('n_descriptors', 0), 'descriptor')}[/]",
  ))
  if rel.get("pruned"):
    rows.append(("left out", f"[dim]{'  '.join(rel['pruned'])} (weak out-of-fold skill)[/]"))
  ad = rel.get("applicability")
  if ad:
    rows.extend(_applicability_rows(rel, ad))
  mw = rel.get("mean_weights", {})
  if mw:
    top = sorted(mw.items(), key=lambda kv: -kv[1])[:4]
    rows.append((
      "mean weights",
      "  ".join(f"{name} [bold]{w:.2f}[/]" for name, w in top),
    ))
  m = _pooled_metrics(d)
  if m:
    rows.append(_pooled_row(params, m))
  return rows


def _pool_rows(d, params):
  """Detail rows for the pool step."""
  rel = _reliability_summary(d)
  if rel:
    return _reliability_rows(d, params, rel)
  rows = []
  algos = _estimator_algorithms(d)
  rows.append(("consensus", f"{_plurals(len(algos), 'descriptor model')}" if algos else "pooled"))
  m = _pooled_metrics(d)
  if m:
    rows.append(_pooled_row(params, m))
  return rows


def _report_rows(d):
  """Detail rows for the report step."""
  from zairachem.base.vars import REPORT_SUBFOLDER, RESULTS_SUBFOLDER

  rows = []
  cats = _plots_by_category(d)
  if cats:
    total = sum(cats.values())
    breakdown = "  ".join(f"{v} {k}" for k, v in sorted(cats.items(), key=lambda kv: -kv[1]))
    rows.append(("plots", f"{total} [dim]· {breakdown}[/]"))
  outs = [f for f in ("output.csv",) if os.path.exists(os.path.join(d, RESULTS_SUBFOLDER, f))]
  if outs:
    rows.append(("outputs", "  ".join(outs)))
  html_path = os.path.join(d, REPORT_SUBFOLDER, "report.html")
  if os.path.exists(html_path):
    rows.append(("html", _collapse(html_path)))
  return rows


def _finish_rows(d):
  """Detail rows for the finish step."""
  from zairachem.base.vars import RESULTS_SUBFOLDER

  rows = [("model", _collapse(d))]
  size = _dir_size_mb(d)
  if size is not None:
    rows.append(("size", f"{size:,.1f} MB"))
  present = [f for f in ("output.csv",) if os.path.exists(os.path.join(d, RESULTS_SUBFOLDER, f))]
  from zairachem.base.vars import REPORT_SUBFOLDER

  if os.path.exists(os.path.join(d, REPORT_SUBFOLDER, "report.html")):
    present.append("report.html")
  if present:
    rows.append(("artifacts", "  ".join(present)))
  return rows


def _detail_rows(key, output_dir=None):
  """Rich (label, value) detail lines per step, read from the run's artifacts; [] if unavailable."""
  d = _resolve_output_dir(output_dir)
  if not d:
    return []
  params = _load_params(d)
  if key in _LIVE_TABLE_STEPS:
    # These steps keep their live table on screen as the record — no separate detail block.
    return []
  if key == "setup":
    return _setup_rows(d, params)
  if key == "pool":
    return _pool_rows(d, params)
  if key == "report":
    return _report_rows(d)
  if key == "finish":
    return _finish_rows(d)
  return []


def final_summary_panel(output_dir=None):
  """Render the closing run-summary panel: headline metrics + output location."""
  # The live checklist is transient; stop it first so the panel lands cleanly below the records.
  # Imported lazily (the tracker module imports this one) to keep the module load acyclic.
  from zairachem.base.utils.pipeline_tracker import tracker

  tracker.stop()
  d = _resolve_output_dir(output_dir)
  if not d:
    return
  params = _load_params(d)
  task = params.get("task", "?")
  n = _n_compounds(d)

  rows = [("Output", f"[dim]{_collapse(d)}[/]"), ("Task", f"[magenta]{task}[/]")]
  if n is not None:
    rows.append(("Compounds", f"[bold]{n:,}[/]"))
  if _is_predict(d):
    _note_training_overlap(rows, d)
  # List the requested featurizers, distinguishing the ones the pooled model actually uses (green)
  # from those dropped by --max-descriptors pre-screening (dimmed). effective_descriptors reads
  # metadata/selected_eos.json when screening ran, else falls back to all trained descriptors.
  all_desc = params.get("featurizer_ids", [])
  try:
    from zairachem.base.utils.descriptors import effective_descriptors

    effective = effective_descriptors(d)
  except Exception:
    effective = list(all_desc)
  used = set(effective)
  if _is_predict(d):
    # Predict only ever uses the selected descriptors — the pre-screened-out ones are never computed
    # for these molecules, so omit them entirely (in order) rather than listing them as unused.
    desc_label = "  ".join(f"[green]{eos}[/]" for eos in (effective or all_desc))
  elif used and 0 < len(used) < len(all_desc):
    # Fit: keep the screened-out descriptors visible (dimmed) — seeing what was dropped is useful here.
    cells = " ".join(
      f"[green]{eos}[/]" if eos in used else f"[dim strike]{eos}[/]" for eos in all_desc
    )
    desc_label = f"{cells}   [dim]({len(used)}/{len(all_desc)} used · rest pre-screened)[/]"
  else:
    desc_label = "  ".join(f"[green]{eos}[/]" for eos in all_desc)
  rows.append(("Descriptors", desc_label))

  m = _pooled_metrics(d)
  if m:

    def fmt(key):
      try:
        return f"{float(m[key]):.3f}"
      except Exception:
        return "—"

    if task == "classification":
      where = f"at the {m.get('cutoff', '?')} cutoff"
      if not _is_predict(d):
        where += ", out-of-fold"
      rows.append((
        "Performance",
        f"AUROC [bold green]{fmt('auroc')}[/] · MCC [bold green]{fmt('mcc')}[/]  [dim]({where})[/]",
      ))
      _note_contradicted_labels(rows, d)
      _warn_if_no_signal(rows, task, m.get("auroc"))
    else:
      # RMSE alongside R2: the first is unitless, the second is in the target's own units --
      # and the assay's own spread, where the input measured anything twice, is what says whether
      # that number is small.
      where = "out-of-fold" if not _is_predict(d) else "on this set"
      floor = _replicate_sd(d)
      against = f" · repeats differ by {floor:.3f}" if floor is not None else ""
      rows.append((
        "Performance",
        f"R² [bold green]{fmt('r2')}[/] · RMSE [bold green]{fmt('rmse')}[/]  "
        f"[dim]({where}{against})[/]",
      ))
      _note_if_at_the_noise_floor(rows, m.get("rmse"), floor)
      _warn_if_no_signal(rows, task, m.get("spearman"), r2=m.get("r2"))
    _note_held_out_skill(rows, d, task, m)
  if not _is_predict(d):
    _note_expensive_descriptors(rows, d)

  plots = _count_plots(d)
  from zairachem.base.vars import OUTPUT_FILENAME, REPORT_SUBFOLDER

  # Paths below are relative to Output, which is shown once in full.
  if os.path.exists(os.path.join(d, OUTPUT_FILENAME)):
    rows.append(("Results", f"[bold]{OUTPUT_FILENAME}[/]"))
  html_path = os.path.join(d, REPORT_SUBFOLDER, "report.html")
  if os.path.exists(html_path):
    suffix = f"  [dim]({_plurals(plots, 'plot')})[/]" if plots else ""
    rows.append(("Report", f"[bold]{REPORT_SUBFOLDER}/report.html[/]{suffix}"))
  elif plots:
    rows.append(("Report", f"{_plurals(plots, 'plot')} in [bold]{REPORT_SUBFOLDER}/[/]"))

  title = "ZairaChem · predictions ready" if _is_predict(d) else "ZairaChem · model ready"
  summary_panel(title, rows, border_style="bright_green", icon="✓")
