"""Per-step result summaries and detail blocks for the ZairaChem CLI.

Reads the run's artifacts (parameters, data.csv, estimator/pool outputs) to produce the one-line
``✓`` summary per step (:data:`SUMMARIES`), the calm detail rows printed under each step banner
(:func:`_detail_rows`) and the closing run-summary panel (:func:`final_summary_panel`). Pure
read-from-disk + formatting; depends only on the console renderer, not on the live widgets or the
tracker (which it touches lazily, only inside :func:`final_summary_panel`).

Re-exported from :mod:`zairachem.base.utils.progress` for backwards compatibility.
"""

import json
import os

from zairachem.base.utils.console import summary_panel

#: Steps that render a persistent live per-item table (see :class:`..live.LiveTableMonitor`). That
#: table stays on screen as the step's record, so these steps intentionally print no separate detail
#: block (``_detail_rows`` returns [] for them).
_LIVE_TABLE_STEPS = {"describe", "projections", "treat", "estimate"}


def _resolve_output_dir(output_dir=None):
  if output_dir:
    return output_dir
  from zairachem.base.vars import BASE_DIR, SESSION_FILE

  try:
    with open(os.path.join(BASE_DIR, SESSION_FILE)) as f:
      return json.load(f)["output_dir"]
  except Exception:
    return None


def _is_predict():
  """True if the active session is a prediction run (so step text can say 'apply', not 'train')."""
  from zairachem.base.vars import BASE_DIR, SESSION_FILE

  try:
    with open(os.path.join(BASE_DIR, SESSION_FILE)) as f:
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


def _descriptor_feature_width(output_dir, featurizer_ids):
  """Total descriptor columns across all featurizers (best-effort; None if unreadable)."""
  from zairachem.base.vars import DESCRIPTORS_SUBFOLDER

  total = 0
  try:
    import h5py

    for eos in featurizer_ids:
      base = os.path.join(output_dir, DESCRIPTORS_SUBFOLDER, eos)
      h5 = os.path.join(base, "raw.h5")
      if not os.path.exists(h5):
        chunks = os.path.join(base, "raw_chunks", "chunk_0000.h5")
        h5 = chunks if os.path.exists(chunks) else None
      if not h5:
        return None
      with h5py.File(h5, "r") as f:
        if "Features" in f:
          total += int(f["Features"].shape[0])
        elif "Values" in f:
          total += int(f["Values"].shape[1])
        else:
          return None
    return total or None
  except Exception:
    return None


def _estimator_algorithms(output_dir):
  """Algorithm names (from exported .onnx bundles), excluding preprocessor/pooler."""
  from zairachem.base.vars import ESTIMATORS_SUBFOLDER

  algos = set()
  try:
    for root, _dirs, files in os.walk(os.path.join(output_dir, ESTIMATORS_SUBFOLDER)):
      for f in files:
        if f.endswith(".onnx"):
          name = f[:-5]
          if name not in ("preprocessor", "pooler"):
            algos.add(name)
  except Exception:
    pass
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
  try:
    import csv

    from zairachem.base.vars import PERFORMANCE_TABLE_FILENAME, RESULTS_SUBFOLDER

    with open(os.path.join(output_dir, RESULTS_SUBFOLDER, PERFORMANCE_TABLE_FILENAME)) as f:
      for row in csv.DictReader(f):
        if row.get("model") == "pooled":
          return row
  except Exception:
    pass
  return None


# --- One-line summaries (the ✓ result line per step) -------------------------------------------


def summarize_setup(output_dir=None):
  d = _resolve_output_dir(output_dir)
  if not d:
    return ""
  params = _load_params(d)
  parts = []
  n = _n_compounds(d)
  if n is not None:
    parts.append(f"{n:,} compounds")
  parts.append(params.get("task", "?"))
  parts.append(_plurals(len(params.get("featurizer_ids", []) or []), "descriptor"))
  return " · ".join(parts)


def summarize_describe(output_dir=None):
  d = _resolve_output_dir(output_dir)
  if not d:
    return ""
  params = _load_params(d)
  featurizers = params.get("featurizer_ids", []) or []
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
  d = _resolve_output_dir(output_dir)
  if not d:
    return ""
  params = _load_params(d)
  extra = params.get("projection_ids") or []
  return "MW/LogP" + (f" + {_plurals(len(extra), 'model')}" if extra else " (built-in)")


def summarize_treat(output_dir=None):
  d = _resolve_output_dir(output_dir)
  if not d:
    return ""
  params = _load_params(d)
  n = _n_compounds(d)
  width = _descriptor_feature_width(d, params.get("featurizer_ids", []) or [])
  return f"{n:,} × {width:,} matrix" if (n is not None and width is not None) else "matrix imputed"


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
  """(mean_oof, best_descriptor, best_oof) or None from a list of cv reports."""
  aucs = [s["oof_auc"] for s in cv if s.get("oof_auc") is not None]
  if not aucs:
    return None
  best = max(cv, key=lambda s: s.get("oof_auc") if s.get("oof_auc") is not None else -1)
  return sum(aucs) / len(aucs), best.get("descriptor"), best.get("oof_auc")


def summarize_estimate(output_dir=None):
  d = _resolve_output_dir(output_dir)
  if not d:
    return ""
  if _is_predict():
    # Predict applies the trained models — there are no freshly trained estimators or CV stats here.
    n = len(_load_params(d).get("featurizer_ids", []) or [])
    return _plurals(n, "descriptor") + " scored" if n else "models applied"
  algos = _estimator_algorithms(d)
  base = _plurals(len(algos), "algorithm") + " trained" if algos else "estimators trained"
  head = _cv_headline(_cv_stats(d))
  if head:
    mean, best, best_auc = head
    base += f" · CV AUROC {mean:.2f} (best {best} {best_auc:.2f})"
  return base


def summarize_pool(output_dir=None):
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


def summarize_holdout(output_dir=None):
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
  scaffold = (summary.get("strategies", {}).get("scaffold", {}).get("auroc", {}) or {}).get("mean")
  parts = [_plurals(n, "fold")]
  if scaffold is not None:
    parts.append(f"scaffold AUROC {scaffold:.2f}")
  return " · ".join(parts)


def summarize_report(output_dir=None):
  d = _resolve_output_dir(output_dir)
  if not d:
    return ""
  n = _count_plots(d)
  return f"{_plurals(n, 'plot')} · HTML report" if n else "report written"


def summarize_finish(output_dir=None):
  d = _resolve_output_dir(output_dir)
  if not d:
    return ""
  return f"model ready · {_collapse(d)}"


SUMMARIES = {
  "setup": summarize_setup,
  "describe": summarize_describe,
  "projections": summarize_projections,
  "treat": summarize_treat,
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
  except Exception:
    pass
  return None


def _provenance(output_dir):
  """The run's data-provenance dict (per-model n_total/n_from_project/n_computed), or {}."""
  try:
    from zairachem.base.utils.isaura_report import _load_provenance

    return _load_provenance(output_dir)
  except Exception:
    return {}


def _model_width(output_dir, eos):
  """Descriptor column count for a single featurizer model, or None."""
  from zairachem.base.vars import DESCRIPTORS_SUBFOLDER

  try:
    import h5py

    base = os.path.join(output_dir, DESCRIPTORS_SUBFOLDER, eos)
    h5 = os.path.join(base, "raw.h5")
    if not os.path.exists(h5):
      chunk = os.path.join(base, "raw_chunks", "chunk_0000.h5")
      h5 = chunk if os.path.exists(chunk) else None
    if not h5:
      return None
    with h5py.File(h5, "r") as f:
      if "Features" in f:
        return int(f["Features"].shape[0])
      if "Values" in f:
        return int(f["Values"].shape[1])
  except Exception:
    pass
  return None


def _treat_widths(output_dir, featurizer_ids):
  """(columns_in, columns_out) summed across featurizers — raw H5 width vs treated info width.

  Returns None if nothing readable. Derived entirely from artifacts (no extra persistence).
  """
  from zairachem.base.vars import DESCRIPTORS_SUBFOLDER, TREATED_DESC_FILENAME

  ci = co = 0
  ok = False
  for eos in featurizer_ids:
    rin = _model_width(output_dir, eos)
    info = os.path.join(
      output_dir, DESCRIPTORS_SUBFOLDER, eos, TREATED_DESC_FILENAME.replace(".h5", ".json")
    )
    try:
      with open(info) as f:
        cout = int(json.load(f).get("features"))
    except Exception:
      cout = None
    if rin is not None and cout is not None:
      ci += rin
      co += cout
      ok = True
  return (ci, co) if ok else None


def _plots_by_category(output_dir):
  """Group report PNGs into coarse categories by filename keyword. Returns {category: count}."""
  from zairachem.base.vars import REPORT_SUBFOLDER

  buckets = {
    "roc": "ROC",
    "auroc": "ROC",
    "calib": "calibration",
    "dist": "distributions",
    "violin": "distributions",
    "strip": "distributions",
    "proj": "projections",
    "cv": "cross-validation",
    "confusion": "confusion",
    "r2": "regression",
    "obs": "regression",
    "hist": "histograms",
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


def _fmt_timing(timing, top=4):
  """Compact 'phase Xs · phase Ys' string from a cv_report timing dict (largest phases first)."""
  if not isinstance(timing, dict):
    return ""
  scalars = [(k, v) for k, v in timing.items() if isinstance(v, (int, float))]
  if not scalars:
    return ""
  items = sorted(scalars, key=lambda kv: -kv[1])[:top]
  return " · ".join(f"{k} {v:.1f}s" for k, v in items)


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
    rows = []
    n = _n_compounds(d)
    if params.get("task") == "classification" and _active_inactive(d):
      a, i = _active_inactive(d)
      rows.append(("compounds", f"{n:,} [dim]·[/] {a:,} active [dim]·[/] {i:,} inactive"))
    elif n is not None:
      rows.append(("compounds", f"{n:,}"))
    feats = params.get("featurizer_ids", []) or []
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
  if key == "pool":
    rows = []
    rel = _reliability_summary(d)
    if rel:
      # Per-sample reliability pooler: show how descriptors were combined. At predict time (no
      # labels → no pooled metrics) these rows are the user's window into what the step did.
      rows.append(("method", "reliability [dim]· per-sample weighted (logit space)[/]"))
      rows.append((
        "weighting",
        f"{rel.get('tier', '?')} [dim]· {_plurals(rel.get('n_descriptors', 0), 'descriptor')}[/]",
      ))
      ad = rel.get("applicability")
      if ad:
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
      mw = rel.get("mean_weights", {})
      if mw:
        top = sorted(mw.items(), key=lambda kv: -kv[1])[:4]
        rows.append((
          "mean weights",
          "  ".join(f"{name} [bold]{w:.2f}[/]" for name, w in top),
        ))
      m = _pooled_metrics(d)
      if m:

        def fmtr(k):
          try:
            return f"{float(m[k]):.3f}"
          except Exception:
            return "—"

        if params.get("task") == "classification":
          rows.append((
            "pooled",
            f"AUROC {fmtr('auroc')} [dim]·[/] acc {fmtr('accuracy')} [dim]·[/] MCC {fmtr('mcc')}",
          ))
        else:
          rows.append(("pooled", f"R² {fmtr('r2')}"))
      return rows
    algos = _estimator_algorithms(d)
    rows.append(("consensus", f"{_plurals(len(algos), 'descriptor model')}" if algos else "pooled"))
    m = _pooled_metrics(d)
    if m:

      def fmt(k):
        try:
          return f"{float(m[k]):.3f}"
        except Exception:
          return "—"

      if params.get("task") == "classification":
        rows.append((
          "pooled",
          f"AUROC {fmt('auroc')} [dim]·[/] acc {fmt('accuracy')} [dim]·[/] MCC {fmt('mcc')}",
        ))
      else:
        rows.append(("pooled", f"R² {fmt('r2')}"))
    return rows
  if key == "report":
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
  if key == "finish":
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
  rows.append((
    "Descriptors",
    "  ".join(f"[green]{m}[/]" for m in params.get("featurizer_ids", [])),
  ))

  m = _pooled_metrics(d)
  if m:

    def fmt(key):
      try:
        return f"{float(m[key]):.3f}"
      except Exception:
        return "—"

    if task == "classification":
      rows.append((
        "Performance",
        f"AUROC [bold green]{fmt('auroc')}[/] · accuracy [bold green]{fmt('accuracy')}[/] · MCC [bold green]{fmt('mcc')}[/]",
      ))
    else:
      rows.append(("Performance", f"R² [bold green]{fmt('r2')}[/]"))

  plots = _count_plots(d)
  from zairachem.base.vars import REPORT_SUBFOLDER

  html_path = os.path.join(d, REPORT_SUBFOLDER, "report.html")
  if os.path.exists(html_path):
    suffix = f"  [dim]({_plurals(plots, 'plot')})[/]" if plots else ""
    rows.append(("Report", f"[dim]{_collapse(html_path)}[/]{suffix}"))
  elif plots:
    rows.append(("Report", f"{_plurals(plots, 'plot')} in [dim]{_collapse(d)}/report[/]"))

  summary_panel("ZairaChem · model ready", rows, border_style="bright_green", icon="✓")
