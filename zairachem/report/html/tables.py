"""HTML table builders: hit list, validation metrics, model table and cross-validation tables."""

import csv
import html
import json
import os

from zairachem.base.utils.logging import logger
from zairachem.base.vars import DEFAULT_CUTOFF, GITHUB_ORG, OUTPUT_TABLE_FILENAME
from zairachem.report import colors as _colors
from zairachem.report.cv import skill_keys
from zairachem.report.html.catalog import _BADGE_DISCARDED, _BADGE_KEPT, _VALIDATION_SCHEMAS


def _fmt_num(v):
  try:
    f = float(v)
    return f"{f:.3f}" if abs(f) < 1000 else f"{int(f):,}"
  except Exception:
    return html.escape(str(v))


def _hitlist_table_html(report_dir, top_n=25):
  """Top predictions by score — the actionable deliverable of a predict report (SMILES + scores).

  Reads the already-written ``output_table.csv`` and ranks by the pooled prediction: the probability
  for classification, shown with its binary call, or the predicted value for regression, shown with
  its prediction interval (a regression run has no call to make). Both carry the per-descriptor
  spread and, when present, the similarity to the training set. No structure depictions.
  """
  import numpy as _np
  import pandas as _pd

  path = os.path.join(report_dir, OUTPUT_TABLE_FILENAME)
  if not os.path.exists(path):
    return ""
  try:
    df = _pd.read_csv(path)
  except Exception:
    return ""
  score_col = "proba" if "proba" in df.columns else "value"
  if score_col not in df.columns or len(df) == 0:
    return ""
  is_clf = score_col == "proba"
  total = len(df)
  prefix = "proba_" if is_clf else "value_"
  desc_cols = [c for c in df.columns if c.startswith(prefix) and c != "proba_raw"]
  has_sim = "similarity" in df.columns
  has_interval = not is_clf and {"lower", "upper"} <= set(df.columns)
  top = df.sort_values(score_col, ascending=False).head(top_n)

  headers = ["#", "SMILES", "Score" if is_clf else "Value"]
  if is_clf:
    headers.append("Call")
  elif has_interval:
    headers.append("Interval")
  if desc_cols or "spread" in df.columns:
    headers.append("Spread")
  if has_sim:
    headers.append("Similarity")
  head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)

  body = []
  for i, (_, r) in enumerate(top.iterrows(), start=1):
    cells = [
      f"<td>{i}</td>",
      f"<td class='smiles'>{html.escape(str(r.get('input_smiles', '')))}</td>",
      f"<td>{_fmt_num(r[score_col])}</td>",
    ]
    if is_clf:
      try:
        call = int(r["call"]) if "call" in top.columns else int(float(r["proba"]) >= DEFAULT_CUTOFF)
      except Exception:
        call = int(float(r.get("proba", 0) or 0) >= DEFAULT_CUTOFF)
      cells.append(f"<td>{'active' if call == 1 else 'inactive'}</td>")
    elif has_interval:
      cells.append(f"<td>{_fmt_num(r['lower'])} – {_fmt_num(r['upper'])}</td>")
    if "spread" in top.columns:
      cells.append(f"<td>{_fmt_num(r['spread'])}</td>")
    elif desc_cols:
      spread = float(_np.nanstd(_np.asarray([r[c] for c in desc_cols], dtype=float)))
      cells.append(f"<td>{_fmt_num(spread)}</td>")
    if has_sim:
      cells.append(f"<td>{_fmt_num(r['similarity'])}</td>")
    body.append(f"<tr>{''.join(cells)}</tr>")

  ranked_by = "predicted probability" if is_clf else "predicted value"
  note = (
    f"<p style='color:#6e7781;font-size:13px'>Top {len(top)} of {total:,} molecules "
    f"by {ranked_by}.</p>"
  )
  return (
    note + "<div class='table-wrap'><table class='metrics'>"
    f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
  )


def _mean_std(values):
  import statistics

  vals = [v for v in values if v is not None]
  if not vals:
    return None, None
  mean = sum(vals) / len(vals)
  std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
  return mean, std


def _validation_table_html(report_dir):
  """Per-schema mean±std held-out AUROC/AUPR table from ``report/validation_table.csv``, or ''."""
  try:
    with open(os.path.join(report_dir, "validation_table.csv")) as f:
      rows = list(csv.DictReader(f))
  except Exception:
    return ""
  if not rows:
    return ""

  def _num(r, k):
    try:
      return float(r[k])
    except (KeyError, ValueError, TypeError):
      return None

  by_strategy = {}
  for r in rows:
    by_strategy.setdefault(r.get("strategy"), []).append(r)
  ordered = [(k, lbl) for k, lbl in _VALIDATION_SCHEMAS if k in by_strategy]
  ordered += [(k, k) for k in by_strategy if k not in {s for s, _ in _VALIDATION_SCHEMAS}]
  if not ordered:
    return ""

  def _ratio(r):
    """Test-set actives:inactives for one fold (``num_1 / num_0``), or None."""
    a, i = _num(r, "num_1"), _num(r, "num_0")
    if a is None or not i:
      return None
    return a / i

  # The folds carry whichever metrics their task produced (see holdout/io.py). Classification also
  # reports the test set's class balance, which regression has no analogue of.
  is_clf = "auroc" in rows[0]
  metrics = (
    [("auroc", "AUROC"), ("aupr", "AUPR")]
    if is_clf
    else [
      ("r2", "R²"),
      ("rmse", "RMSE"),
      ("spearman", "Spearman ρ"),
    ]
  )
  body = []
  for strat, label in ordered:
    srows = by_strategy[strat]
    cells = []
    if is_clf:
      ratio_m, _ = _mean_std([_ratio(r) for r in srows])
      cells.append(f"{ratio_m:.2f}" if ratio_m is not None else "—")
    for key, _lbl in metrics:
      m, sd = _mean_std([_num(r, key) for r in srows])
      cells.append(f"{m:.3f} ± {sd:.3f}" if m is not None else "—")
    body.append(
      f"<tr><td>{html.escape(label)}</td><td>{len(srows)}</td>"
      + "".join(f"<td>{c}</td>" for c in cells)
      + "</tr>"
    )
  head = "<th>Split schema</th><th>Folds</th>"
  if is_clf:
    head += "<th>Active:inactive (test)</th>"
  head += "".join(f"<th>{lbl} (mean ± std)</th>" for _key, lbl in metrics)
  note = ""
  try:
    with open(os.path.join(report_dir, "holdout_summary.json")) as f:
      summary = json.load(f)
    failed = int(summary.get("n_folds_failed") or 0)
    if failed:
      total = summary.get("n_folds_defined", len(rows) + failed)
      note = (
        f"<p style='color:#6e7781;font-size:13px'>{failed} of {total} held-out folds failed and are not included "
        "above.</p>"
      )
  except (OSError, ValueError, TypeError) as e:
    logger.debug(f"[report] no held-out fold note: {e}")
  return (
    "<div class='table-wrap'><table class='metrics'>"
    f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>{note}"
  )


def _model_table_html(ids, versions, titles, selected=None):
  """A table of Ersilia models: monospace id linked to the hub, version pill, human title.

  When ``selected`` is provided (a set of the descriptors kept by --max-descriptors screening), a
  Status column is added marking each model ``kept`` (green) or ``discarded`` (red). Passed only for
  featurizers on a screened run; ``None`` (no column) otherwise.
  """
  if not ids:
    return ""
  show_status = selected is not None
  rows = []
  for mid in ids:
    href = f"https://github.com/{GITHUB_ORG}/{html.escape(mid)}"
    ver = versions.get(mid)
    ver_html = f"<span class='ver'>{html.escape(str(ver))}</span>" if ver else "—"
    title = titles.get(mid)
    title_html = html.escape(title) if title else "—"
    status = ""
    if show_status:
      badge = _BADGE_KEPT if mid in selected else _BADGE_DISCARDED
      status = f"<td>{badge}</td>"
    cls = "" if (not show_status or mid in selected) else " style='opacity:.6'"
    rows.append(
      f"<tr{cls}><td><a class='model' href='{href}' target='_blank'>{html.escape(mid)} ↗</a></td>"
      f"<td>{ver_html}</td><td class='title'>{title_html}</td>{status}</tr>"
    )
  status_head = "<th>Status</th>" if show_status else ""
  return (
    "<div class='table-wrap'><table class='cfg-models'>"
    f"<thead><tr><th>Model</th><th>Version</th><th>Title</th>{status_head}</tr></thead>"
    f"<tbody>{''.join(rows)}</tbody></table></div>"
  )


def _cv_summary(cv_stats):
  key, _train_key, label = skill_keys(cv_stats)
  skills = [s[key] for s in cv_stats if s.get(key) is not None]
  if not skills:
    return {}
  gaps = [s["overfit_gap"] for s in cv_stats if s.get("overfit_gap") is not None]
  best = cv_stats[0]
  return {
    "n_descriptors": len(cv_stats),
    "skill_label": label,
    "mean_skill": sum(skills) / len(skills),
    "best_skill": best.get(key),
    "best_descriptor": best.get("descriptor"),
    "mean_overfit_gap": (sum(gaps) / len(gaps)) if gaps else None,
  }


def _cv_table_html(cv_stats):
  if not cv_stats:
    return ""
  key, _train_key, label = skill_keys(cv_stats)
  if key == "oof_auc":
    cols = [
      ("descriptor", "Descriptor"),
      ("oof_auc", "Inner CV AUROC"),
      ("aupr", "Inner CV AUPRC"),
      ("train_auc", "Train AUROC"),
      ("overfit_gap", "Overfit gap"),
      ("decision_cutoff_proba", "Balanced cutoff"),
    ]
  else:
    # Regression has no second ranking metric and no cutoff; the overfit gap means the same thing.
    cols = [
      ("descriptor", "Descriptor"),
      ("oof_rho", f"Inner CV {label}"),
      ("train_rho", f"Train {label}"),
      ("overfit_gap", "Overfit gap"),
    ]
  head = "".join(f"<th>{html.escape(label)}</th>" for _, label in cols)
  body = []
  for i, r in enumerate(cv_stats):
    cells = [f"<td>{_cv_cell(k, r.get(k))}</td>" for k, _ in cols]
    # Highlight the best descriptor (cv_stats is sorted best-first by the task's skill).
    cls = " class='pooled'" if i == 0 else ""
    body.append(f"<tr{cls}>{''.join(cells)}</tr>")
  return (
    "<div class='table-wrap'><table class='metrics'>"
    f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
  )


def _cv_cell(key, value):
  """Format one cross-validation table cell's content for column ``key``."""
  if key == "descriptor":
    return html.escape(str(value))
  if value is None:
    return "—"
  if key == "overfit_gap":
    return f"<span class='pill {_gap_level(value)}'>{float(value):.3f}</span>"
  try:
    return f"{float(value):.3f}"
  except Exception:
    return html.escape(str(value))


def _gap_level(gap):
  """Overfit-gap → colour class: low (<0.05) green, medium (<0.12) amber, else red."""
  try:
    g = float(gap)
  except (TypeError, ValueError):
    return "lvl-med"
  if g < 0.05:
    return "lvl-low"
  return "lvl-med" if g < 0.12 else "lvl-high"


def _cv_bars_html(stats):
  """HTML inner-CV skill bar per descriptor (AUROC, or Spearman rho for regression).

  The bar carries the descriptor's **identity colour** (matching its line in the ROC/PR figures, so
  this strip doubles as the shared legend); the overfit gap is shown as a colour-coded pill.
  ``stats`` is best-first, so its index is the identity rank."""
  key, _train_key, _label = skill_keys(stats)
  palette = _colors.descriptor_colors_hex(len(stats))
  rows = []
  for i, s in enumerate(stats):
    auc = s.get(key)
    if auc is None:
      continue
    gap = s.get("overfit_gap")
    # AUROC fills the track from 0.5 (chance); rho fills it from 0, which is its own chance level.
    span = (auc - 0.5) / 0.5 if key == "oof_auc" else auc
    pct = max(2.0, min(100.0, span * 100))
    did = html.escape(str(s.get("descriptor")))
    href = f"https://github.com/{GITHUB_ORG}/{did}"
    gap_pill = (
      f"<span class='pill {_gap_level(gap)}'>gap {float(gap):.2f}</span>" if gap is not None else ""
    )
    rows.append(
      f"<div class='perf-row'><a class='model' href='{href}' target='_blank'>{did} ↗</a>"
      f"<span class='perf-bar'><i class='diag-fill' style='width:{pct:.1f}%;background:{palette[i]}'></i></span>"
      f"<span class='perf-val'>{auc:.3f}</span><span class='perf-pills'>{gap_pill}</span></div>"
    )
  return f"<div class='perf-steps'>{''.join(rows)}</div>" if rows else ""
