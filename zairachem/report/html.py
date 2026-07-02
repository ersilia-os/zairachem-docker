"""Build a self-contained HTML report that displays the report figures, grouped by topic.

The page is deliberately plain and publication-oriented: a white background, the system font stack,
a sticky section nav, and responsive grids of figure cards. Redundant figures (e.g. the violin /
strip score distributions, or the UMAP / t-SNE / PCA projections) are collapsed into a single
*carousel* card that the reader pages through with arrows and dots.

Figures are referenced from the sibling ``png/`` folder (relative links), not base64-inlined — this
keeps ``report.html`` small, at the cost of the page not being standalone (it must travel with its
``png/`` folder). Written to ``<output_dir>/report/report.html`` next to ``png/``.
"""

import csv
import glob
import html
import json
import os
from datetime import datetime

from zairachem.base.vars import (
  DATA_FILENAME,
  DATA_SUBFOLDER,
  ESTIMATORS_SUBFOLDER,
  GITHUB_ORG,
  INPUT_SCHEMA_FILENAME,
  REPORT_SUBFOLDER,
  SESSION_FILE,
)
from zairachem.report import CELL_CM as _CELL_CM
from zairachem.report import GRID_COLS as _GRID_COLS
from zairachem.report import GRID_ROWS as _GRID_ROWS
from zairachem.report import colors as _colors
from zairachem.report import colors
from zairachem.report import perf

# Plots grouped into sections (anchor, heading, description, member stems). Unlisted → "More".
_CATEGORIES = [
  (
    "performance",
    "Model performance",
    "How the pooled model and individual estimators score against the ground-truth labels "
    "(resubstitution at fit; held-out validation when predicting a labelled set).",
    [
      "roc-curve",
      "confusion-matrix",
      "confusion-normalized",
      "roc-individual",
      "raw-classification-scores",
      "r2-individual",
      "regression-raw",
      "regression-trans",
    ],
  ),
  (
    "ranking",
    "Ranking & operating point",
    "Imbalance-aware ranking quality and the trade-offs around the decision threshold — what "
    "matters when triaging or screening a library.",
    ["pr-curve", "enrichment-curve", "enrichment-factor", "threshold-sweep"],
  ),
  (
    "validation",
    "Held-out validation",
    "Out-of-sample pooled AUROC/AUPR under random, scaffold and Butina 80:20 splits, repeated "
    "across seeds. Random is the optimism anchor; a large drop under scaffold/Butina indicates "
    "limited generalization to novel chemistry.",
    ["heldout-validation"],
  ),
  (
    "scores",
    "Score distributions",
    "Predicted scores across the active and inactive classes.",
    ["score-violin", "score-strip", "histogram-raw", "histogram-trans"],
  ),
  (
    "transform",
    "Value transformation",
    "How raw activity values were transformed for modeling.",
    ["transformation"],
  ),
]

_TITLES = {
  "actives-inactives": "Class balance",
  "roc-curve": "ROC curve",
  "confusion-matrix": "Confusion matrix",
  "score-violin": "Score distribution (violin)",
  "score-strip": "Score distribution (strip)",
  "roc-individual": "ROC by estimator",
  "raw-classification-scores": "Raw classification scores",
  "r2-individual": "R² by estimator",
  "regression-raw": "Predicted vs observed (raw)",
  "regression-trans": "Predicted vs observed (transformed)",
  "histogram-raw": "Value histogram (raw)",
  "histogram-trans": "Value histogram (transformed)",
  "transformation": "Value transformation",
  "cv-auroc": "Inner CV AUROC by descriptor",
  "cv-aupr": "Inner CV AUPR by descriptor",
  "cv-roc": "Cross-validation ROC",
  "cv-pr": "Cross-validation precision-recall",
  "cv-calibration": "Calibration (cross-validated)",
  "cv-score-distribution": "Out-of-fold score distribution",
  "projection-mwlogp": "Molecular weight vs LogP",
  "projection-umap": "UMAP projection",
  "projection-tsne": "t-SNE projection",
  "projection-pca": "PCA projection",
  "projection-tmap": "TMAP projection",
  "projection-correctness": "Prediction correctness in chemical space",
  "pr-curve": "Precision-recall curve",
  "enrichment-curve": "Enrichment curve",
  "enrichment-factor": "Enrichment factor",
  "threshold-sweep": "Threshold sweep",
  "confusion-normalized": "Confusion matrix (normalized)",
  "descriptor-metric-heatmap": "Descriptor metric heatmap",
  "oof-overfit-scatter": "Generalization vs overfitting",
  "pooled-vs-best-auroc": "Pooled vs per-descriptor AUROC",
  "heldout-validation": "Held-out AUROC by split",
  "class-donut": "Class balance (donut)",
  "class-waffle": "Class balance (waffle)",
  "property-mw": "Molecular weight",
  "property-logp": "LogP",
  "overview": "Report overview",
  "step-timing": "Step timing",
  "phase-time": "Time by phase",
  "model-timing": "Time per model",
  "resource-timeline": "Resource usage over time",
  "compute-provenance": "Compute provenance",
}

# Redundant / near-redundant figures collapsed into a single carousel card. A group renders once, at
# its ``home`` section; its members are suppressed everywhere else. Members appear in listed order and
# only if their PNG is present, so a group left with a single present member degrades to a plain card.
_GROUPS = [
  {"key": "roc", "title": "ROC curve", "home": "performance", "members": ["roc-curve"]},
  {
    "key": "confusion",
    "title": "Confusion matrix",
    "home": "performance",
    "members": ["confusion-matrix", "confusion-normalized"],
  },
  {
    "key": "regression",
    "title": "Predicted vs observed",
    "home": "performance",
    "members": ["regression-trans", "regression-raw"],
  },
  {
    "key": "scoredist",
    "title": "Score distribution",
    "home": "scores",
    "members": ["score-violin", "score-strip"],
  },
  {
    "key": "histogram",
    "title": "Value histogram",
    "home": "scores",
    "members": ["histogram-trans", "histogram-raw"],
  },
]
_STEM_TO_GROUP = {m: g for g in _GROUPS for m in g["members"]}

_ACRONYMS = {
  "roc": "ROC",
  "auroc": "AUROC",
  "umap": "UMAP",
  "tsne": "t-SNE",
  "pca": "PCA",
  "r2": "R²",
}

# Curated metric columns for the performance table (only those present are shown).
_METRIC_COLS = [
  ("model", "Model"),
  ("auroc", "AUROC"),
  ("aupr", "AUPR"),
  ("accuracy", "Accuracy"),
  ("balanced_accuracy", "Bal. acc."),
  ("precision", "Precision"),
  ("recall", "Recall"),
  ("f1_score", "F1"),
  ("mcc", "MCC"),
  ("r2", "R²"),
  ("mae", "MAE"),
  ("rmse", "RMSE"),
  ("pearson", "Pearson"),
  ("spearman", "Spearman"),
]


def _projection_label(stem):
  """Clean label for a dynamic projection stem, reusing the base projection titles in ``_TITLES``.

  ``projection-merged-umap`` → "UMAP projection"; ``projection-umap-active`` → "UMAP · actives".
  """
  body = stem[len("projection-") :]
  if body.startswith("merged-"):
    return _TITLES.get(f"projection-{body[len('merged-') :]}")
  for noun in ("active", "inactive"):
    if body.endswith(f"-{noun}"):
      base = _TITLES.get(f"projection-{body[: -(len(noun) + 1)]}")
      short = base.replace(" projection", "") if base else None
      return f"{short} · {noun}s" if short else None
  return None


def _humanize(stem):
  if stem in _TITLES:
    return _TITLES[stem]
  if stem.startswith("projection-"):
    label = _projection_label(stem)
    if label:
      return label
  words = stem.replace("_", " ").replace("-", " ").split()
  return " ".join(_ACRONYMS.get(w.lower(), w.capitalize()) for w in words)


def _img_src(report_dir, stem):
  """Relative path to ``png/{stem}.png``.

  The report references the figure files directly (kept alongside the HTML in ``png/``) instead of
  base64-inlining them, which keeps ``report.html`` small rather than duplicating every PNG into it.
  The page is therefore not standalone: it must travel together with its ``png/`` folder.
  """
  return f"png/{stem}.png"


# --- Physical figure sizes & the composite reference grid ----------------------------------------
# Figures are exported by stylia at 600 DPI (see report/__init__.py), so a saved PNG's real size is
# pixels / 600 inches. The report documents a reference grid (it mirrors the old 6×10 matplotlib
# poster grid) so users can tile the downloaded plots into a composite figure at the right scale.
_FIGURE_DPI = 600
_CELL_IN = _CELL_CM / 2.54  # ≈ 1.181 in


def _png_px(path):
  """(width_px, height_px) read from a PNG's IHDR header, or None on any failure."""
  try:
    with open(path, "rb") as f:
      head = f.read(24)
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n":
      return None
    w = int.from_bytes(head[16:20], "big")
    h = int.from_bytes(head[20:24], "big")
    return (w, h) if w > 0 and h > 0 else None
  except Exception:
    return None


def _figure_size(report_dir, stem):
  """Physical size of a figure PNG and its footprint on the reference grid, or None.

  Returns ``{w_cm, h_cm, w_in, h_in, cols, rows}`` where cols/rows are the size rounded to whole
  3 cm cells (minimum 1).
  """
  px = _png_px(os.path.join(report_dir, "png", f"{stem}.png"))
  if px is None:
    return None
  w_in, h_in = px[0] / _FIGURE_DPI, px[1] / _FIGURE_DPI
  w_cm, h_cm = w_in * 2.54, h_in * 2.54
  return {
    "w_cm": w_cm,
    "h_cm": h_cm,
    "w_in": w_in,
    "h_in": h_in,
    "cols": max(1, round(w_cm / _CELL_CM)),
    "rows": max(1, round(h_cm / _CELL_CM)),
  }


def _figure_dims(report_dir, stem):
  """``(cells, size)`` display strings for a figure, e.g. ``("2×4", "12.0×6.1 cm · 4.73×2.40 in")``."""
  sz = _figure_size(report_dir, stem)
  if sz is None:
    return None
  cells = f"{sz['rows']}×{sz['cols']}"
  size = f"{sz['w_cm']:.1f}×{sz['h_cm']:.1f} cm · {sz['w_in']:.2f}×{sz['h_in']:.2f} in"
  return cells, size


def _dim_badge(report_dir, stem):
  """Small card caption: the figure's grid footprint (n×m cells) + its real size in cm and inches."""
  dims = _figure_dims(report_dir, stem)
  if dims is None:
    return ""
  cells, size = dims
  return (
    f"<div class='dim'><span class='cells'>{cells}</span> <span class='dimsize'>{size}</span></div>"
  )


def _grid_svg():
  """Inline SVG of the reference grid (full width) with cm/inch axis labels and one example plot."""
  s = 16  # px per cm
  cell = int(_CELL_CM * s)  # 48
  gw, gh = _GRID_COLS * cell, _GRID_ROWS * cell
  x0, y0 = 52, 24
  vb_w, vb_h = x0 + gw + 16, y0 + gh + 34
  full_cm_w = _GRID_COLS * _CELL_CM
  full_cm_h = _GRID_ROWS * _CELL_CM
  lines = [
    f"<line x1='{x0 + i * cell}' y1='{y0}' x2='{x0 + i * cell}' y2='{y0 + gh}'/>"
    for i in range(_GRID_COLS + 1)
  ] + [
    f"<line x1='{x0}' y1='{y0 + j * cell}' x2='{x0 + gw}' y2='{y0 + j * cell}'/>"
    for j in range(_GRID_ROWS + 1)
  ]
  # One example footprint: "2×3" = 2 rows × 3 columns, i.e. 3 cells wide × 2 cells tall.
  ew, eh = 3 * cell, 2 * cell
  example = (
    f"<rect class='ex-a' x='{x0}' y='{y0}' width='{ew}' height='{eh}'/>"
    f"<text class='exlab' x='{x0 + ew // 2}' y='{y0 + eh // 2 + 5}'>2×3</text>"
  )
  top = (
    f"<text class='axis' x='{x0 + gw // 2}' y='15' text-anchor='middle'>"
    f"{full_cm_w:.0f} cm · {full_cm_w / 2.54:.2f} in</text>"
  )
  left = (
    f"<text class='axis' x='16' y='{y0 + gh // 2}' text-anchor='middle' "
    f"transform='rotate(-90 16 {y0 + gh // 2})'>{full_cm_h:.0f} cm · {full_cm_h / 2.54:.2f} in</text>"
  )
  cap = (
    f"<text class='cap' x='{x0 + gw // 2}' y='{y0 + gh + 24}' text-anchor='middle'>"
    f"1 cell = {_CELL_CM:.0f} × {_CELL_CM:.0f} cm ({_CELL_IN:.2f} × {_CELL_IN:.2f} in)</text>"
  )
  return (
    f"<svg class='gridfig' viewBox='0 0 {vb_w} {vb_h}' role='img' "
    f"aria-label='{_GRID_COLS} by {_GRID_ROWS} reference grid of {_CELL_CM:.0f} cm cells'>"
    f"{example}{''.join(lines)}{top}{left}{cap}</svg>"
  )


def _color_key_html():
  """The single transversal color key shared by every plot (built from colors.LEGEND)."""
  items = "".join(
    f"<span class='item'><span class='swatch' style='background:{_colors.hexcol(key)}'></span>"
    f"{html.escape(label)}</span>"
    for label, key in _colors.LEGEND
  )
  return f"<div class='legend-key'>{items}</div>"


def _about_figures_html():
  """Show the composite reference grid (full width), a short caption, and the shared color key."""
  cap = (
    f"<p class='grid-cap'>Reference grid: <b>{_GRID_ROWS} rows × {_GRID_COLS} columns</b>, one cell "
    f"<b>{_CELL_CM:.0f} cm ({_CELL_IN:.2f} in)</b> square "
    f"(full sheet {_GRID_COLS * _CELL_CM:.0f} cm wide × {_GRID_ROWS * _CELL_CM:.0f} cm tall). Each "
    "plot card's <b>n×m</b> badge is its footprint in cells (<b>rows × columns</b>) — download the "
    "plots and tile them to compose a figure at the right scale.</p>"
  )
  key_caption = "<p class='grid-cap'>Colors are consistent across every plot:</p>"
  return f"<div class='about'>{_grid_svg()}{cap}{key_caption}{_color_key_html()}</div>"


def _semantic_css():
  """CSS for semantic colors, generated from ``report.colors`` so it can never drift from the plots.

  Defines the phase badges (``.ph-*``), provenance bars (``.prov-*``), the Task badges
  (``.badge-clf`` / ``.badge-reg``), and the About color-key swatches.
  """
  ph = "".join(f".ph-{k}{{background:{v};}}" for k, v in _colors.PHASE_COLORS.items())
  prov = (
    f".prov-store{{background:{_colors.hexcol('store')};}}"
    f".prov-computed{{background:{_colors.hexcol('computed')};}}"
  )
  clf, reg = _colors.hexcol("classification"), _colors.hexcol("regression")
  badges = (
    f".badge-clf{{background:{clf}22;border:1px solid {clf};color:var(--fg);}}"
    f".badge-reg{{background:{reg}22;border:1px solid {reg};color:var(--fg);}}"
  )
  key = (
    ".about .legend-key{display:flex;flex-wrap:wrap;gap:10px 18px;justify-content:center;"
    "margin:8px auto 0;max-width:760px;font-size:13px;color:var(--fg);}"
    ".about .legend-key .item{display:inline-flex;align-items:center;gap:7px;}"
    ".about .swatch{width:13px;height:13px;border-radius:3px;display:inline-block;"
    "border:1px solid rgba(0,0,0,.08);}"
  )
  return ph + prov + badges + key


def _load_params(output_dir):
  from zairachem.base import params_path

  try:
    with open(params_path(output_dir)) as f:
      return json.load(f)
  except Exception:
    return {}


def _n_compounds(output_dir):
  try:
    import pandas as pd

    return len(pd.read_csv(os.path.join(output_dir, DATA_SUBFOLDER, DATA_FILENAME)))
  except Exception:
    return None


def _read_performance(report_dir):
  """Return (header, rows) of performance_table.csv (same dir as the page), or (None, None)."""
  try:
    with open(os.path.join(report_dir, "performance_table.csv")) as f:
      reader = csv.DictReader(f)
      rows = list(reader)
      return reader.fieldnames, rows
  except Exception:
    return None, None


def _fmt_num(v):
  try:
    f = float(v)
    return f"{f:.3f}" if abs(f) < 1000 else f"{int(f):,}"
  except Exception:
    return html.escape(str(v))


def _performance_table_html(report_dir):
  fields, rows = _read_performance(report_dir)
  if not rows:
    return ""
  cols = [(k, label) for k, label in _METRIC_COLS if fields and k in fields]
  if len(cols) <= 1:
    return ""
  head = "".join(f"<th>{html.escape(label)}</th>" for _, label in cols)
  body = []
  for r in rows:
    is_pooled = r.get("model") == "pooled"
    cells = []
    for k, _ in cols:
      val = r.get(k, "")
      val = html.escape(str(val)) if k == "model" else _fmt_num(val)
      cells.append(f"<td>{val}</td>")
    cls = " class='pooled'" if is_pooled else ""
    body.append(f"<tr{cls}>{''.join(cells)}</tr>")
  return (
    "<div class='table-wrap'><table class='metrics'>"
    f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
  )


# Held-out validation schemas: (csv strategy key, display label). Random is styled as the anchor.
_VALIDATION_SCHEMAS = [
  ("random", "Random"),
  ("scaffold", "Scaffold"),
  ("scaffold_det", "Scaffold (DeepChem)"),
  ("butina", "Butina"),
]


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
  body = []
  for strat, label in ordered:
    srows = by_strategy[strat]
    au_m, au_s = _mean_std([_num(r, "auroc") for r in srows])
    ap_m, ap_s = _mean_std([_num(r, "aupr") for r in srows])
    au = f"{au_m:.3f} ± {au_s:.3f}" if au_m is not None else "—"
    ap = f"{ap_m:.3f} ± {ap_s:.3f}" if ap_m is not None else "—"
    cls = " class='pooled'" if strat == "random" else ""
    body.append(
      f"<tr{cls}><td>{html.escape(label)}</td><td>{len(srows)}</td><td>{au}</td><td>{ap}</td></tr>"
    )
  head = "<th>Split schema</th><th>Folds</th><th>AUROC (mean ± std)</th><th>AUPR (mean ± std)</th>"
  return (
    "<div class='table-wrap'><table class='metrics'>"
    f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
  )


_CSS = """
:root { color-scheme: light; --fg:#1f2328; --muted:#6e7781; --line:#e6e8eb; --bg:#fff; --soft:#f6f8fa; --link:#0969da; --sidebar:248px; --card-h:420px; }
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; color:var(--fg); background:var(--bg); line-height:1.55; }
.layout { display:flex; align-items:flex-start; }
.sidebar { position:sticky; top:0; height:100vh; width:var(--sidebar); flex:0 0 var(--sidebar); border-right:1px solid var(--line); padding:32px 20px; overflow-y:auto; }
.sidebar .brand { font-size:16px; font-weight:600; letter-spacing:-.01em; word-break:break-word; }
.sidebar .brand-sub { color:var(--muted); font-size:12.5px; margin-top:2px; }
.sidebar { display:flex; flex-direction:column; }
.sidebar nav { flex:0 0 auto; }
.sidebar footer { margin-top:auto; padding-top:18px; border-top:1px solid var(--line); color:var(--muted); font-size:11.5px; display:flex; flex-direction:column; gap:8px; }
.sidebar footer a { color:var(--link); text-decoration:none; }
.sidebar footer a:hover { text-decoration:underline; }
.badge { display:inline-block; border-radius:999px; padding:1px 9px; font-size:12px; font-weight:600; }
.badge-on { background:#eaf6ec; border:1px solid #b7e0c0; color:#1a7f37; }
.badge-off { background:var(--soft); border:1px solid var(--line); color:var(--muted); }
.store-proj { color:var(--muted); font-size:12px; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
.sidebar nav { display:flex; flex-direction:column; gap:1px; margin-top:22px; }
.sidebar nav a { color:var(--muted); text-decoration:none; font-size:13.5px; padding:6px 10px; border-radius:7px; }
.sidebar nav a:hover { background:var(--soft); color:var(--fg); }
.content { flex:1 1 auto; min-width:0; max-width:920px; margin:0 auto; padding:44px 40px 88px; }
header h1 { font-size:24px; font-weight:600; margin:0 0 2px; letter-spacing:-.01em; }
header .sub { color:var(--muted); font-size:14px; }
.chips { margin-top:16px; display:flex; flex-wrap:wrap; gap:8px; }
.chip { font-size:13px; background:var(--soft); border:1px solid var(--line); border-radius:999px; padding:4px 12px; }
.chip b { font-weight:600; }
section { padding-top:40px; scroll-margin-top:24px; }
section > h2 { font-size:17px; font-weight:600; margin:0 0 4px; }
section > .desc { color:var(--muted); font-size:13.5px; margin:0 0 18px; }
.grid { display:grid; gap:20px; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); }
/* Two figure cards side by side, filling the main content column (same width as the tables/text). */
.grid2 { grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }
@media (max-width:720px) { .grid2 { grid-template-columns:1fr; } }
/* Every figure card is a fixed-height flex column (uniform boxes across the whole report): the header,
   badge, controls and links are fixed rows; the image area flexes to fill and the figure is contained
   (centred, never stretched) within it. */
.card { margin:0; height:var(--card-h); display:flex; flex-direction:column; border:1px solid var(--line); border-radius:12px; background:#fff; padding:16px; transition:box-shadow .18s ease, transform .18s ease; }
.card:hover { box-shadow:0 8px 24px rgba(27,31,36,.09); transform:translateY(-2px); }
.card h3 { font-size:14.5px; font-weight:600; margin:0 0 12px; flex:0 0 auto; }
.card a.fig { flex:1 1 auto; min-height:0; display:flex; align-items:center; justify-content:center; }
.card img { max-width:100%; max-height:100%; width:auto; height:auto; display:block; border-radius:6px; }
.card .links { margin-top:10px; font-size:12.5px; color:var(--muted); flex:0 0 auto; }
.card .links a { color:var(--link); text-decoration:none; }
.card .links a:hover { text-decoration:underline; }
.card .dim { margin:-4px 0 12px; font-size:12px; color:var(--muted); font-variant-numeric:tabular-nums; flex:0 0 auto; }
.card .dim .cells { display:inline-block; background:var(--soft); border:1px solid var(--line); border-radius:999px; padding:0 8px; margin-right:6px; font-weight:600; color:var(--fg); }
.about .gridfig { display:block; width:100%; max-width:760px; height:auto; margin:6px auto 0; }
.about .gridfig line { stroke:var(--line); stroke-width:1; }
.about .gridfig .ex-a { fill:#dbeafe; }
.about .gridfig .exlab { fill:var(--fg); font:600 13px sans-serif; text-anchor:middle; }
.about .gridfig .axis { fill:var(--muted); font:600 11px sans-serif; }
.about .gridfig .cap { fill:var(--muted); font:11px sans-serif; }
.about .grid-cap { text-align:center; color:var(--muted); font-size:13px; max-width:760px; margin:12px auto 0; }
.about .grid-cap b { color:var(--fg); }
.carousel-head { display:flex; align-items:baseline; justify-content:space-between; gap:10px; margin:0 0 12px; flex:0 0 auto; }
.carousel-head h3 { margin:0; }
.carousel-label { color:var(--muted); font-size:12.5px; white-space:nowrap; }
.carousel-track { position:relative; flex:1 1 auto; min-height:0; }
.carousel .slide { display:none; }
.carousel .slide.active { display:flex; align-items:center; justify-content:center; height:100%; }
/* Contain (not stretch) slide images so a tall figure in a wide slider doesn't blow up the card:
   wide slides still fill the width, tall ones are centred and height-capped. */
.carousel .slide img { max-width:100%; max-height:100%; width:auto; height:auto; object-fit:contain; display:block; border-radius:6px; }
.carousel-ctl { display:flex; align-items:center; justify-content:center; gap:14px; margin-top:12px; flex:0 0 auto; }
.carousel-ctl button.prev, .carousel-ctl button.next { border:1px solid var(--line); background:#fff; color:var(--fg); width:30px; height:30px; border-radius:50%; font-size:16px; line-height:1; cursor:pointer; padding:0; }
.carousel-ctl button.prev:hover, .carousel-ctl button.next:hover { background:var(--soft); }
.carousel-ctl .dots { display:flex; gap:7px; align-items:center; }
.carousel-ctl .dot { width:8px; height:8px; padding:0; border-radius:50%; border:1px solid var(--muted); background:#fff; cursor:pointer; }
.carousel-ctl .dot.active { background:var(--link); border-color:var(--link); }
.carousel:focus { outline:2px solid var(--link); outline-offset:2px; }
.poster { display:flex; flex-direction:column; gap:12px; }
.poster .prow { display:flex; gap:12px; align-items:flex-start; }
.poster .pcell { flex:1 1 0; min-width:0; border:1px solid var(--line); border-radius:10px; background:#fff; padding:8px; transition:box-shadow .18s ease, transform .18s ease; }
.poster .pcell:hover { box-shadow:0 6px 18px rgba(27,31,36,.09); transform:translateY(-2px); }
.poster .pcell img { width:100%; height:auto; display:block; border-radius:4px; }
@media (max-width:720px) { .poster .prow { flex-wrap:wrap; } .poster .pcell { flex-basis:45%; } }
.table-wrap { overflow-x:auto; border:1px solid var(--line); border-radius:12px; }
table.metrics { border-collapse:collapse; width:100%; font-size:13px; }
table.metrics th, table.metrics td { padding:9px 14px; text-align:right; white-space:nowrap; border-bottom:1px solid var(--line); }
table.metrics th:first-child, table.metrics td:first-child { text-align:left; }
table.metrics thead th { background:var(--soft); font-weight:600; color:var(--fg); }
table.metrics tbody tr:last-child td { border-bottom:none; }
table.metrics tr.pooled td { font-weight:600; background:#fbfdff; }
table.kv { border-collapse:collapse; width:100%; max-width:560px; font-size:13.5px; border:1px solid var(--line); border-radius:12px; overflow:hidden; }
table.kv td { padding:9px 14px; border-bottom:1px solid var(--line); vertical-align:top; }
table.kv tr:last-child td { border-bottom:none; }
table.kv td:first-child { color:var(--muted); width:200px; white-space:nowrap; }
table.kv code { background:var(--soft); border-radius:4px; padding:1px 6px; font-size:12.5px; }
.cfg-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:8px 28px; margin-bottom:8px; }
.cfg-cell { display:flex; flex-direction:column; gap:2px; padding:2px 0; }
.cfg-cell .k { font-size:10.5px; font-weight:500; letter-spacing:.05em; text-transform:uppercase; color:var(--muted); }
.cfg-cell .v { font-size:13.5px; font-weight:500; letter-spacing:-.005em; word-break:break-word; }
h3.cfg-h { font-size:13px; font-weight:600; margin:24px 0 10px; }
table.cfg-models { border-collapse:collapse; width:100%; font-size:13px; }
table.cfg-models th, table.cfg-models td { padding:9px 14px; text-align:left; border-bottom:1px solid var(--line); vertical-align:middle; }
table.cfg-models thead th { background:var(--soft); font-weight:600; font-size:12px; letter-spacing:.02em; text-transform:uppercase; color:var(--muted); }
table.cfg-models tbody tr:last-child td { border-bottom:none; }
table.cfg-models td.title { color:var(--fg); width:100%; }
a.model { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12.5px; color:var(--link); text-decoration:none; white-space:nowrap; }
a.model:hover { text-decoration:underline; }
span.ver { display:inline-block; background:var(--soft); border:1px solid var(--line); color:var(--muted); border-radius:999px; padding:1px 9px; font-size:12px; }
.perf-legend { display:flex; flex-wrap:wrap; gap:14px; margin:0 0 12px; font-size:12px; color:var(--muted); }
.perf-leg { display:inline-flex; align-items:center; gap:6px; }
.perf-leg i { width:11px; height:11px; border-radius:3px; display:inline-block; }
.perf-machine { color:var(--muted); font-size:12.5px; margin:2px 0 0; }
.perf-steps { display:flex; flex-direction:column; gap:7px; }
.perf-row { display:grid; grid-template-columns:150px 1fr 52px 188px; align-items:center; gap:12px; font-size:13px; }
.perf-lbl { color:var(--fg); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.perf-bar { height:14px; background:var(--soft); border-radius:7px; overflow:hidden; }
.perf-bar i { display:block; height:100%; border-radius:7px; }
.perf-val { color:var(--muted); font-variant-numeric:tabular-nums; white-space:nowrap; text-align:right; }
.perf-pills { display:flex; gap:6px; justify-content:flex-end; }
.pill { display:inline-flex; align-items:center; justify-content:center; min-width:84px; border:1px solid var(--line); border-radius:999px; padding:1px 9px; font-size:11.5px; white-space:nowrap; font-variant-numeric:tabular-nums; }
.pill.lvl-low { background:#eaf6ec; border-color:#b7e0c0; color:#1a7f37; }
.pill.lvl-med { background:#fdf3e3; border-color:#f2d9a8; color:#9a6700; }
.pill.lvl-high { background:#fdecec; border-color:#f3c0c0; color:#c1342d; }
.caution { margin:14px 0 4px; padding:10px 14px; border-left:3px solid #c1342d; background:#fdecec; color:#8f2018; font-size:12.5px; line-height:1.5; border-radius:0 6px 6px 0; }
.caution strong { color:#c1342d; }
.prov-row { display:grid; grid-template-columns:120px 1fr auto; align-items:center; gap:12px; font-size:13px; margin-bottom:7px; }
.prov-bar { display:flex; height:14px; background:var(--soft); border-radius:7px; overflow:hidden; }
.balance-bar { display:flex; height:16px; background:var(--soft); border-radius:8px; overflow:hidden; margin:2px 0 10px; }
.balance-bar i { display:block; height:100%; }
.prov-bar i { display:block; height:100%; }
.perf-leg i.prov-store, .perf-leg i.prov-computed { border-radius:3px; }
.prov-val { color:var(--muted); font-variant-numeric:tabular-nums; white-space:nowrap; }
footer { margin-top:56px; padding-top:22px; border-top:1px solid var(--line); color:var(--muted); font-size:13px; display:flex; flex-wrap:wrap; gap:16px; justify-content:space-between; }
footer a { color:var(--link); text-decoration:none; } footer a:hover { text-decoration:underline; }
@media (max-width:820px) {
  .layout { flex-direction:column; }
  .sidebar { position:static; height:auto; width:auto; flex:none; border-right:none; border-bottom:1px solid var(--line); padding:20px 24px; }
  .sidebar nav { flex-direction:row; flex-wrap:wrap; gap:4px 14px; margin-top:14px; }
  .content { padding:32px 24px 64px; max-width:none; }
}
@media print {
  .sidebar { display:none; }
  .content { max-width:none; padding:0; }
  .card { break-inside:avoid; box-shadow:none; height:auto; }
  section { break-inside:avoid-page; }
  .card:hover { transform:none; box-shadow:none; }
  .carousel .slide { display:block !important; margin-bottom:10px; height:auto; }
  .carousel-ctl, .carousel-label { display:none; }
}
"""


def _y_column(output_dir):
  """The user-supplied label column name from ``inputs/input_schema.json`` (e.g. ``"dili"``)."""
  try:
    with open(os.path.join(output_dir, DATA_SUBFOLDER, INPUT_SCHEMA_FILENAME)) as f:
      return json.load(f).get("values_column")
  except Exception:
    return None


def _run_date(output_dir):
  """Human-readable run date from the ``time_stamp`` in ``session.json``, or ``None``."""
  try:
    with open(os.path.join(output_dir, SESSION_FILE)) as f:
      ts = json.load(f).get("time_stamp")
    return datetime.fromtimestamp(ts).strftime("%-d %b %Y") if ts else None
  except Exception:
    return None


def _run_mode(output_dir):
  """Run mode ("fit" or "predict") from ``session.json``; defaults to "fit" on any error."""
  try:
    with open(os.path.join(output_dir, SESSION_FILE)) as f:
      return "predict" if json.load(f).get("mode") == "predict" else "fit"
  except Exception:
    return "fit"


def _model_titles(output_dir, params, ids):
  """Resolve ``eosXXX`` → human title, fetching missing ones from GitHub and caching to params.

  Titles live under ``parameters.json``'s ``model_titles`` map. Newly fetched (real) titles are
  written back so later report runs need no network. Fetch failures degrade to ``None`` (rows then
  show id + version only); a read-only directory simply skips the write-back.
  """
  from zairachem.base.utils.utils import fetch_model_metadata

  cached = dict(params.get("model_titles") or {})
  resolved = {}
  newly = {}
  for mid in ids:
    if mid in cached:
      resolved[mid] = cached[mid]
      continue
    meta = fetch_model_metadata(mid) or {}
    title = meta.get("Title")
    resolved[mid] = title
    if title:
      newly[mid] = title
  if newly:
    from zairachem.base import params_path

    try:
      cached.update(newly)
      params["model_titles"] = cached
      with open(params_path(output_dir)) as f:
        on_disk = json.load(f)
      on_disk["model_titles"] = {**(on_disk.get("model_titles") or {}), **newly}
      with open(params_path(output_dir), "w") as f:
        json.dump(on_disk, f, indent=2)
    except Exception:
      pass
  return resolved


def _model_table_html(ids, versions, titles):
  """A table of Ersilia models: monospace id linked to the hub, version pill, human title."""
  if not ids:
    return ""
  rows = []
  for mid in ids:
    href = f"https://github.com/{GITHUB_ORG}/{html.escape(mid)}"
    ver = versions.get(mid)
    ver_html = f"<span class='ver'>{html.escape(str(ver))}</span>" if ver else "—"
    title = titles.get(mid)
    title_html = html.escape(title) if title else "—"
    rows.append(
      f"<tr><td><a class='model' href='{href}' target='_blank'>{html.escape(mid)} ↗</a></td>"
      f"<td>{ver_html}</td><td class='title'>{title_html}</td></tr>"
    )
  return (
    "<div class='table-wrap'><table class='cfg-models'>"
    "<thead><tr><th>Model</th><th>Version</th><th>Title</th></tr></thead>"
    f"<tbody>{''.join(rows)}</tbody></table></div>"
  )


def _config_section_html(output_dir, params, n):
  """Render the run configuration: a scalar stat grid plus featurizer and projection model tables."""
  task = params.get("task") or ""
  stats = []
  y_col = _y_column(output_dir)
  if y_col:
    stats.append(("Y column", y_col))
  if n is not None:
    stats.append(("Compounds", f"{n:,}"))
  date = _run_date(output_dir)
  if date:
    stats.append(("Run date", date))
  # Task as a colored badge (Classification = turquoise / Regression = amber); plain text otherwise.
  t = task.lower()
  if t == "classification":
    task_v = "<span class='badge badge-clf'>Classification</span>"
  elif t == "regression":
    task_v = "<span class='badge badge-reg'>Regression</span>"
  else:
    task_v = html.escape(task.capitalize() if task else "—")
  cells = f"<div class='cfg-cell'><span class='k'>Task</span><span class='v'>{task_v}</span></div>"
  cells += "".join(
    f"<div class='cfg-cell'><span class='k'>{html.escape(k)}</span>"
    f"<span class='v'>{html.escape(str(v))}</span></div>"
    for k, v in stats
  )
  # Isaura store as a colored badge: green "on" (with the project id) / grey "off".
  store = params.get("store") or params.get("contribute_store")
  if store:
    store_v = (
      "<span class='badge badge-on'>on</span> "
      f"<span class='store-proj'>{html.escape(str(store))}</span>"
    )
  else:
    store_v = "<span class='badge badge-off'>off</span>"
  cells += f"<div class='cfg-cell'><span class='k'>Isaura store</span><span class='v'>{store_v}</span></div>"
  grid = f"<div class='cfg-grid'>{cells}</div>"

  feats = params.get("featurizer_ids") or []
  projs = params.get("projection_ids") or []
  versions = params.get("latest_featurizer_version") or {}
  titles = _model_titles(output_dir, params, feats + projs)

  out = [grid]
  feat_table = _model_table_html(feats, versions, titles)
  if feat_table:
    out.append(f"<h3 class='cfg-h'>Featurizers</h3>{feat_table}")
  proj_table = _model_table_html(projs, versions, titles)
  if proj_table:
    out.append(f"<h3 class='cfg-h'>Projections</h3>{proj_table}")
  return "".join(out)


def _load_level(pct):
  """Map a usage percentage to a colour class: low (<50) → green, med (<80) → amber, else red."""
  try:
    p = float(pct)
  except (TypeError, ValueError):
    return "lvl-med"
  if p < 50:
    return "lvl-low"
  return "lvl-med" if p < 80 else "lvl-high"


def _machine_line(output_dir):
  """A muted one-line summary of the host machine, or ``""`` if no host specs were recorded."""
  h = perf.host_info(output_dir)
  if not h:
    return ""
  bits = []
  cores = h.get("cpu_logical")
  phys = h.get("cpu_physical")
  if cores:
    bits.append(f"{cores} CPUs" + (f" ({phys} cores)" if phys and phys != cores else ""))
  if h.get("ram_total_gb"):
    bits.append(f"{h['ram_total_gb']:.0f} GB RAM")
  for key in ("arch", "system"):
    if h.get(key):
      bits.append(html.escape(str(h[key])))
  if not bits:
    return ""
  return f"<p class='perf-machine'>⊟ {' · '.join(bits)}</p>"


def _perf_steps_html(steps):
  """Timing + per-step resources: one bar row per sub-step, coloured by phase, with CPU/RAM pills."""
  timed = [s for s in steps if s["seconds"] is not None]
  if not timed:
    return ""
  longest = max((s["seconds"] for s in timed), default=0) or 1
  phases = []
  seen = set()
  for s in timed:
    if s["phase"] not in seen:
      seen.add(s["phase"])
      phases.append(s["phase"])
  legend = "".join(
    f"<span class='perf-leg'><i class='ph-{html.escape(p)}'></i>{html.escape(p.capitalize())}</span>"
    for p in phases
  )
  rows = []
  for s in timed:
    pct = max(2.0, 100.0 * s["seconds"] / longest)
    pills = []
    if s["cpu"] is not None:
      pills.append(f"<span class='pill cpu {_load_level(s['cpu'])}'>CPU {s['cpu']:.0f}%</span>")
    if s["ram_used_gb"] is not None:
      lvl = _load_level(s["ram_pct"])
      pills.append(f"<span class='pill ram {lvl}'>RAM {s['ram_used_gb']:.1f} GB</span>")
    rows.append(
      f"<div class='perf-row'><span class='perf-lbl'>{html.escape(s['label'])}</span>"
      f"<span class='perf-bar'><i class='ph-{html.escape(s['phase'])}' style='width:{pct:.1f}%'></i></span>"
      f"<span class='perf-val'>{perf.fmt_duration(s['seconds'])}</span>"
      f"<span class='perf-pills'>{''.join(pills)}</span></div>"
    )
  return f"<div class='perf-legend'>{legend}</div><div class='perf-steps'>{''.join(rows)}</div>"


def _provenance_html(prov):
  """Per-model stacked store-vs-computed bars plus an aggregate reuse summary."""
  if not prov:
    return ""
  rows = []
  for m in prov["models"]:
    total = m["total"] or 1
    store_pct = 100.0 * m["from_store"] / total
    comp_pct = 100.0 * m["computed"] / total
    href = f"https://github.com/{GITHUB_ORG}/{html.escape(m['id'])}"
    rows.append(
      f"<div class='prov-row'>"
      f"<a class='model' href='{href}' target='_blank'>{html.escape(m['id'])} ↗</a>"
      f"<span class='prov-bar'>"
      f"<i class='prov-store' style='width:{store_pct:.1f}%'></i>"
      f"<i class='prov-computed' style='width:{comp_pct:.1f}%'></i></span>"
      f"<span class='prov-val'>{m['from_store']:,} stored · {m['computed']:,} computed</span>"
      f"</div>"
    )
  t = prov["totals"]
  summary = ""
  if t["reuse_pct"] is not None:
    summary = (
      f"<p class='desc'><b>{t['reuse_pct']:.0f}%</b> of descriptor computations were served from the "
      f"store ({t['from_store']:,} reused · {t['computed']:,} freshly computed).</p>"
    )
  migrated_total = sum(v for v in (prov["migrated"] or {}).values() if isinstance(v, (int, float)))
  migrated_note = ""
  if migrated_total:
    migrated_note = (
      f"<p class='desc'>{int(migrated_total):,} descriptor rows were seeded from the public lake "
      f"into the project store during setup.</p>"
    )
  legend = (
    "<div class='perf-legend'><span class='perf-leg'><i class='prov-store'></i>From store</span>"
    "<span class='perf-leg'><i class='prov-computed'></i>Computed</span></div>"
  )
  return f"{summary}{legend}<div class='perf-steps'>{''.join(rows)}</div>{migrated_note}"


# Matplotlib compute figures, grouped into sliders. Each group renders as one carousel card (members
# pageable with arrows/dots); a group with a single present figure degrades to a plain card and
# upgrades to a slider automatically once more figures are added to it.
_COMPUTE_GROUPS = [
  {
    "key": "compute-timing",
    "title": "Timing & resources",
    "members": ["step-timing", "phase-time", "resource-timeline"],
  },
  {
    "key": "compute-models",
    "title": "Per-model",
    "members": ["compute-provenance", "model-timing"],
  },
]


def _computational_performance_html(output_dir, report_dir, present, assigned, params):
  """Dashboard: run cost (timing + RAM/CPU), host machine, store size, and descriptor provenance.

  Also embeds the matplotlib compute figures (when present) and marks their stems ``assigned`` so they
  don't fall through to the "More" section.
  """
  tel = perf.step_telemetry(output_dir)
  prov = perf.provenance(output_dir)

  stats = []
  if tel["total_seconds"] is not None:
    stats.append(("Total time", perf.fmt_duration(tel["total_seconds"])))
  if tel["peak_ram_gb"] is not None:
    total_ram = f" / {tel['ram_total_gb']:.0f} GB" if tel["ram_total_gb"] else ""
    stats.append(("Peak RAM", f"{tel['peak_ram_gb']:.1f}{total_ram}"))
  if tel["peak_cpu"] is not None:
    stats.append(("Peak CPU", f"{tel['peak_cpu']:.0f}%"))
  if prov and prov["totals"]["reuse_pct"] is not None:
    stats.append(("Descriptors reused", f"{prov['totals']['reuse_pct']:.0f}%"))
  store = perf.store_size(params)
  if store:
    stats.append(("Store size", perf.fmt_size(store["total_bytes"])))

  # Build the figure sliders: one carousel per group (or a plain card when a group has a single
  # present figure). ``fig_cards`` is the rendered HTML; ``fig_stems`` the stems to mark assigned.
  fig_cards = []
  fig_stems = []
  for g in _COMPUTE_GROUPS:
    members = [m for m in g["members"] if m in present]
    if not members:
      continue
    fig_stems.extend(members)
    fig_cards.append(
      _card(report_dir, members[0]) if len(members) == 1 else _carousel(report_dir, g, members)
    )
  if not stats and not tel["steps"] and not prov and not fig_cards:
    return ""

  out = []
  if stats:
    cells = "".join(
      f"<div class='cfg-cell'><span class='k'>{html.escape(k)}</span>"
      f"<span class='v'>{html.escape(str(v))}</span></div>"
      for k, v in stats
    )
    out.append(f"<div class='cfg-grid'>{cells}</div>")
  out.append(_machine_line(output_dir))

  steps_html = _perf_steps_html(tel["steps"])
  if steps_html:
    out.append(f"<h3 class='cfg-h'>Step timing &amp; resources</h3>{steps_html}")
  elif prov:
    out.append(
      "<p class='desc'>Per-step timing and resource usage are recorded on a fresh fit run.</p>"
    )

  prov_html = _provenance_html(prov)
  if prov_html:
    out.append(f"<h3 class='cfg-h'>Compute provenance</h3>{prov_html}")

  if fig_cards:
    assigned.update(fig_stems)
    grid = "<div class='grid grid2'>" + "".join(fig_cards) + "</div>"
    out.append(f"<h3 class='cfg-h'>Figures</h3>{grid}")
  return "".join(out)


_DATASET_GROUPS = [
  {
    "key": "class-balance",
    "title": "Class balance",
    "members": ["actives-inactives", "class-donut", "class-waffle"],
  },
  {
    "key": "properties",
    "title": "Property distributions",
    "members": ["property-mw", "property-logp"],
  },
]


def _class_counts(output_dir):
  """Active/inactive counts of the labelled set from ``inputs/data.csv``, or ``None``."""
  try:
    import pandas as pd

    df = pd.read_csv(os.path.join(output_dir, DATA_SUBFOLDER, DATA_FILENAME))
    col = next((c for c in df.columns if "bin" in c and "_skip" not in c and "_aux" not in c), None)
    if col is None:
      return None
    y = pd.to_numeric(df[col], errors="coerce").dropna()
    total = int(len(y))
    if total == 0:
      return None
    actives = int((y == 1).sum())
    return {
      "total": total,
      "actives": actives,
      "inactives": total - actives,
      "pct_active": 100.0 * actives / total,
    }
  except Exception:
    return None


def _dataset_html(output_dir, report_dir, present, assigned):
  """Dataset composition: class-balance stat cells + bar, then the class-balance & property sliders."""
  counts = _class_counts(output_dir)
  fig_cards, fig_stems = [], []
  for g in _DATASET_GROUPS:
    members = [m for m in g["members"] if m in present]
    if not members:
      continue
    fig_stems.extend(members)
    fig_cards.append(
      _card(report_dir, members[0]) if len(members) == 1 else _carousel(report_dir, g, members)
    )
  if not counts and not fig_cards:
    return ""

  out = []
  if counts:
    stats = [
      ("Compounds", f"{counts['total']:,}"),
      ("Actives", f"{counts['actives']:,}"),
      ("Inactives", f"{counts['inactives']:,}"),
      ("% active", f"{counts['pct_active']:.1f}%"),
    ]
    cells = "".join(
      f"<div class='cfg-cell'><span class='k'>{html.escape(k)}</span>"
      f"<span class='v'>{html.escape(v)}</span></div>"
      for k, v in stats
    )
    a_col, i_col = colors.hexcol("active"), colors.hexcol("inactive")
    a = counts["pct_active"]
    out.append(f"<div class='cfg-grid'>{cells}</div>")
    out.append(
      "<div class='balance-bar'>"
      f"<i style='width:{a:.1f}%;background:{a_col}'></i>"
      f"<i style='width:{100 - a:.1f}%;background:{i_col}'></i></div>"
      "<div class='perf-legend'>"
      f"<span class='perf-leg'><i style='background:{a_col}'></i>Active</span>"
      f"<span class='perf-leg'><i style='background:{i_col}'></i>Inactive</span></div>"
    )
  if fig_cards:
    assigned.update(fig_stems)
    grid = "<div class='grid grid2'>" + "".join(fig_cards) + "</div>"
    out.append(f"<h3 class='cfg-h'>Figures</h3>{grid}")
  return "".join(out)


# Projection display order (interpretable MW-vs-LogP first, then the learned embeddings).
_PROJECTION_ORDER = ["mwlogp", "umap", "tsne", "pca", "tmap"]


def _chemical_space_html(output_dir, report_dir, present, assigned):
  """Chemical space: two sliders over the projections, each coloured by true class (no in-plot legend).

  Container 1 ("merged") pages the per-class density+point map of each projection; container 2
  ("by class") pages, per projection, the actives map then the inactives map. Projections are
  discovered from the ``projection-merged-*`` stems present.
  """
  bases = [s[len("projection-merged-") :] for s in present if s.startswith("projection-merged-")]
  if not bases:
    return ""
  bases.sort(key=lambda b: (_PROJECTION_ORDER.index(b) if b in _PROJECTION_ORDER else 99, b))

  merged = [f"projection-merged-{b}" for b in bases if f"projection-merged-{b}" in present]
  by_class = []
  for b in bases:
    for noun in ("active", "inactive"):
      stem = f"projection-{b}-{noun}"
      if stem in present:
        by_class.append(stem)

  cards = []
  if merged:
    g = {"key": "chemspace-merged", "title": "Overview", "members": merged}
    cards.append(
      _card(report_dir, merged[0]) if len(merged) == 1 else _carousel(report_dir, g, merged)
    )
  if by_class:
    g = {"key": "chemspace-class", "title": "By class", "members": by_class}
    cards.append(
      _card(report_dir, by_class[0]) if len(by_class) == 1 else _carousel(report_dir, g, by_class)
    )
  if not cards:
    return ""
  assigned.update(merged + by_class)

  # No in-plot legends (per design) — a single crimson/cobalt colour key for the whole section.
  a_col, i_col = colors.hexcol("active"), colors.hexcol("inactive")
  legend = (
    "<div class='perf-legend'>"
    f"<span class='perf-leg'><i style='background:{a_col}'></i>Active</span>"
    f"<span class='perf-leg'><i style='background:{i_col}'></i>Inactive</span></div>"
  )
  grid = "<div class='grid grid2'>" + "".join(cards) + "</div>"
  return legend + grid


def _read_cv_stats(output_dir):
  """Per-descriptor lazy-qsar CV reports (estimators/*/*/cv_report.json), best OOF AUROC first."""
  stats = []
  pattern = os.path.join(output_dir, ESTIMATORS_SUBFOLDER, "*", "*", "cv_report.json")
  for f in glob.glob(pattern):
    try:
      with open(f) as fh:
        rep = json.load(fh)
      rep["descriptor"] = os.path.basename(os.path.dirname(f))
      stats.append(rep)
    except Exception:
      continue
  stats.sort(key=lambda r: r.get("oof_auc") if r.get("oof_auc") is not None else -1, reverse=True)
  return stats


def _cv_summary(cv_stats):
  aucs = [s["oof_auc"] for s in cv_stats if s.get("oof_auc") is not None]
  if not aucs:
    return {}
  gaps = [s["overfit_gap"] for s in cv_stats if s.get("overfit_gap") is not None]
  best = cv_stats[0]
  return {
    "n_descriptors": len(cv_stats),
    "mean_oof_auc": sum(aucs) / len(aucs),
    "best_oof_auc": best.get("oof_auc"),
    "best_descriptor": best.get("descriptor"),
    "mean_overfit_gap": (sum(gaps) / len(gaps)) if gaps else None,
  }


def _cv_table_html(cv_stats):
  if not cv_stats:
    return ""
  cols = [
    ("descriptor", "Descriptor"),
    ("oof_auc", "Inner CV AUROC"),
    ("train_auc", "Train AUROC"),
    ("overfit_gap", "Overfit gap"),
    ("portfolio", "Algorithms"),
    ("decision_cutoff_proba", "Cutoff"),
  ]
  head = "".join(f"<th>{html.escape(label)}</th>" for _, label in cols)
  body = []
  for i, r in enumerate(cv_stats):
    cells = []
    for k, _ in cols:
      v = r.get(k)
      if k == "descriptor":
        cell = html.escape(str(v))
      elif k == "portfolio":
        cell = html.escape(", ".join(v) if isinstance(v, list) else str(v or "—"))
      elif v is None:
        cell = "—"
      elif k == "overfit_gap":
        cell = f"<span class='pill {_gap_level(v)}'>{float(v):.3f}</span>"
      else:
        try:
          cell = f"{float(v):.3f}"
        except Exception:
          cell = html.escape(str(v))
      cells.append(f"<td>{cell}</td>")
    # Highlight the best descriptor (cv_stats is sorted by OOF AUROC desc).
    cls = " class='pooled'" if i == 0 else ""
    body.append(f"<tr{cls}>{''.join(cells)}</tr>")
  return (
    "<div class='table-wrap'><table class='metrics'>"
    f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
  )


def _gap_level(gap):
  """Overfit-gap → colour class: low (<0.05) green, medium (<0.12) amber, else red."""
  try:
    g = float(gap)
  except (TypeError, ValueError):
    return "lvl-med"
  if g < 0.05:
    return "lvl-low"
  return "lvl-med" if g < 0.12 else "lvl-high"


def _cv_stats(output_dir):
  """Per-descriptor CV stats from the finished model, or the cached ``report/cv_stats.json`` fallback."""
  stats = _read_cv_stats(output_dir)
  if stats:
    return stats
  try:
    with open(os.path.join(output_dir, REPORT_SUBFOLDER, "cv_stats.json")) as f:
      return json.load(f).get("per_descriptor") or []
  except Exception:
    return []


def _cv_bars_html(stats):
  """HTML inner-CV AUROC bar per descriptor. The bar carries the descriptor's **identity colour**
  (matching its line in the ROC/PR figures, so this strip doubles as the shared legend); the overfit
  gap is shown as a colour-coded pill. ``stats`` is best-first, so its index is the identity rank."""
  palette = _colors.descriptor_colors_hex(len(stats))
  rows = []
  for i, s in enumerate(stats):
    auc = s.get("oof_auc")
    if auc is None:
      continue
    gap = s.get("overfit_gap")
    pct = max(2.0, min(100.0, (auc - 0.5) / 0.5 * 100))  # AUROC 0.5..1.0 → 0..100% of the track
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


_DIAG_GROUPS = [
  {
    "key": "diag-compare",
    "title": "Per-descriptor comparison",
    "members": [
      "descriptor-metric-heatmap",
      "cv-auroc",
      "cv-aupr",
      "pooled-vs-best-auroc",
      "oof-overfit-scatter",
    ],
  },
  {
    "key": "diag-curves",
    "title": "Cross-validation curves",
    "members": ["cv-roc", "cv-pr", "cv-calibration", "cv-score-distribution"],
  },
]


def _diagnostics_html(output_dir, report_dir, present, assigned):
  """Per-descriptor inner (lazy-qsar CV) diagnostics: summary stats, OOF-AUROC bars, table, sliders."""
  stats = _cv_stats(output_dir)
  summary = _cv_summary(stats) if stats else {}

  fig_cards, fig_stems = [], []
  for g in _DIAG_GROUPS:
    members = [m for m in g["members"] if m in present]
    if not members:
      continue
    fig_stems.extend(members)
    fig_cards.append(
      _card(report_dir, members[0]) if len(members) == 1 else _carousel(report_dir, g, members)
    )
  if not summary and not fig_cards:
    return ""

  out = []
  if summary:
    cells_data = [
      ("Descriptors", str(summary["n_descriptors"])),
      ("Best descriptor", summary.get("best_descriptor") or "—"),
    ]
    if summary.get("best_oof_auc") is not None:
      cells_data.append(("Best inner CV AUROC", f"{summary['best_oof_auc']:.3f}"))
    if summary.get("mean_oof_auc") is not None:
      cells_data.append(("Mean inner CV AUROC", f"{summary['mean_oof_auc']:.3f}"))
    if summary.get("mean_overfit_gap") is not None:
      cells_data.append(("Mean overfit gap", f"{summary['mean_overfit_gap']:.3f}"))
    cells = "".join(
      f"<div class='cfg-cell'><span class='k'>{html.escape(k)}</span>"
      f"<span class='v'>{html.escape(v)}</span></div>"
      for k, v in cells_data
    )
    out.append(f"<div class='cfg-grid'>{cells}</div>")
    out.append(
      "<p class='caution'>&#9888; These are <strong>inner</strong> cross-validation scores, "
      "computed inside lazy-qsar during descriptor and algorithm selection. Treat them with "
      "caution: they are optimistically biased and most likely <strong>overestimate</strong> "
      "real-world performance. For an honest out-of-sample estimate use the held-out validation.</p>"
    )
    out.append(f"<h3 class='cfg-h'>Per-descriptor table</h3>{_cv_table_html(stats)}")
    out.append(f"<h3 class='cfg-h'>Inner CV AUROC by descriptor</h3>{_cv_bars_html(stats)}")
  if fig_cards:
    assigned.update(fig_stems)
    out.append(
      "<h3 class='cfg-h'>Figures</h3><div class='grid grid2'>" + "".join(fig_cards) + "</div>"
    )
  return "".join(out)


# Overview poster layout: rows of (figure stem, relative width). Mirrors the old 6×10 matplotlib
# grid, but assembled in HTML from the individual figure PNGs. Cells whose PNG is missing are
# dropped; an all-missing row is skipped. Row widths are proportional (flex-grow).
_POSTER_ROWS = [
  [("pr-curve", 2), ("enrichment-curve", 3), ("threshold-sweep", 3), ("enrichment-factor", 2)],
  [
    ("confusion-normalized", 2),
    ("oof-overfit-scatter", 2),
    ("pooled-vs-best-auroc", 2),
    ("descriptor-metric-heatmap", 2),
    ("projection-correctness", 2),
  ],
  [("property-mw", 5), ("property-logp", 5)],
]


def _overview_poster_html(report_dir, present):
  """Assemble the overview poster: a flex grid of the individual figure PNGs. ``""`` if none."""
  rows_html = []
  for row in _POSTER_ROWS:
    cells = [(s, w) for s, w in row if s in present]
    if not cells:
      continue
    cell_html = "".join(
      f"<a class='pcell' style='flex-grow:{w}' href='png/{s}.png' target='_blank'>"
      f"<img src='{_img_src(report_dir, s)}' alt='{html.escape(_humanize(s))}' loading='lazy'></a>"
      for s, w in cells
    )
    rows_html.append(f"<div class='prow'>{cell_html}</div>")
  if not rows_html:
    return ""
  return "<div class='poster'>" + "".join(rows_html) + "</div>"


def _download_name(report_dir, stem, ext):
  """Suggested download filename, encoding the figure's grid footprint, e.g. ``step-timing_2x4.png``."""
  sz = _figure_size(report_dir, stem)
  return f"{stem}_{sz['rows']}x{sz['cols']}.{ext}" if sz else f"{stem}.{ext}"


def _links_html(report_dir, stem):
  """The ``PNG · PDF`` download row for a single figure (PDF only if its file exists).

  The ``download`` attribute names the saved file by its grid footprint (e.g. ``step-timing_2x4.png``)
  so a downloaded plot carries its n×m for assembling composites; the on-disk path stays ``png/…``.
  """
  png_name = _download_name(report_dir, stem, "png")
  links = [f"<a class='png' href='png/{stem}.png' download='{png_name}' target='_blank'>PNG</a>"]
  if os.path.exists(os.path.join(report_dir, "pdf", f"{stem}.pdf")):
    pdf_name = _download_name(report_dir, stem, "pdf")
    links.append(
      f"<a class='pdf' href='pdf/{stem}.pdf' download='{pdf_name}' target='_blank'>PDF</a>"
    )
  return " · ".join(links)


def _card(report_dir, stem):
  title = html.escape(_humanize(stem))
  return (
    f"<figure class='card'><h3>{title}</h3>"
    f"{_dim_badge(report_dir, stem)}"
    f"<a class='fig' href='png/{stem}.png' target='_blank'>"
    f"<img src='{_img_src(report_dir, stem)}' alt='{title}' loading='lazy'></a>"
    f"<div class='links'>{_links_html(report_dir, stem)}</div></figure>"
  )


def _carousel(report_dir, group, members):
  """Collapse a group of related figures into one paged card. ``members`` are present stems, ordered.

  Each ``.slide`` carries its variant label and relative PDF path as data attributes so the small
  vanilla-JS controller can update the caption and the PNG/PDF download links as the reader pages.
  """
  slides = []
  for i, stem in enumerate(members):
    label = html.escape(_humanize(stem))
    pdf = (
      f"pdf/{stem}.pdf" if os.path.exists(os.path.join(report_dir, "pdf", f"{stem}.pdf")) else ""
    )
    active = " active" if i == 0 else ""
    sz = _figure_size(report_dir, stem)
    dl = f"{sz['rows']}x{sz['cols']}" if sz else ""
    dims = _figure_dims(report_dir, stem)
    cells_attr = html.escape(dims[0]) if dims else ""
    size_attr = html.escape(dims[1]) if dims else ""
    slides.append(
      f"<a class='slide{active}' data-label=\"{label}\" data-pdf='{pdf}' data-dl='{dl}' "
      f'data-cells="{cells_attr}" data-size="{size_attr}" '
      f"href='png/{stem}.png' target='_blank'>"
      f"<img src='{_img_src(report_dir, stem)}' alt='{label}' loading='lazy'></a>"
    )
  ctl = ""
  if len(members) > 1:
    dots = []
    for i in range(len(members)):
      active = " active" if i == 0 else ""
      dots.append(f"<button class='dot{active}' data-i='{i}' aria-label='Figure {i + 1}'></button>")
    ctl = (
      "<div class='carousel-ctl'><button class='prev' aria-label='Previous'>‹</button>"
      f"<div class='dots'>{''.join(dots)}</div>"
      "<button class='next' aria-label='Next'>›</button></div>"
    )
  return (
    f"<figure class='card carousel'><div class='carousel-head'>"
    f"<h3>{html.escape(group['title'])}</h3>"
    f"<span class='carousel-label'>{html.escape(_humanize(members[0]))}</span></div>"
    f"{_dim_badge(report_dir, members[0])}"
    f"<div class='carousel-track'>{''.join(slides)}</div>{ctl}"
    f"<div class='links'>{_links_html(report_dir, members[0])}</div></figure>"
  )


def _render_items(report_dir, anchor, items, present, rendered_groups, assigned):
  """Render a section's items, collapsing grouped figures into one carousel at their home section.

  Grouped stems seen outside their home section (or after their group already rendered) are skipped;
  they surface in the carousel emitted at the home section instead. ``assigned`` tracks everything
  rendered so the trailing "More" section can pick up genuine leftovers.
  """
  cards = []
  for stem in items:
    group = _STEM_TO_GROUP.get(stem)
    if group is None:
      cards.append(_card(report_dir, stem))
      assigned.add(stem)
      continue
    if group["home"] != anchor or group["key"] in rendered_groups:
      continue
    rendered_groups.add(group["key"])
    group_members = [m for m in group["members"] if m in present]
    assigned.update(group_members)
    if len(group_members) == 1:
      cards.append(_card(report_dir, group_members[0]))
    else:
      cards.append(_carousel(report_dir, group, group_members))
  return cards


_JS = r"""
document.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('.carousel').forEach(function (car) {
    var slides = Array.prototype.slice.call(car.querySelectorAll('.slide'));
    if (slides.length < 2) return;
    var dots = Array.prototype.slice.call(car.querySelectorAll('.dot'));
    var label = car.querySelector('.carousel-label');
    var pngLink = car.querySelector('.links a.png');
    var pdfLink = car.querySelector('.links a.pdf');
    var cellsEl = car.querySelector('.dim .cells');
    var sizeEl = car.querySelector('.dim .dimsize');
    var i = 0;
    function show(n) {
      i = (n + slides.length) % slides.length;
      slides.forEach(function (s, k) { s.classList.toggle('active', k === i); });
      dots.forEach(function (d, k) { d.classList.toggle('active', k === i); });
      var s = slides[i];
      if (label) label.textContent = s.getAttribute('data-label') || '';
      if (cellsEl) cellsEl.textContent = s.getAttribute('data-cells') || '';
      if (sizeEl) sizeEl.textContent = s.getAttribute('data-size') || '';
      var href = s.getAttribute('href');
      var stem = href.replace(/^png\//, '').replace(/\.png$/, '');
      var dl = s.getAttribute('data-dl');
      if (pngLink) {
        pngLink.setAttribute('href', href);
        if (dl) pngLink.setAttribute('download', stem + '_' + dl + '.png');
      }
      if (pdfLink) {
        var pdf = s.getAttribute('data-pdf');
        if (pdf) {
          pdfLink.setAttribute('href', pdf);
          if (dl) pdfLink.setAttribute('download', stem + '_' + dl + '.pdf');
          pdfLink.style.display = '';
        } else {
          pdfLink.style.display = 'none';
        }
      }
    }
    var prev = car.querySelector('.prev'), next = car.querySelector('.next');
    if (prev) prev.addEventListener('click', function () { show(i - 1); });
    if (next) next.addEventListener('click', function () { show(i + 1); });
    dots.forEach(function (d, k) { d.addEventListener('click', function () { show(k); }); });
    car.setAttribute('tabindex', '0');
    car.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowLeft') { show(i - 1); e.preventDefault(); }
      else if (e.key === 'ArrowRight') { show(i + 1); e.preventDefault(); }
    });
  });
});
"""


def write_html_report(output_dir):
  """Write ``report/report.html`` grouping every figure PNG by topic. Returns the path or None."""
  report_dir = os.path.join(output_dir, REPORT_SUBFOLDER)
  png_dir = os.path.join(report_dir, "png")
  stems = sorted(
    os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(png_dir, "*.png"))
  )
  if not stems:
    return None
  present = set(stems)

  params = _load_params(output_dir)
  model = os.path.basename(os.path.normpath(output_dir))
  n = _n_compounds(output_dir)

  # Header: a big report-type title ("ZairaChem Fit Report"), with the model name and run date as a
  # smaller subtitle. No chips / figure tally.
  kind = "Predict" if _run_mode(output_dir) == "predict" else "Fit"
  date = _run_date(output_dir)
  report_heading = f"ZairaChem {kind} Report"
  subtitle = f"Model name: {html.escape(model)}" + (f" · {html.escape(date)}" if date else "")
  tab_title = f"{report_heading} · {model}"

  # Build sections. Configuration first; then the plot categories ("Chemical space" gathers any
  # projection-* figure generically); then any leftovers.
  assigned = set()
  perf_table = _performance_table_html(report_dir)
  config_table = _config_section_html(output_dir, params, n)
  cv_stats = _read_cv_stats(output_dir)
  if cv_stats:
    with open(os.path.join(report_dir, "cv_stats.json"), "w") as f:
      json.dump({"per_descriptor": cv_stats, "summary": _cv_summary(cv_stats)}, f, indent=2)
  sections = []
  rendered_groups = set()
  # (The Overview poster is intentionally omitted for now.)
  sections.append(("config", "Configuration", "Run settings and model identifiers.", config_table))
  perf_section = _computational_performance_html(output_dir, report_dir, present, assigned, params)
  if perf_section:
    sections.append((
      "compute",
      "Computational performance",
      "How long the run took, how much memory and CPU it used, and where the molecular descriptors "
      "came from — reused from the isaura store or freshly computed.",
      perf_section,
    ))
  dataset_section = _dataset_html(output_dir, report_dir, present, assigned)
  if dataset_section:
    sections.append((
      "dataset",
      "Dataset",
      "Composition of the labelled set (training compounds at fit; the labelled inputs when "
      "predicting) — class balance and molecular property distributions.",
      dataset_section,
    ))
  space_section = _chemical_space_html(output_dir, report_dir, present, assigned)
  if space_section:
    sections.append((
      "space",
      "Chemical space",
      "Low-dimensional embeddings of the molecules — the built-in molecular-weight-vs-LogP map and "
      "each computed projection (UMAP, t-SNE, …).",
      space_section,
    ))
  for anchor, heading, desc, members in _CATEGORIES:
    if anchor == "space":
      items = sorted(s for s in stems if s.startswith("projection-"))
    else:
      items = [s for s in members if s in present]
    cards = _render_items(report_dir, anchor, items, present, rendered_groups, assigned)
    grid = "<div class='grid'>" + "".join(cards) + "</div>" if cards else ""
    if anchor == "performance":
      inner = perf_table + grid
    elif anchor == "validation":
      inner = _validation_table_html(report_dir) + grid
    else:
      inner = grid
    if inner:
      sections.append((anchor, heading, desc, inner))
  # Per-descriptor inner diagnostics (inner lazy-qsar CV): inserted right after the chemical space.
  diag_section = _diagnostics_html(output_dir, report_dir, present, assigned)
  if diag_section:
    entry = (
      "diagnostics",
      "Per descriptor inner diagnostics",
      "How each molecular descriptor performed under lazy-qsar's internal cross-validation — the "
      "out-of-fold AUROC, the train-vs-CV overfit gap, and the algorithm portfolio chosen per "
      "descriptor.",
      diag_section,
    )
    idx = next((i + 1 for i, s in enumerate(sections) if s[0] == "space"), len(sections))
    sections.insert(idx, entry)
  leftovers = [s for s in stems if s not in assigned]
  if leftovers:
    grid = "<div class='grid'>" + "".join(_card(report_dir, s) for s in leftovers) + "</div>"
    sections.append(("more", "More", "", grid))

  # About the figures: kept last — reference material on plot sizes and the composite grid.
  sections.append((
    "about",
    "About the figures",
    "The physical size of each plot and the reference grid for assembling a composite figure.",
    _about_figures_html(),
  ))

  nav = "".join(f"<a href='#{a}'>{html.escape(h)}</a>" for a, h, _, _ in sections)

  def _section(a, h, d, inner):
    desc = f"<p class='desc'>{html.escape(d)}</p>" if d else ""
    return f"<section id='{a}'><h2>{html.escape(h)}</h2>{desc}{inner}</section>"

  body_sections = "".join(_section(a, h, d, inner) for a, h, d, inner in sections)

  # Footer downloads (tables live next to the page; xlsx at the model root once finish ran).
  links = []
  for fname, label in (
    ("output_table.csv", "Predictions (CSV)"),
    ("performance_table.csv", "Performance (CSV)"),
  ):
    if os.path.exists(os.path.join(report_dir, fname)):
      links.append(f"<a href='{fname}'>{label}</a>")
  footer = (f"<div>{' · '.join(links)}</div>" if links else "<div></div>") + (
    "<div>Brought to you by the "
    "<a href='https://github.com/ersilia-os/ersilia'>Ersilia Open Source Initiative</a>. "
    "Produced with <a href='https://github.com/ersilia-os/zaira-chem'>ZairaChem</a>.</div>"
  )

  doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(tab_title)}</title>
<style>{_CSS}{_semantic_css()}</style>
</head>
<body>
<div class="layout">
  <aside class="sidebar">
    <div class="brand">{html.escape(kind)} Report</div>
    <div class="brand-sub">Model name: {html.escape(model)}</div>
    <nav>{nav}</nav>
    <footer>{footer}</footer>
  </aside>
  <main class="content">
    <header>
      <h1>{html.escape(report_heading)}</h1>
      <div class="sub">{subtitle}</div>
    </header>
    {body_sections}
  </main>
</div>
<script>{_JS}</script>
</body>
</html>
"""
  out = os.path.join(report_dir, "report.html")
  with open(out, "w") as f:
    f.write(doc)
  return out
