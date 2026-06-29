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

from zairachem.base.vars import (
  DATA_FILENAME,
  DATA_SUBFOLDER,
  ESTIMATORS_SUBFOLDER,
  OUTPUT_XLSX_FILENAME,
  REPORT_SUBFOLDER,
)

# Plots grouped into sections (anchor, heading, description, member stems). Unlisted → "More".
_CATEGORIES = [
  (
    "dataset",
    "Dataset",
    "Composition of the labelled set (training compounds at fit; the labelled inputs when predicting).",
    ["actives-inactives", "property-distributions"],
  ),
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
    "diagnostics",
    "Per-descriptor diagnostics",
    "How each descriptor generalizes in cross-validation, and whether pooling beats the best "
    "single descriptor.",
    ["descriptor-metric-heatmap", "oof-overfit-scatter", "pooled-vs-best-auroc"],
  ),
  (
    "crossval",
    "Cross-validation",
    "Internal lazy-qsar cross-validation (out-of-fold) per descriptor — honest performance, "
    "unlike the resubstitution numbers in Model performance.",
    [],
  ),
  (
    "scores",
    "Score distributions",
    "Predicted scores across the active and inactive classes.",
    ["score-violin", "score-strip", "histogram-raw", "histogram-trans"],
  ),
  (
    "space",
    "Chemical space",
    "Low-dimensional embeddings of the molecules.",
    ["projection-umap", "projection-tsne", "projection-pca"],
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
  "cv-auroc": "Cross-validation AUROC",
  "cv-roc": "Cross-validation ROC",
  "cv-calibration": "Calibration (cross-validated)",
  "cv-score-distribution": "Out-of-fold score distribution",
  "projection-mwlogp": "Molecular weight vs LogP",
  "projection-umap": "UMAP projection",
  "projection-tsne": "t-SNE projection",
  "projection-pca": "PCA projection",
  "projection-correctness": "Prediction correctness in chemical space",
  "pr-curve": "Precision-recall curve",
  "enrichment-curve": "Enrichment curve",
  "enrichment-factor": "Enrichment factor",
  "threshold-sweep": "Threshold sweep",
  "confusion-normalized": "Confusion matrix (normalized)",
  "descriptor-metric-heatmap": "Descriptor metric heatmap",
  "oof-overfit-scatter": "Generalization vs overfitting",
  "pooled-vs-best-auroc": "Pooled vs per-descriptor AUROC",
  "property-distributions": "Property distributions by class",
  "overview": "Report overview",
}

# Redundant / near-redundant figures collapsed into a single carousel card. A group renders once, at
# its ``home`` section; its members are suppressed everywhere else. Members appear in listed order and
# only if their PNG is present, so a group left with a single present member degrades to a plain card.
_GROUPS = [
  {"key": "roc", "title": "ROC curve", "home": "performance", "members": ["roc-curve", "cv-roc"]},
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
    "key": "descauroc",
    "title": "Per-descriptor AUROC",
    "home": "diagnostics",
    "members": ["roc-individual", "cv-auroc", "pooled-vs-best-auroc"],
  },
  {
    "key": "scoredist",
    "title": "Score distribution",
    "home": "scores",
    "members": ["score-violin", "score-strip", "cv-score-distribution"],
  },
  {
    "key": "histogram",
    "title": "Value histogram",
    "home": "scores",
    "members": ["histogram-trans", "histogram-raw"],
  },
  {
    "key": "projection",
    "title": "Chemical space projection",
    "home": "space",
    "members": ["projection-mwlogp", "projection-umap", "projection-tsne", "projection-pca"],
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


def _humanize(stem):
  if stem in _TITLES:
    return _TITLES[stem]
  words = stem.replace("_", " ").replace("-", " ").split()
  return " ".join(_ACRONYMS.get(w.lower(), w.capitalize()) for w in words)


def _img_src(report_dir, stem):
  """Relative path to ``png/{stem}.png``.

  The report references the figure files directly (kept alongside the HTML in ``png/``) instead of
  base64-inlining them, which keeps ``report.html`` small rather than duplicating every PNG into it.
  The page is therefore not standalone: it must travel together with its ``png/`` folder.
  """
  return f"png/{stem}.png"


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


_CSS = """
:root { color-scheme: light; --fg:#1f2328; --muted:#6e7781; --line:#e6e8eb; --bg:#fff; --soft:#f6f8fa; --link:#0969da; --sidebar:248px; }
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; color:var(--fg); background:var(--bg); line-height:1.55; }
.layout { display:flex; align-items:flex-start; }
.sidebar { position:sticky; top:0; height:100vh; width:var(--sidebar); flex:0 0 var(--sidebar); border-right:1px solid var(--line); padding:32px 20px; overflow-y:auto; }
.sidebar .brand { font-size:16px; font-weight:600; letter-spacing:-.01em; word-break:break-word; }
.sidebar .brand-sub { color:var(--muted); font-size:12.5px; margin-top:2px; }
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
.card { border:1px solid var(--line); border-radius:12px; background:#fff; padding:16px; transition:box-shadow .18s ease, transform .18s ease; }
.card:hover { box-shadow:0 8px 24px rgba(27,31,36,.09); transform:translateY(-2px); }
.card h3 { font-size:14.5px; font-weight:600; margin:0 0 12px; }
.card a.fig { display:block; }
.card img { width:100%; height:auto; display:block; border-radius:6px; }
.card .links { margin-top:10px; font-size:12.5px; color:var(--muted); }
.card .links a { color:var(--link); text-decoration:none; }
.card .links a:hover { text-decoration:underline; }
.carousel-head { display:flex; align-items:baseline; justify-content:space-between; gap:10px; margin:0 0 12px; }
.carousel-head h3 { margin:0; }
.carousel-label { color:var(--muted); font-size:12.5px; white-space:nowrap; }
.carousel-track { position:relative; }
.carousel .slide { display:none; }
.carousel .slide.active { display:block; }
.carousel .slide img { width:100%; height:auto; display:block; border-radius:6px; }
.carousel-ctl { display:flex; align-items:center; justify-content:center; gap:14px; margin-top:12px; }
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
  .card { break-inside:avoid; box-shadow:none; }
  section { break-inside:avoid-page; }
  .card:hover { transform:none; box-shadow:none; }
  .carousel .slide { display:block !important; margin-bottom:10px; }
  .carousel-ctl, .carousel-label { display:none; }
}
"""


def _config_table_html(params, n):
  """Render the run configuration (parameters.json) as a clean key/value table."""

  def code(v):
    return f"<code>{html.escape(str(v))}</code>"

  rows = [("Task", html.escape(str(params.get("task", "—"))))]
  if n is not None:
    rows.append(("Compounds", f"{n:,}"))
  feats = params.get("featurizer_ids") or []
  rows.append(("Descriptors", "  ".join(code(m) for m in feats) if feats else "—"))
  projs = params.get("projection_ids") or []
  rows.append((
    "Projections",
    "MW vs LogP " + ("(built-in)" if not projs else "+ " + "  ".join(code(m) for m in projs)),
  ))
  store = params.get("store") or params.get("contribute_store")
  rows.append(("Isaura store", code(store) if store else "off"))
  vers = params.get("latest_featurizer_version") or {}
  if vers:
    rows.append((
      "Model versions",
      "  ".join(f"{code(k)} {html.escape(str(v))}" for k, v in vers.items()),
    ))
  body = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in rows)
  return f"<table class='kv'><tbody>{body}</tbody></table>"


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
    ("oof_auc", "CV AUROC"),
    ("train_auc", "Train AUROC"),
    ("overfit_gap", "Overfit gap"),
    ("portfolio", "Algorithms"),
    ("decision_cutoff_proba", "Cutoff"),
  ]
  head = "".join(f"<th>{html.escape(label)}</th>" for _, label in cols)
  body = []
  for r in cv_stats:
    cells = []
    for k, _ in cols:
      v = r.get(k)
      if k == "descriptor":
        cell = html.escape(str(v))
      elif k == "portfolio":
        cell = html.escape(", ".join(v) if isinstance(v, list) else str(v or "—"))
      elif v is None:
        cell = "—"
      else:
        try:
          cell = f"{float(v):.3f}"
        except Exception:
          cell = html.escape(str(v))
      cells.append(f"<td>{cell}</td>")
    body.append(f"<tr>{''.join(cells)}</tr>")
  return (
    "<div class='table-wrap'><table class='metrics'>"
    f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
  )


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
  [("property-distributions", 10)],
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


def _links_html(report_dir, stem):
  """The ``PNG · PDF`` download row for a single figure (PDF only if its file exists)."""
  links = [f"<a class='png' href='png/{stem}.png' target='_blank'>PNG</a>"]
  if os.path.exists(os.path.join(report_dir, "pdf", f"{stem}.pdf")):
    links.append(f"<a class='pdf' href='pdf/{stem}.pdf' target='_blank'>PDF</a>")
  return " · ".join(links)


def _card(report_dir, stem):
  title = html.escape(_humanize(stem))
  return (
    f"<figure class='card'><h3>{title}</h3>"
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
    slides.append(
      f"<a class='slide{active}' data-label=\"{label}\" data-pdf='{pdf}' "
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


_JS = """
document.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('.carousel').forEach(function (car) {
    var slides = Array.prototype.slice.call(car.querySelectorAll('.slide'));
    if (slides.length < 2) return;
    var dots = Array.prototype.slice.call(car.querySelectorAll('.dot'));
    var label = car.querySelector('.carousel-label');
    var pngLink = car.querySelector('.links a.png');
    var pdfLink = car.querySelector('.links a.pdf');
    var i = 0;
    function show(n) {
      i = (n + slides.length) % slides.length;
      slides.forEach(function (s, k) { s.classList.toggle('active', k === i); });
      dots.forEach(function (d, k) { d.classList.toggle('active', k === i); });
      var s = slides[i];
      if (label) label.textContent = s.getAttribute('data-label') || '';
      if (pngLink) pngLink.setAttribute('href', s.getAttribute('href'));
      if (pdfLink) {
        var pdf = s.getAttribute('data-pdf');
        if (pdf) { pdfLink.setAttribute('href', pdf); pdfLink.style.display = ''; }
        else { pdfLink.style.display = 'none'; }
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
  task = params.get("task", "")
  n = _n_compounds(output_dir)

  # Header chips: task, compounds, descriptors, headline metrics.
  chips = []
  if task:
    chips.append(f"<span class='chip'>{html.escape(task)}</span>")
  if n is not None:
    chips.append(f"<span class='chip'><b>{n:,}</b> compounds</span>")
  for m in params.get("featurizer_ids", []) or []:
    chips.append(f"<span class='chip'>{html.escape(m)}</span>")
  _, perf_rows = _read_performance(report_dir)
  pooled = next((r for r in (perf_rows or []) if r.get("model") == "pooled"), None)
  if pooled:

    def chip(key, label):
      try:
        return f"<span class='chip'>{label} <b>{float(pooled[key]):.3f}</b></span>"
      except Exception:
        return ""

    if task == "classification":
      chips += [chip("auroc", "AUROC"), chip("accuracy", "Accuracy"), chip("mcc", "MCC")]
    else:
      chips += [chip("r2", "R²")]
  chips_html = "".join(c for c in chips if c)

  # Build sections. Configuration first; then the plot categories ("Chemical space" gathers any
  # projection-* figure generically); then any leftovers.
  assigned = set()
  perf_table = _performance_table_html(report_dir)
  config_table = _config_table_html(params, n)
  cv_stats = _read_cv_stats(output_dir)
  cv_table = _cv_table_html(cv_stats)
  if cv_stats:
    with open(os.path.join(report_dir, "cv_stats.json"), "w") as f:
      json.dump({"per_descriptor": cv_stats, "summary": _cv_summary(cv_stats)}, f, indent=2)
  sections = []
  rendered_groups = set()
  poster = _overview_poster_html(report_dir, present)
  if poster:
    # The poster reuses the individual figure PNGs; they also appear full size in their own
    # sections below, so we intentionally do NOT add them to `assigned`.
    sections.append((
      "overview",
      "Overview",
      "A one-page summary of the most informative figures; each also appears full size in its "
      "section below.",
      poster,
    ))
  sections.append(("config", "Configuration", "Run settings and model identifiers.", config_table))
  for anchor, heading, desc, members in _CATEGORIES:
    if anchor == "space":
      items = sorted(s for s in stems if s.startswith("projection-"))
    elif anchor == "crossval":
      items = sorted(s for s in stems if s.startswith("cv-"))
    else:
      items = [s for s in members if s in present]
    cards = _render_items(report_dir, anchor, items, present, rendered_groups, assigned)
    grid = "<div class='grid'>" + "".join(cards) + "</div>" if cards else ""
    if anchor == "performance":
      inner = perf_table + grid
    elif anchor == "crossval":
      inner = cv_table + grid
    else:
      inner = grid
    if inner:
      sections.append((anchor, heading, desc, inner))
  leftovers = [s for s in stems if s not in assigned]
  if leftovers:
    grid = "<div class='grid'>" + "".join(_card(report_dir, s) for s in leftovers) + "</div>"
    sections.append(("more", "More", "", grid))

  nav = "".join(f"<a href='#{a}'>{html.escape(h)}</a>" for a, h, _, _ in sections)

  def _section(a, h, d, inner):
    desc = f"<p class='desc'>{html.escape(d)}</p>" if d else ""
    return f"<section id='{a}'><h2>{html.escape(h)}</h2>{desc}{inner}</section>"

  body_sections = "".join(_section(a, h, d, inner) for a, h, d, inner in sections)

  # Figure tally for the header: redundant figures collapse into one carousel panel, so the visible
  # panel count is lower than the raw figure count.
  grouped_present = {m for g in _GROUPS for m in g["members"] if m in present}
  n_groups = sum(1 for g in _GROUPS if any(m in present for m in g["members"]))
  n_panels = len(stems) - len(grouped_present) + n_groups
  fig_label = (
    f"{len(stems)} figures in {n_panels} panels"
    if n_panels != len(stems)
    else f"{len(stems)} figures"
  )

  # Footer downloads (tables live next to the page; xlsx at the model root once finish ran).
  links = []
  for fname, label in (
    ("output_table.csv", "Predictions (CSV)"),
    ("performance_table.csv", "Performance (CSV)"),
  ):
    if os.path.exists(os.path.join(report_dir, fname)):
      links.append(f"<a href='{fname}'>{label}</a>")
  if os.path.exists(os.path.join(output_dir, OUTPUT_XLSX_FILENAME)):
    links.append("<a href='../results/output.xlsx'>Predictions (XLSX)</a>")
  footer = (
    f"<div>{' · '.join(links)}</div>" if links else "<div></div>"
  ) + "<div>Generated by ZairaChem</div>"

  doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(model)} · ZairaChem report</title>
<style>{_CSS}</style>
</head>
<body>
<div class="layout">
  <aside class="sidebar">
    <div class="brand">{html.escape(model)}</div>
    <div class="brand-sub">ZairaChem report</div>
    <nav>{nav}</nav>
  </aside>
  <main class="content">
    <header>
      <h1>{html.escape(model)}</h1>
      <div class="sub">ZairaChem report · {fig_label}</div>
      <div class="chips">{chips_html}</div>
    </header>
    {body_sections}
    <footer>{footer}</footer>
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
