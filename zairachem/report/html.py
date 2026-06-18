"""Build a self-contained HTML report that displays the report figures, grouped by topic.

The page is deliberately plain and publication-oriented: a white background, the system font stack,
a sticky section nav, and responsive grids of figure cards (each linking to its PNG and vector PDF).
No external assets, no branding. Written to ``<output_dir>/report/report.html`` next to ``png/`` and
``pdf/`` so it opens straight from the filesystem.
"""

import csv
import glob
import html
import json
import os

from zairachem.base.vars import DATA_FILENAME, DATA_SUBFOLDER, PARAMETERS_FILE, REPORT_SUBFOLDER

# Plots grouped into sections (anchor, heading, description, member stems). Unlisted → "More".
_CATEGORIES = [
  ("dataset", "Dataset", "Composition of the training set.", ["actives-inactives"]),
  (
    "performance",
    "Model performance",
    "How the pooled model and individual estimators score on held-out predictions.",
    [
      "roc-curve",
      "confusion-matrix",
      "roc-individual",
      "raw-classification-scores",
      "r2-individual",
      "regression-raw",
      "regression-trans",
    ],
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
  "projection-mwlogp": "Molecular weight vs LogP",
  "projection-umap": "UMAP projection",
  "projection-tsne": "t-SNE projection",
  "projection-pca": "PCA projection",
}

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


def _load_params(output_dir):
  try:
    with open(os.path.join(output_dir, DATA_SUBFOLDER, PARAMETERS_FILE)) as f:
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
  rows.append((
    "Isaura store (read)",
    code(params["read_store"]) if params.get("read_store") else "off",
  ))
  rows.append((
    "Isaura store (write)",
    code(params["contribute_store"]) if params.get("contribute_store") else "off",
  ))
  rows.append(("Nearest neighbors", "on" if params.get("enable_nns") else "off"))
  vers = params.get("latest_featurizer_version") or {}
  if vers:
    rows.append((
      "Model versions",
      "  ".join(f"{code(k)} {html.escape(str(v))}" for k, v in vers.items()),
    ))
  body = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in rows)
  return f"<table class='kv'><tbody>{body}</tbody></table>"


def _card(report_dir, stem):
  title = html.escape(_humanize(stem))
  links = [f"<a href='png/{stem}.png' target='_blank'>PNG</a>"]
  if os.path.exists(os.path.join(report_dir, "pdf", f"{stem}.pdf")):
    links.append(f"<a href='pdf/{stem}.pdf' target='_blank'>PDF</a>")
  return (
    f"<figure class='card'><h3>{title}</h3>"
    f"<a class='fig' href='png/{stem}.png' target='_blank'>"
    f"<img src='png/{stem}.png' alt='{title}' loading='lazy'></a>"
    f"<div class='links'>{' · '.join(links)}</div></figure>"
  )


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
  sections = [("config", "Configuration", "Run settings and model identifiers.", config_table)]
  for anchor, heading, desc, members in _CATEGORIES:
    if anchor == "space":
      items = sorted(s for s in stems if s.startswith("projection-"))
    else:
      items = [s for s in members if s in present]
    assigned.update(items)
    grid = (
      "<div class='grid'>" + "".join(_card(report_dir, s) for s in items) + "</div>"
      if items
      else ""
    )
    inner = (perf_table + grid) if anchor == "performance" else grid
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

  # Footer downloads (tables live next to the page; xlsx at the model root once finish ran).
  links = []
  for fname, label in (
    ("output_table.csv", "Predictions (CSV)"),
    ("performance_table.csv", "Performance (CSV)"),
  ):
    if os.path.exists(os.path.join(report_dir, fname)):
      links.append(f"<a href='{fname}'>{label}</a>")
  if os.path.exists(os.path.join(output_dir, "output.xlsx")):
    links.append("<a href='../output.xlsx'>Predictions (XLSX)</a>")
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
      <div class="sub">ZairaChem report · {len(stems)} figures</div>
      <div class="chips">{chips_html}</div>
    </header>
    {body_sections}
    <footer>{footer}</footer>
  </main>
</div>
</body>
</html>
"""
  out = os.path.join(report_dir, "report.html")
  with open(out, "w") as f:
    f.write(doc)
  return out
