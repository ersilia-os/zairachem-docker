"""Top-level entry point that assembles and writes ``report/report.html``."""

import glob
import html
import json
import os

from zairachem.base.vars import REPORT_SUBFOLDER
from zairachem.report.html.assets import _CSS, _JS, _about_figures_html, _semantic_css
from zairachem.report.html.cards import _card, _render_items
from zairachem.report.html.catalog import (
  _CATEGORIES,
  _PREDICT_SECTION,
  _PREDICT_SECTION_NO_TRUTH,
)
from zairachem.report.html.sections import (
  _chemical_space_html,
  _computational_performance_html,
  _config_section_html,
  _dataset_html,
  _diagnostics_html,
  _load_params,
  _n_compounds,
  _read_cv_stats,
  _run_date,
  _run_mode,
  _scalar_space_html,
)
from zairachem.report.html.tables import _cv_summary, _hitlist_table_html, _validation_table_html


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
  config_table = _config_section_html(output_dir, params, n)
  _write_cv_stats(output_dir, report_dir)
  sections = []
  rendered_groups = set()
  # (The Overview poster is intentionally omitted for now.)
  sections.append(("config", "Configuration", "Run settings and model identifiers.", config_table))
  _add_lead_sections(sections, output_dir, report_dir, present, assigned, params, kind)
  _add_space_section(sections, output_dir, report_dir, present, assigned)
  _add_category_sections(
    sections, report_dir, stems, present, rendered_groups, assigned, kind, _has_truth(output_dir)
  )
  _insert_diagnostics_section(sections, output_dir, report_dir, present, assigned)
  _add_closing_sections(sections, report_dir, stems, assigned, params.get("task", "classification"))

  doc = _page_html(kind, model, report_heading, subtitle, tab_title, sections, report_dir)
  out = os.path.join(report_dir, "report.html")
  with open(out, "w") as f:
    f.write(doc)
  return out


def _write_cv_stats(output_dir, report_dir):
  """Write ``report/cv_stats.json`` (per-descriptor inner-CV stats plus summary) when available."""
  cv_stats = _read_cv_stats(output_dir)
  if cv_stats:
    with open(os.path.join(report_dir, "cv_stats.json"), "w") as f:
      json.dump({"per_descriptor": cv_stats, "summary": _cv_summary(cv_stats)}, f, indent=2)


def _has_truth(output_dir):
  """True when this run's ``data.csv`` carries the observed values of its task."""
  from zairachem.base.vars import DATA_FILENAME, DATA_SUBFOLDER, TARGET_COLUMN

  try:
    import pandas as pd

    cols = set(
      pd.read_csv(os.path.join(output_dir, DATA_SUBFOLDER, DATA_FILENAME), nrows=0).columns
    )
  except Exception:
    return False
  return any(target in cols for target in TARGET_COLUMN.values())


def _add_lead_sections(sections, output_dir, report_dir, present, assigned, params, kind):
  """Append the Hitlist (predict only), Computational performance and Dataset sections."""
  # Predict deliverable: the top-scoring molecules, up front (right after Configuration).
  if kind == "Predict":
    hitlist = _hitlist_table_html(report_dir)
    if hitlist:
      sections.append((
        "hitlist",
        "Hitlist",
        "Ranked by predicted value, highest first; which end is the useful one "
        "depends on the target."
        if params.get("task") == "regression"
        else "The highest-scoring molecules, ranked by predicted probability.",
        hitlist,
      ))
  perf_section = _computational_performance_html(output_dir, report_dir, present, assigned, params)
  if perf_section:
    sections.append((
      "compute",
      "Computational performance",
      "Run time, memory/CPU usage, and where the descriptors came from.",
      perf_section,
    ))
  dataset_section = _dataset_html(output_dir, report_dir, present, assigned)
  if dataset_section:
    if params.get("task") != "regression":
      desc = "Class balance and molecular property distributions of the labelled set."
    elif _has_truth(output_dir):
      desc = "Spread of the target value and how it relates to molecular properties."
    else:
      # Unlabelled molecules: the panels plot the predicted value, and there is no spread of
      # observed values to describe.
      desc = "How the predicted values relate to molecular properties."
    sections.append(("dataset", "Dataset", desc, dataset_section))


def _add_space_section(sections, output_dir, report_dir, present, assigned):
  """Append the Chemical space section, however this run is able to colour its projections.

  A labelled classification run colours by true class; otherwise the maps are coloured by one
  scalar -- the target value (regression) or the predicted probability (predict without truth).
  Every run that has projections at all therefore gets a Chemical space section rather than
  leaving the maps to the "More" catch-all.

  The builders are tried in order and the first to produce anything wins. They are called lazily
  because each marks the stems it used as assigned, and a stem assigned by a builder whose output
  is then discarded would disappear from the report altogether.
  """
  for build, desc in (
    (
      lambda: _chemical_space_html(output_dir, report_dir, present, assigned),
      "Low-dimensional embeddings of the molecules (MW-vs-LogP and each projection).",
    ),
    (
      lambda: _scalar_space_html(report_dir, present, assigned, "-value"),
      "Each 2-D projection coloured by the target value.",
    ),
    (
      lambda: _scalar_space_html(report_dir, present, assigned, "-proba"),
      "Each 2-D projection coloured by the predicted probability.",
    ),
  ):
    section = build()
    if section:
      sections.append(("space", "Chemical space", desc, section))
      return


def _add_category_sections(
  sections, report_dir, stems, present, rendered_groups, assigned, kind, has_truth=True
):
  """Append one section per plot category that has any figure (or table) to show.

  At predict a couple of headings shift meaning, and shift again when the molecules carry no
  observed values: there is no test-set performance to report on an unlabelled set.
  """
  overrides = _PREDICT_SECTION if has_truth else _PREDICT_SECTION_NO_TRUTH
  for anchor, heading, desc, members in _CATEGORIES:
    if kind == "Predict" and anchor in overrides:
      heading, desc = overrides[anchor]
    if anchor == "space":
      items = sorted(s for s in stems if s.startswith("projection-"))
    else:
      items = [s for s in members if s in present]
    cards = _render_items(report_dir, anchor, items, present, rendered_groups, assigned, kind)
    grid = "<div class='grid'>" + "".join(cards) + "</div>" if cards else ""
    if anchor == "validation":
      inner = _validation_table_html(report_dir) + grid
    else:
      inner = grid
    if inner:
      sections.append((anchor, heading, desc, inner))


def _insert_diagnostics_section(sections, output_dir, report_dir, present, assigned):
  """Insert the per-descriptor inner diagnostics section right after the Chemical space."""
  # Per-descriptor inner diagnostics (inner lazy-qsar CV): inserted right after the chemical space.
  diag_section = _diagnostics_html(output_dir, report_dir, present, assigned)
  if diag_section:
    entry = (
      "diagnostics",
      "Per descriptor inner diagnostics",
      "Per-descriptor lazy-qsar cross-validation (out-of-fold skill and overfit gap).",
      diag_section,
    )
    idx = next((i + 1 for i, s in enumerate(sections) if s[0] == "space"), len(sections))
    sections.insert(idx, entry)


def _add_closing_sections(sections, report_dir, stems, assigned, task="classification"):
  """Append the More (unassigned figures) and About the figures sections."""
  leftovers = [s for s in stems if s not in assigned]
  if leftovers:
    grid = "<div class='grid'>" + "".join(_card(report_dir, s) for s in leftovers) + "</div>"
    sections.append(("more", "More", "", grid))

  # About the figures: kept last — reference material on plot sizes and the composite grid.
  sections.append((
    "about",
    "About the figures",
    "The physical size of each plot and the reference grid for assembling a composite figure.",
    _about_figures_html(task),
  ))


def _footer_html(report_dir):
  """Return the sidebar footer: table download links plus the Ersilia / ZairaChem credit."""
  # Footer downloads (tables live next to the page; xlsx at the model root once finish ran).
  links = []
  for fname, label in (
    ("output_table.csv", "Predictions (CSV)"),
    ("performance_table.csv", "Performance (CSV)"),
  ):
    if os.path.exists(os.path.join(report_dir, fname)):
      links.append(f"<a href='{fname}'>{label}</a>")
  return (f"<div>{' · '.join(links)}</div>" if links else "<div></div>") + (
    "<div>Brought to you by the "
    "<a href='https://github.com/ersilia-os/ersilia'>Ersilia Open Source Initiative</a>. "
    "Produced with <a href='https://github.com/ersilia-os/zaira-chem'>ZairaChem</a>.</div>"
  )


def _page_html(kind, model, report_heading, subtitle, tab_title, sections, report_dir):
  """Assemble the full report HTML document from its header strings and sections."""
  nav = "".join(f"<a href='#{a}'>{html.escape(h)}</a>" for a, h, _, _ in sections)

  def _section(a, h, d, inner):
    desc = f"<p class='desc'>{html.escape(d)}</p>" if d else ""
    return f"<section id='{a}'><h2>{html.escape(h)}</h2>{desc}{inner}</section>"

  body_sections = "".join(_section(a, h, d, inner) for a, h, d, inner in sections)
  footer = _footer_html(report_dir)

  return f"""<!doctype html>
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
