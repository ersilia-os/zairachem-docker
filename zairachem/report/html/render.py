"""Top-level entry point that assembles and writes ``report/report.html``."""

import glob
import html
import json
import os

from zairachem.base.vars import REPORT_SUBFOLDER
from zairachem.report.html.assets import _CSS, _JS, _about_figures_html, _semantic_css
from zairachem.report.html.cards import _card, _render_items
from zairachem.report.html.catalog import _CATEGORIES, _PREDICT_SECTION
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
  _add_space_section(sections, output_dir, report_dir, present, rendered_groups, assigned, kind)
  _add_category_sections(sections, report_dir, stems, present, rendered_groups, assigned, kind)
  _insert_diagnostics_section(sections, output_dir, report_dir, present, assigned)
  _add_closing_sections(sections, report_dir, stems, assigned)

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


def _add_lead_sections(sections, output_dir, report_dir, present, assigned, params, kind):
  """Append the Hitlist (predict only), Computational performance and Dataset sections."""
  # Predict deliverable: the top-scoring molecules, up front (right after Configuration).
  if kind == "Predict":
    hitlist = _hitlist_table_html(report_dir)
    if hitlist:
      sections.append((
        "hitlist",
        "Hitlist",
        "The highest-scoring molecules, ranked by predicted probability.",
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
    sections.append((
      "dataset",
      "Dataset",
      "Class balance and molecular property distributions of the labelled set.",
      dataset_section,
    ))


def _add_space_section(sections, output_dir, report_dir, present, rendered_groups, assigned, kind):
  """Append the Chemical space section (truth-keyed maps, or predicted-probability maps)."""
  space_section = _chemical_space_html(output_dir, report_dir, present, assigned)
  if space_section:
    sections.append((
      "space",
      "Chemical space",
      "Low-dimensional embeddings of the molecules (MW-vs-LogP and each projection).",
      space_section,
    ))
  elif kind == "Predict":
    # No truth-keyed (projection-merged-*) maps in a prediction run — gather the projections coloured
    # by predicted probability instead, so they form a proper Chemical space section (not "More").
    proba = sorted(s for s in present if s.startswith("projection-") and s.endswith("-proba"))
    cards = _render_items(report_dir, "space", proba, present, rendered_groups, assigned, kind)
    if cards:
      sections.append((
        "space",
        "Chemical space",
        "Each 2-D projection coloured by the predicted probability.",
        "<div class='grid'>" + "".join(cards) + "</div>",
      ))


def _add_category_sections(sections, report_dir, stems, present, rendered_groups, assigned, kind):
  """Append one section per plot category that has any figure (or table) to show."""
  for anchor, heading, desc, members in _CATEGORIES:
    if kind == "Predict" and anchor in _PREDICT_SECTION:
      heading, desc = _PREDICT_SECTION[anchor]
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
      "Per-descriptor lazy-qsar cross-validation (out-of-fold AUROC and overfit gap).",
      diag_section,
    )
    idx = next((i + 1 for i, s in enumerate(sections) if s[0] == "space"), len(sections))
    sections.insert(idx, entry)


def _add_closing_sections(sections, report_dir, stems, assigned):
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
    _about_figures_html(),
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
