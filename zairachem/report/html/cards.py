"""Figure cards and carousels, and the generic renderer that lays a section's figures out."""

import html
import os

from zairachem.report.html.assets import _dim_badge, _figure_dims, _figure_size, _img_src
from zairachem.report.html.catalog import _PREDICT_GROUP_TITLE, _STEM_TO_GROUP, _humanize

# Overview poster layout: rows of (figure stem, relative width). Mirrors the old 6×10 matplotlib
# grid, but assembled in HTML from the individual figure PNGs. Cells whose PNG is missing are
# dropped; an all-missing row is skipped. Row widths are proportional (flex-grow).


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


def _render_items(report_dir, anchor, items, present, rendered_groups, assigned, kind=None):
  """Render a section's items, collapsing grouped figures into one carousel at their home section.

  Grouped stems seen outside their home section (or after their group already rendered) are skipped;
  they surface in the carousel emitted at the home section instead. ``assigned`` tracks everything
  rendered so the trailing "More" section can pick up genuine leftovers. ``kind`` ("Predict"/"Fit")
  selects the predict-aware carousel title overrides.
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
      if kind == "Predict" and group["key"] in _PREDICT_GROUP_TITLE:
        group = {**group, "title": _PREDICT_GROUP_TITLE[group["key"]]}
      cards.append(_carousel(report_dir, group, group_members))
  return cards
