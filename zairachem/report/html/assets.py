"""Static page assets (CSS, JS, reference-grid SVG, colour key) and figure image/size helpers."""

import html
import os

from zairachem.report import (
  CELL_CM as _CELL_CM,
  GRID_COLS as _GRID_COLS,
  GRID_ROWS as _GRID_ROWS,
  colors as _colors,
)


def _img_src(report_dir, stem):
  """Relative path to ``png/{stem}.png``.

  The report references the figure files directly (kept alongside the HTML in ``png/``) instead of
  base64-inlining them, which keeps ``report.html`` small rather than duplicating every PNG into it.
  The page is therefore not standalone: it must travel together with its ``png/`` folder.
  """
  return f"png/{stem}.png"


# --- Physical figure sizes & the composite reference grid ----------------------------------------
# The report documents a reference grid (mirrors the old 6×10 matplotlib poster grid) so users can
# tile the downloaded plots into a composite figure at the right scale. A card's footprint badge is
# the plot's DECLARED cells (rows, cols) — persisted to report/figure_cells.json at render time — not
# reverse-engineered from the (tight-cropped) PNG, which is unreliable.
_CELL_IN = _CELL_CM / 2.54  # ≈ 1.181 in


_FIGURE_CELLS_CACHE = {}


def _load_figure_cells(report_dir):
  """``{stem: (rows, cols)}`` declared footprints written by the report render; cached per report dir."""
  cached = _FIGURE_CELLS_CACHE.get(report_dir)
  if cached is not None:
    return cached
  import json

  data = {}
  try:
    with open(os.path.join(report_dir, "figure_cells.json")) as f:
      raw = json.load(f)
    data = {
      k: (int(v[0]), int(v[1]))
      for k, v in raw.items()
      if isinstance(v, (list, tuple)) and len(v) == 2
    }
  except Exception:
    data = {}
  _FIGURE_CELLS_CACHE[report_dir] = data
  return data


def _figure_size(report_dir, stem):
  """Physical size + grid footprint of a figure from its declared ``cells`` (rows, cols), recorded at
  render time. Returns ``{w_cm, h_cm, w_in, h_in, cols, rows}``, or None if the footprint is unknown."""
  cells = _load_figure_cells(report_dir).get(stem)
  if cells is None:
    return None
  rows, cols = cells
  w_cm, h_cm = cols * _CELL_CM, rows * _CELL_CM
  return {
    "w_cm": w_cm,
    "h_cm": h_cm,
    "w_in": w_cm / 2.54,
    "h_in": h_cm / 2.54,
    "cols": cols,
    "rows": rows,
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


def _color_key_html(task="classification"):
  """The transversal color key shared by every plot, one row per family (``colors.legend_groups``)."""
  rows = []
  for family, entries in _colors.legend_groups(task):
    items = "".join(
      f"<span class='item'><span class='swatch' style='background:{_colors.hexcol(key)}'></span>"
      f"{html.escape(label)}</span>"
      for label, key in entries
    )
    rows.append(
      f"<div class='legend-key'><span class='fam'>{html.escape(family)}</span>{items}</div>"
    )
  return "".join(rows)


def _about_figures_html(task="classification"):
  """Show the composite reference grid (full width), a short caption, and the shared color key."""
  cap = (
    f"<p class='grid-cap'>Reference grid: <b>{_GRID_ROWS} rows × {_GRID_COLS} columns</b>, one cell "
    f"<b>{_CELL_CM:.0f} cm ({_CELL_IN:.2f} in)</b> square "
    f"(full sheet {_GRID_COLS * _CELL_CM:.0f} cm wide × {_GRID_ROWS * _CELL_CM:.0f} cm tall). Each "
    "plot card's <b>n×m</b> badge is its footprint in cells (<b>rows × columns</b>) — download the "
    "plots and tile them to compose a figure at the right scale.</p>"
  )
  key_caption = "<p class='grid-cap'>Colors are consistent across every plot:</p>"
  return f"<div class='about'>{_grid_svg()}{cap}{key_caption}{_color_key_html(task)}</div>"


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
    ".about .legend-key .fam{color:#6e7781;font-size:12px;letter-spacing:.04em;"
    "text-transform:uppercase;}"
    ".about .swatch{width:13px;height:13px;border-radius:3px;display:inline-block;"
    "border:1px solid rgba(0,0,0,.08);}"
  )
  return ph + prov + badges + key


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
.table-wrap + .grid { margin-top:26px; }
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
.carousel-head h3 { margin:0; flex:0 1 auto; min-width:0; }
.carousel-label { color:var(--muted); font-size:12.5px; white-space:nowrap; flex:0 1 auto; min-width:0; overflow:hidden; text-overflow:ellipsis; text-align:right; }
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
/* A SMILES has no width limit of its own, and a long one pushed the hitlist's value,
   interval and similarity past the right edge of the card. Clipped with an ellipsis; the
   full string is the cell's title. */
table.metrics th.smiles, table.metrics td.smiles { text-align:left; max-width:32ch; overflow:hidden; text-overflow:ellipsis; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12px; }
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
    show(0);  // sync badge/label/links to the first slide from the JS (single source of truth)
  });
});
"""
