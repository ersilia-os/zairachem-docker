"""Report section builders and the run-metadata readers (params, session, CV stats) they use."""

import csv
import glob
import html
import json
import os
from datetime import datetime

from zairachem.base.utils.logging import logger
from zairachem.base.vars import (
  DATA_FILENAME,
  DATA_SUBFOLDER,
  ESTIMATORS_SUBFOLDER,
  GITHUB_ORG,
  INPUT_SCHEMA_FILENAME,
  REPORT_SUBFOLDER,
  SESSION_FILE,
)
from zairachem.report import colors, perf
from zairachem.report.html.cards import _card, _carousel
from zairachem.report.html.catalog import (
  _COMPUTE_GROUPS,
  _DATASET_GROUPS,
  _DIAG_GROUPS,
  _PROJECTION_ORDER,
)
from zairachem.report.html.tables import (
  _cv_bars_html,
  _cv_summary,
  _cv_table_html,
  _model_table_html,
)


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
    except (OSError, ValueError) as e:
      logger.debug(f"[report] model titles not cached in params: {e}")
  return resolved


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
  # If --max-descriptors pre-screening ran, mark each featurizer kept/discarded in the table.
  selected = None
  sel_path = os.path.join(output_dir, "metadata", "selected_eos.json")
  if os.path.exists(sel_path):
    try:
      with open(sel_path) as f:
        selected = set(json.load(f))
    except Exception:
      selected = None
  feat_table = _model_table_html(feats, versions, titles, selected=selected)
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


def _oof_aupr(oof_path):
  """AUPRC (average precision) from a descriptor's out-of-fold predictions, or None."""
  try:
    from sklearn.metrics import average_precision_score

    ys, ps = [], []
    with open(oof_path) as fh:
      for row in csv.DictReader(fh):
        ys.append(int(float(row["y"])))
        ps.append(float(row["oof_proba"]))
    if len(set(ys)) < 2:
      return None
    return float(average_precision_score(ys, ps))
  except Exception:
    return None


def _read_cv_stats(output_dir):
  """Per-descriptor lazy-qsar CV reports (estimators/*/*/cv_report.json), best OOF AUROC first.

  Each report is enriched with ``aupr`` (average precision computed from the sibling oof.csv), so the
  cached cv_stats.json — and the table built from it — carries the inner-CV AUPRC alongside AUROC."""
  stats = []
  pattern = os.path.join(output_dir, ESTIMATORS_SUBFOLDER, "*", "*", "cv_report.json")
  for f in glob.glob(pattern):
    try:
      with open(f) as fh:
        rep = json.load(fh)
      rep["descriptor"] = os.path.basename(os.path.dirname(f))
      rep["aupr"] = _oof_aupr(os.path.join(os.path.dirname(f), "oof.csv"))
      stats.append(rep)
    except Exception:
      continue
  stats.sort(key=lambda r: r.get("oof_auc") if r.get("oof_auc") is not None else -1, reverse=True)
  return stats


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
