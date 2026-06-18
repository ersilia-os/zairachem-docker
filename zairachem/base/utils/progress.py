"""Pipeline progress display, per-step theming, and result summaries for the ZairaChem CLI.

ZairaChem runs quiet by default (logging hidden), so the user's sense of "what is happening / which
step am I in" comes entirely from this module. The :data:`tracker` singleton, driven by the
orchestration layer (``cli.process_group`` + ``run_fit``/``run_predict``), renders:

* a **run header** panel once (title, facts, the 7-step plan as a breadcrumb);
* a sleek **section rule** per step (emoji · name · step N/7), themed in the step's colour;
* dense **detail lines** + a **✓ result line** for each step;
* a **final summary** panel with headline metrics.

Only three bordered panels appear per run (header, setup facts, final); everything else is rules,
borderless tables and aligned lines — so the output reads as one clean, information-rich stream.
Each step owns an accent colour (:data:`STEP_COLORS`); when it starts, that colour becomes the
console's active colour so every box/table printed during the step matches. The tracker is a no-op
until :meth:`PipelineTracker.begin`.
"""

import json
import os
import time

from rich import box
from rich.panel import Panel

from zairachem.base.utils.console import console, detail, rule, set_active_color, summary_panel

#: Ordered pipeline steps as (key, emoji, label).
PIPELINE_STEPS = [
  ("setup", "📋", "Setup"),
  ("describe", "🧬", "Describe"),
  ("treat", "🧪", "Treat"),
  ("estimate", "🤖", "Estimate"),
  ("pool", "🧮", "Pool"),
  ("report", "📊", "Report"),
  ("finish", "🏁", "Finish"),
]

#: Distinct accent colour per step — every box/line printed during a step is themed in its colour.
STEP_COLORS = {
  "setup": "cyan",
  "describe": "green",
  "treat": "magenta",
  "estimate": "yellow",
  "pool": "blue",
  "report": "bright_magenta",
  "finish": "bright_green",
}

_CIRCLED = "①②③④⑤⑥⑦⑧⑨"


class PipelineTracker:
  """Drives the sleek per-step display. No-op until :meth:`begin`."""

  def __init__(self):
    self._active = False
    self.order = []
    self.meta = {}
    self._t0 = {}

  def begin(self, title, subtitle="", steps=None):
    """Print the run header (title, subtitle, step-plan breadcrumb) and activate the display."""
    self.meta = {k: (e, lbl) for k, e, lbl in PIPELINE_STEPS}
    self.order = list(steps) if steps else [k for k, _, _ in PIPELINE_STEPS]
    self._t0 = {}
    self._active = True
    crumb = "   ".join(
      f"[dim]{_CIRCLED[i]} {self.meta.get(k, ('', k.title()))[1]}[/]"
      for i, k in enumerate(self.order)
    )
    body = (f"[dim]{subtitle}[/]\n\n" if subtitle else "") + crumb
    console.print(
      Panel(
        body,
        title=f"[bold white]{title}[/]",
        title_align="left",
        border_style="white",
        box=box.ROUNDED,
        padding=(1, 2),
        expand=False,
      )
    )

  @property
  def color(self):
    return STEP_COLORS.get(self._current, "cyan") if getattr(self, "_current", None) else "cyan"

  def start(self, key):
    """Theme the console to the step's colour and print its section rule."""
    if not self._active or key not in self.order:
      return
    self._current = key
    set_active_color(STEP_COLORS.get(key, "cyan"))
    self._t0[key] = time.time()
    emoji, label = self.meta.get(key, ("", key.title()))
    idx = self.order.index(key) + 1
    rule(f"{emoji}  {label}", right=f"step {idx}/{len(self.order)}")

  def complete(self, key, summary=None):
    """Print the step's detail lines (counts) and a themed ✓ result line."""
    if not self._active or key not in self.order:
      return
    color = STEP_COLORS.get(key, "cyan")
    rows = _detail_rows(key)
    if rows:
      detail(rows, color=color)
    elapsed = time.time() - self._t0.get(key, time.time())
    summary = summary or ""
    console.print(f"  [{color}]✓[/] [{color}]{summary}[/]  [dim]{elapsed:.1f}s[/]")


#: Process-wide tracker shared across run_fit/run_predict (Setup) and process_group (other steps).
tracker = PipelineTracker()


# --- Artifact resolution -----------------------------------------------------------------------


def _resolve_output_dir(output_dir=None):
  if output_dir:
    return output_dir
  from zairachem.base.vars import BASE_DIR, SESSION_FILE

  try:
    with open(os.path.join(BASE_DIR, SESSION_FILE)) as f:
      return json.load(f)["output_dir"]
  except Exception:
    return None


def _load_params(output_dir):
  from zairachem.base.vars import DATA_SUBFOLDER, PARAMETERS_FILE

  try:
    with open(os.path.join(output_dir, DATA_SUBFOLDER, PARAMETERS_FILE)) as f:
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


def _pooled_metrics(output_dir):
  try:
    import csv

    with open(os.path.join(output_dir, "performance_table.csv")) as f:
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


def summarize_treat(output_dir=None):
  d = _resolve_output_dir(output_dir)
  if not d:
    return ""
  params = _load_params(d)
  n = _n_compounds(d)
  width = _descriptor_feature_width(d, params.get("featurizer_ids", []) or [])
  base = f"{n:,} × {width:,} matrix" if (n is not None and width is not None) else "matrix imputed"
  extra = params.get("projection_ids") or []
  base += "  ·  MW/LogP projection" + (f" + {len(extra)} more" if extra else "")
  return base


def summarize_estimate(output_dir=None):
  d = _resolve_output_dir(output_dir)
  if not d:
    return ""
  algos = _estimator_algorithms(d)
  return _plurals(len(algos), "algorithm") + " trained" if algos else "estimators trained"


def summarize_pool(output_dir=None):
  d = _resolve_output_dir(output_dir)
  if not d:
    return ""
  algos = _estimator_algorithms(d)
  return f"consensus of {_plurals(len(algos), 'algorithm')}" if algos else "predictions pooled"


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
  "treat": summarize_treat,
  "estimate": summarize_estimate,
  "pool": summarize_pool,
  "report": summarize_report,
  "finish": summarize_finish,
}


# --- Per-step detail lines (counts), printed borderless under the rule --------------------------


def _detail_rows(key, output_dir=None):
  """Dense (label, value) detail lines for the info-rich silent steps; [] otherwise."""
  d = _resolve_output_dir(output_dir)
  if not d:
    return []
  if key == "estimate":
    algos = _estimator_algorithms(d)
    n = _n_compounds(d)
    rows = []
    if algos:
      rows.append(("algorithms", "  ".join(f"[bold]{a}[/]" for a in algos)))
    if n is not None:
      rows.append(("trained on", f"[bold]{n:,}[/] compounds"))
    return rows
  if key == "report":
    from zairachem.base.vars import REPORT_SUBFOLDER

    n = _count_plots(d)
    outs = [f for f in ("output.csv", "output.xlsx") if os.path.exists(os.path.join(d, f))]
    rows = []
    if n:
      rows.append(("plots", f"[bold]{n}[/] [dim]· png + pdf[/]"))
    html_path = os.path.join(d, REPORT_SUBFOLDER, "report.html")
    if os.path.exists(html_path):
      rows.append(("html", f"[bold]{_collapse(html_path)}[/]"))
    if outs:
      rows.append(("outputs", "  ".join(outs)))
    return rows
  return []


def final_summary_panel(output_dir=None):
  """Render the closing run-summary panel: headline metrics + output location."""
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
