"""Pipeline lifecycle tracker for the ZairaChem CLI.

Drives the run as a clean, top-to-bottom stream: a run header, a themed step banner per step, and a
calm detail block + ``✓`` record when each step finishes. There is intentionally no persistent
animated panel — the only live region in a run is the live table owned by a step (see
:mod:`..live`). The :data:`tracker` singleton is shared across the Setup entrypoints
(``run_fit``/``run_predict``) and ``cli.process_group``.

The presentation primitives live in :mod:`..live` and the per-step summaries/detail in
:mod:`..summaries`; this module composes them. Re-exported from :mod:`..progress` for compatibility.
"""

import time

from rich import box
from rich.panel import Panel

from zairachem.base.utils.console import console, detail, rule, set_active_color
from zairachem.base.utils.live import _resource_caption
from zairachem.base.utils.summaries import _detail_rows, _is_predict, _PREDICT_STEP_DESC

#: Ordered pipeline steps as (key, label, description). No emojis — the checklist uses glyphs only.
PIPELINE_STEPS = [
  ("setup", "Setup", "Standardize molecules and prepare the dataset"),
  ("describe", "Describe", "Compute molecular descriptors for each model"),
  ("projections", "Projections", "Compute the 2-D projections shown in the report"),
  ("treat", "Treat", "Impute and scale the descriptor matrix"),
  ("estimate", "Estimate", "Train and cross-validate models per descriptor"),
  ("pool", "Pool", "Combine descriptor models into a consensus"),
  ("report", "Report", "Render plots, tables and the HTML report"),
  ("finish", "Finish", "Finalize outputs and clean up"),
]

#: Distinct accent colour per step — every box/line printed during a step is themed in its colour.
STEP_COLORS = {
  "setup": "cyan",
  "describe": "green",
  "projections": "bright_cyan",
  "treat": "magenta",
  "estimate": "yellow",
  "pool": "blue",
  "report": "bright_magenta",
  "finish": "bright_green",
}


class PipelineTracker:
  """Drives the pipeline display as a clean, top-to-bottom stream — no global live region.

  Each step prints a dim one-line **start marker** (:meth:`start`) and, when it finishes, a themed
  section-rule **record** plus its detail block (:meth:`complete`). Between the two, the step's own
  output streams normally. There is intentionally no persistent animated panel: the only live region
  in the whole run is the Estimate per-descriptor table, owned by the estimate monitor (fixed-height
  and isolated). This keeps the output flicker-free and correctly ordered.

  No-op until :meth:`begin`. The same markup renders fine whether or not stdout is a TTY.
  """

  def __init__(self):
    self._active = False
    self.order = []
    self.meta = {}
    self._t0 = {}
    self._current = None

  def begin(self, title, subtitle="", steps=None):
    """Print the run header and arm the tracker."""
    self.meta = {k: (lbl, desc) for k, lbl, desc in PIPELINE_STEPS}
    self.order = list(steps) if steps else [k for k, _, _ in PIPELINE_STEPS]
    self._t0 = {}
    self._current = None
    self._active = True
    body = f"[dim]{subtitle}[/]" if subtitle else "[dim]starting…[/]"
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
    return STEP_COLORS.get(self._current, "cyan") if self._current else "cyan"

  def start(self, key):
    """Open the step with a clear, themed horizontal banner (the section divider) + description."""
    if not self._active or key not in self.order:
      return
    self._current = key
    color = STEP_COLORS.get(key, "cyan")
    set_active_color(color)
    self._t0[key] = time.time()
    label, desc = self.meta.get(key, (key.title(), ""))
    if _is_predict() and key in _PREDICT_STEP_DESC:
      desc = _PREDICT_STEP_DESC[key]  # predict applies a trained model — not "train/cross-validate"
    idx = self.order.index(key) + 1
    console.print()
    # Show host CPU/RAM on every step banner (right-aligned), so resource usage is visible throughout
    # the run — not only during the steps that have a live table.
    rule(f"STEP {idx}/{len(self.order)} · {label}", style=color, right=_resource_caption())
    if desc:
      console.print(f"[dim]{desc}[/]")

  def complete(self, key, summary=None, ok=True):
    """Close the step: its (calm) detail rows then a single, modest done line."""
    if not self._active or key not in self.order:
      return
    color = STEP_COLORS.get(key, "cyan")
    elapsed = time.time() - self._t0.get(key, time.time())
    summary = summary or ""
    rows = _detail_rows(key)
    if rows:
      detail(rows, color=color)
    glyph = "✓" if ok else "✕"
    gcolor = color if ok else "red"
    tail = f"{summary} · {elapsed:.1f}s" if summary else f"{elapsed:.1f}s"
    res = _resource_caption()
    if res:
      tail += f" · {res}"
    console.print(f"  [{gcolor}]{glyph}[/] [dim]{tail}[/]")
    self._current = None

  def stop(self):
    """Disarm the tracker (kept for API symmetry; there is no live region to tear down)."""
    self._active = False


#: Process-wide tracker shared across run_fit/run_predict (Setup) and process_group (other steps).
tracker = PipelineTracker()
