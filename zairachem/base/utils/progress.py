"""Pipeline progress display, per-step theming, and result summaries for the ZairaChem CLI.

ZairaChem runs quiet by default (logging hidden), so the user's sense of "what is happening / which
step am I in" comes entirely from this module. The :data:`tracker` singleton, driven by the
orchestration layer (``cli.process_group`` + ``run_fit``/``run_predict``), renders, as **plain
sequential output** (no global live region):

* a **run header** panel once (title + subtitle);
* a clear, themed **step banner** when each step begins — a full-width horizontal section rule
  (``──── STEP 2/7 · Describe ────``) plus a dim description; then the step's own output streams
  below it;
* the step's (deliberately calm — minimal colour/bold) **detail rows** and a single modest
  ``✓ done`` line when it finishes;
* a **final summary** panel with headline metrics.

The structure carries the emphasis (themed banners and rules); the inner content stays quiet and
readable, so the eye lands on the step boundaries, not on speckled bold text.

Deliberately there is **no persistent, animated panel** spanning the run: a long-lived multi-line
``Live`` region that other code prints over flickers and leaves misplaced lines, so the pipeline
output is just a clean top-to-bottom stream. The *only* animated region is the Estimate step's
per-descriptor table (see :mod:`...lazy_qsar.monitor`), which is fixed-height and fully isolated
(lazy-qsar's logs are captured while it runs, so nothing prints over it) — the one place a live
view is both safe and worth it.

Bordered panels appear only for the header and final summary; everything else is thin section rules,
borderless tables and aligned lines — no emojis. Each step owns an accent colour
(:data:`STEP_COLORS`); when it starts, that colour becomes the console's active colour so every
box/table printed during the step matches. The tracker is a no-op until :meth:`PipelineTracker.begin`.
"""

import contextlib
import json
import os
import threading
import time

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from zairachem.base.utils.console import (
  active_color,
  console,
  detail,
  echo,
  rule,
  set_active_color,
  summary_panel,
)

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

#: Steps that render a persistent live per-item table (see :class:`LiveTableMonitor`). That table
#: stays on screen as the step's record, so these steps intentionally print no separate detail block.
_LIVE_TABLE_STEPS = {"describe", "projections", "treat", "estimate"}


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


# --- Live per-item table (shared progress monitor) ---------------------------------------------

#: Braille spinner frames; indexed by wall-clock so a running row visibly animates between refreshes
#: even when its determinate progress can't advance (e.g. a single model-server request in flight).
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _spinner_frame():
  return _SPINNER_FRAMES[int(time.time() * 10) % len(_SPINNER_FRAMES)]


def _resource_caption():
  """Host CPU + RAM usage as a compact one-liner, or None if psutil isn't available.

  Host-level (not per-container) on purpose: the model servers run inside Docker Desktop's VM, whose
  load shows up as host CPU/RAM — which is exactly the "is my machine maxed out" signal wanted. Cheap
  and non-blocking (``cpu_percent(interval=None)`` reads the rolling value since the last call)."""
  try:
    import psutil
  except Exception:
    return None
  try:
    cpu = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    return f"CPU {cpu:.0f}%  ·  RAM {vm.used / 1e9:.1f}/{vm.total / 1e9:.1f} GB ({vm.percent:.0f}%)"
  except Exception:
    return None


class LiveProgressBar:
  """A single-line, in-place progress bar for a step that is a flat sequence of N items and doesn't
  warrant a per-item table. Shares the live-table aesthetic — a themed title, the same mini
  :func:`_bar`, a ``done/total`` count and a dim trailing note (the current item) — but stays one calm
  line instead of N animated rows. Transient while running; prints one final summary line on exit as
  the step's record. Off-TTY it degrades to a single terse completion line (no animation).
  """

  def __init__(
    self, title, total, color="cyan", width=24, persist=True, discrete=False, show_bar=True
  ):
    self.title = title
    self.total = max(0, int(total))
    self.color = color
    self.width = width
    self.persist = persist
    # Discrete: a calm, unemphasised line — no bold, a percentage instead of a done/total count, and
    # the title shown dim (as the "activity"). Used for the Setup sub-steps.
    self.discrete = discrete
    # When False, show just the percentage (no bar glyph) — used for the Report step.
    self.show_bar = show_bar
    self.done = 0
    self.note = ""
    self._live = None
    self._plain = False

  def _render(self, frac, spin=True):
    if self.discrete:
      label = self.title + (f" [dim]·[/] {self.note}" if self.note else "")
      bar = f"{_bar(frac, self.width)} " if self.show_bar else ""
      return Text.from_markup(f"{bar}[dim]{frac * 100:>3.0f}%[/]  [dim]{label}[/]")
    head = "" if (not spin or self.done >= self.total) else f"{_spinner_frame()} "
    note = f"  [dim]{self.note}[/]" if self.note else ""
    return Text.from_markup(
      f"[bold {self.color}]{head}{self.title}[/]  {_bar(frac, self.width)} "
      f"[dim]{self.done}/{self.total}[/]{note}"
    )

  def __rich__(self):
    return self._render((self.done / self.total) if self.total else 1.0)

  def advance(self):
    """Mark one item complete and redraw."""
    self.done = min(self.total, self.done + 1)
    self._refresh()

  def set_note(self, note):
    """Set the dim trailing note (e.g. the item currently being worked) and redraw."""
    self.note = note or ""
    self._refresh()

  def _refresh(self):
    # No refresh=True: these bars can advance thousands of times in a tight loop, so we just swap the
    # renderable and let Live's auto-refresh thread redraw at its capped rate (forcing a redraw per
    # advance would be its own slowdown). The spinner still animates on each auto-refresh tick.
    live = self._live
    if live is not None:
      live.update(self)

  @contextlib.contextmanager
  def live(self):
    if not console.is_terminal:
      self._plain, self._live = True, None
      try:
        yield self
      finally:
        if self.persist:
          echo(f"{self.title}: {self.done}/{self.total}", kind="info")
      return
    from rich.live import Live

    try:
      with Live(
        self, console=console, transient=True, refresh_per_second=8, auto_refresh=True
      ) as live:
        self._live = live
        yield self
    finally:
      self._live = None
      # Optionally leave one calm summary line (full bar) as the step's record.
      if self.persist:
        console.print(self._render(1.0, spin=False))


class SetupProgress:
  """Calm single-line determinate progress for the Setup sub-steps, in the shared bar style.

  A drop-in for the old ``rich.Progress`` usage so call sites need not change::

      with SetupProgress() as p:
        t = p.add_task("Standardizing molecules", total=n)
        for ...:
          p.update(t, advance=1)            # optionally advance=k / description="…"

  Renders a discrete :class:`LiveProgressBar` (a plain bar + dim percentage + the operation name shown
  dim as the activity — no bold, no count) instead of the old multi-column *pulsing* bar, which both
  blinked and clashed with the rest of the UI. The final bar persists as the sub-step's record (like
  the other steps), rather than vanishing. The optional ``description`` on :meth:`update` is appended as
  the dim sub-phase (e.g. the merge phase).
  """

  def __init__(self, color=None, width=24):
    self.color = color or active_color()
    self.width = width
    self._bar = None
    self._cm = None

  def __enter__(self):
    return self

  def add_task(self, description, total=0):
    self._bar = LiveProgressBar(
      description, total=total, color=self.color, width=self.width, persist=True, discrete=True
    )
    self._cm = self._bar.live()
    self._cm.__enter__()
    return 0  # single-task token; these setup loops only ever track one task at a time

  def update(self, task, advance=0, description=None):
    if self._bar is None:
      return
    if description:
      self._bar.set_note(description)
    if advance:
      self._bar.done = min(self._bar.total, self._bar.done + advance)
      self._bar._refresh()

  def __exit__(self, *exc):
    if self._cm is not None:
      self._cm.__exit__(*exc)
      self._cm = None
    return False


class LiveTableMonitor:
  """A transient, fixed-height live table — one row per work item — shared across pipeline steps.

  This is the generalisation of the Estimate step's per-descriptor table (the one consistent,
  isolated live region in the run). Each item moves ``queued → running:<substep> → done/skipped``
  with result columns and a wall-clock; the table animates in place and clears at the end, leaving
  the durable per-step record (``_detail_rows``) as the permanent output.

  Subclasses supply the per-step presentation by setting :attr:`title`, :attr:`item_label`,
  :attr:`running_verb` and overriding :meth:`_columns`/:meth:`_row_cells` (the result columns after
  ``Item`` and ``Status``). The lifecycle/threading/off-TTY machinery lives here.

  All state mutations and refreshes are guarded by a re-entrant lock, so worker threads (e.g. the
  parallel Describe pool) can each drive their own row of one shared table safely.

  Parameters
  ----------
  item_ids : list of str
      Work items, in display order. Rows are created up front (all ``queued``) so the table is
      fixed-height and never grows mid-run.
  color : str
      Accent colour (the owning step's colour).
  result_reader : callable, optional
      ``item_id -> result`` consulted on successful :meth:`finish` to fill the row from persisted
      artifacts. Its return value is merged via :meth:`_store_result` (default: a dict updates the
      item's ``extra``). Best-effort; exceptions are swallowed.
  """

  #: First-column header.
  item_label = "Item"
  #: Table title.
  title = "Working"
  #: Status verb shown while an item is running (e.g. "training", "featurizing").
  running_verb = "running"
  #: Fixed width of the Status column. Status text varies as work progresses (substep, spinner); a
  #: fixed width stops it pushing later columns around. Truncated with an ellipsis if longer.
  status_width = 28
  #: Opt in (subclass) to a trailing dim "Activity" column that streams the latest background line
  #: (set via :meth:`set_activity`) for the running row — e.g. lazy-qsar's raw loguru output.
  show_activity = False
  #: Approximate combined width of all non-activity columns (item + status + result columns + padding).
  #: The activity column takes the remaining terminal width; if too little is left, it's dropped. A
  #: slight overestimate is safe (activity a touch narrower) — an underestimate risks the table being
  #: cropped on the right. Subclasses that opt into activity should tune this.
  reserved_width = 80
  #: Minimum width below which the activity column is hidden rather than shown uselessly narrow.
  _min_activity_width = 18

  def __init__(self, item_ids, color="cyan", result_reader=None):
    self.color = color
    self.result_reader = result_reader
    self.order = list(item_ids)
    self.state = {i: self._new_state() for i in self.order}
    self._active = None
    self._live = None  # this step's own isolated, fixed-height live region; None until live()
    self._plain = False  # True off-TTY: emit terse start/finish lines instead of a live table
    self._lock = threading.RLock()  # guards state + refresh so parallel drivers are safe
    # Fixed Item-column width (longest id/header) so the first column never resizes mid-run.
    self._item_width = max([len(self.item_label), *(len(str(i)) for i in self.order)] or [4])

  @staticmethod
  def _new_state():
    return {"status": "queued", "t0": None, "elapsed": None, "substep": "", "extra": {}}

  def _idx(self, item_id):
    return self.order.index(item_id) + 1 if item_id in self.order else 0

  # --- state transitions (all lock-guarded) ---

  def start(self, item_id):
    with self._lock:
      self._active = item_id
      s = self.state.get(item_id)
      if s is not None:
        s.update(status="running", t0=time.time(), substep="")
      plain = self._plain
    if plain:
      echo(
        f"{self.running_verb} {item_id} [dim]({self._idx(item_id)}/{len(self.order)})[/]",
        kind="run",
      )
    self._refresh()

  def set_substep(self, item_id, text):
    if not text:
      return
    with self._lock:
      s = self.state.get(item_id)
      if s is not None:
        s["substep"] = text
    self._refresh()

  def update_fields(self, item_id, **fields):
    """Merge arbitrary result fields into an item's ``extra`` (read by :meth:`_row_cells`)."""
    with self._lock:
      s = self.state.get(item_id)
      if s is not None:
        s["extra"].update(fields)
    self._refresh()

  def set_activity(self, item_id, text):
    """Set the latest background line shown (dim) in the trailing Activity column for ``item_id``.

    No-op unless the subclass sets :attr:`show_activity`. The text is the raw line from whatever the
    item is doing in the background (e.g. a lazy-qsar loguru message); it is stored verbatim and
    escaped/truncated only at render time.
    """
    if not self.show_activity:
      return
    with self._lock:
      s = self.state.get(item_id)
      if s is not None:
        s["extra"]["activity"] = text
    self._refresh()

  def finish(self, item_id, ok=True):
    with self._lock:
      s = self.state.get(item_id)
      if s is not None:
        s["status"] = "done" if ok else "skipped"
        s["elapsed"] = (time.time() - s["t0"]) if s["t0"] else None
        s["substep"] = ""
        if ok and self.result_reader:
          with contextlib.suppress(Exception):
            self._store_result(s, self.result_reader(item_id))
      if self._active == item_id:
        self._active = None
      plain = self._plain
      snapshot = dict(s) if s is not None else None
    if plain:
      self._plain_finish(item_id, snapshot)
    self._refresh()

  def _store_result(self, s, result):
    """Merge a ``result_reader`` return value into the item state. Default: a dict updates extra."""
    if isinstance(result, dict):
      s["extra"].update(result)

  def _plain_finish(self, item_id, s):
    """Terse off-TTY completion line. Subclasses may override to include result columns."""
    status = s["status"] if s else "done"
    el = f"{s['elapsed']:.1f}s" if s and s["elapsed"] is not None else "—"
    echo(f"{status} {item_id}  [dim]{el}[/]", kind="info")

  # --- rendering ---

  def _elapsed(self, s):
    if s["elapsed"] is not None:
      return s["elapsed"]
    if s["status"] == "running" and s["t0"]:
      return time.time() - s["t0"]
    return None

  def _fmt_time(self, s):
    el = self._elapsed(s)
    return f"{el:.1f}s" if el is not None else "[dim]—[/]"

  def _status_cell(self, s):
    st = s["status"]
    if st == "queued":
      return "[dim]queued[/]"
    if st == "running":
      sub = f" [dim]·[/] {s['substep']}" if s["substep"] else ""
      return f"[{self.color}]{_spinner_frame()} {self.running_verb}[/]{sub}"
    if st == "skipped":
      return "[red]skipped[/]"
    return "[green]✓ done[/]"

  def _columns(self, table):
    """Add the result columns that follow ``Item`` and ``Status``. Default: just ``Time``."""
    table.add_column("Time", justify="right", width=8, no_wrap=True)

  def _row_cells(self, item_id, s):
    """Cells (in column order) for the columns added by :meth:`_columns`. Default: just time."""
    return [self._fmt_time(s)]

  def _activity_width(self):
    """Width for the trailing Activity column: whatever terminal space is left after the other
    columns. Returns 0 (column hidden) when the subclass hasn't opted in or too little space remains."""
    if not self.show_activity:
      return 0
    avail = console.width - self.reserved_width
    return avail if avail >= self._min_activity_width else 0

  def _activity_cell(self, s):
    """The dim background line for the running row (blank otherwise). Escaped + relying on the
    column's no_wrap/ellipsis to truncate, so a long or markup-bearing line can't break the layout."""
    from rich.markup import escape

    txt = s["extra"].get("activity") if s["status"] == "running" else None
    return f"[dim]{escape(str(txt))}[/]" if txt else ""

  def __rich__(self):
    # No caption: the table height must stay constant across refreshes, otherwise a transient Live
    # miscounts lines when clearing (a blinking CPU/RAM caption left a doubled title/header behind on
    # fast, frequently-toggling steps). Host CPU/RAM is shown on the step banner and the ✓ done line.
    table = Table(
      title=f"[bold {self.color}]{self.title}[/]",
      title_justify="left",
      box=None,
      pad_edge=False,
      padding=(0, 2),
      expand=False,
    )
    # Fixed widths + no_wrap throughout so columns never reflow as cell contents change length.
    # Item text is constant per row, so a min_width floor keeps the column stable without ever
    # truncating an id (a fixed width would ellipsize ids exactly as long as the width).
    table.add_column(self.item_label, style="bold", min_width=self._item_width, no_wrap=True)
    table.add_column("Status", width=self.status_width, no_wrap=True, overflow="ellipsis")
    self._columns(table)
    activity_w = self._activity_width()
    if activity_w:
      table.add_column("Activity", width=activity_w, no_wrap=True, overflow="ellipsis")
    with self._lock:
      for m in self.order:
        s = self.state[m]
        cells = list(self._row_cells(m, s))
        if activity_w:
          cells.append(self._activity_cell(s))
        table.add_row(m, self._status_cell(s), *cells)
    return table

  def _refresh(self):
    # Do NOT hold self._lock here: Live.update() acquires rich's own refresh lock and then calls
    # __rich__() (which takes self._lock). Rich's auto-refresh thread takes those two locks in the
    # same order; holding self._lock across update() would invert the order and deadlock. State
    # mutators already release self._lock before calling _refresh, so the only nesting is
    # rich-lock → self._lock, consistently, in both paths.
    live = self._live
    if live is not None:
      live.update(self, refresh=True)

  # --- lifecycle hooks (overridden e.g. to attach a loguru sink) ---

  def _on_enter(self):
    pass

  def _on_exit(self):
    pass

  @contextlib.contextmanager
  def live(self):
    """Animate this table in place (TTY) or degrade to terse start/finish lines (off-TTY).

    The animated region is **transient** — it redraws cleanly while the table's height changes (the
    CPU/RAM caption and spinner come and go) with no stale-line artifacts. On exit, the final table is
    printed once as a static block, so it persists on screen as the step's record (not cleared, not
    replaced by a separate summary). Only one ``Live`` region may be active at a time — don't nest.
    """
    self._on_enter()
    # Prime the rolling CPU sampler so the first rendered caption shows a real value, not 0%.
    with contextlib.suppress(Exception):
      import psutil

      psutil.cpu_percent(interval=None)
    if not console.is_terminal:
      # Non-interactive (e.g. piped output): no live region; terse start/finish lines instead.
      self._live = None
      self._plain = True
      try:
        yield self
      finally:
        self._on_exit()
      return

    from rich.live import Live

    try:
      with Live(
        self, console=console, transient=True, refresh_per_second=8, auto_refresh=True
      ) as live:
        self._live = live
        yield self
    finally:
      self._live = None
      self._on_exit()
      # Leave the final table on screen as the permanent record (the live region itself was transient
      # to avoid height-change redraw artifacts while animating).
      console.print(self)


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


def _is_predict():
  """True if the active session is a prediction run (so step text can say 'apply', not 'train')."""
  from zairachem.base.vars import BASE_DIR, SESSION_FILE

  try:
    with open(os.path.join(BASE_DIR, SESSION_FILE)) as f:
      return json.load(f).get("mode") == "predict"
  except Exception:
    return False


#: Step descriptions that differ at predict (the run applies a trained model rather than building one).
_PREDICT_STEP_DESC = {
  "estimate": "Apply the trained models to score each molecule",
  "treat": "Apply the fitted transformers to the descriptor matrix",
}


def _load_params(output_dir):
  from zairachem.base.vars import METADATA_SUBFOLDER, PARAMETERS_FILE

  try:
    with open(os.path.join(output_dir, METADATA_SUBFOLDER, PARAMETERS_FILE)) as f:
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


def _reliability_summary(output_dir):
  """The reliability pooler's run summary (pool/reliability_summary.json), or None."""
  from zairachem.base.vars import POOL_SUBFOLDER

  try:
    with open(os.path.join(output_dir, POOL_SUBFOLDER, "reliability_summary.json")) as f:
      return json.load(f)
  except Exception:
    return None


def _pooled_metrics(output_dir):
  try:
    import csv

    from zairachem.base.vars import PERFORMANCE_TABLE_FILENAME, RESULTS_SUBFOLDER

    with open(os.path.join(output_dir, RESULTS_SUBFOLDER, PERFORMANCE_TABLE_FILENAME)) as f:
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


def summarize_projections(output_dir=None):
  d = _resolve_output_dir(output_dir)
  if not d:
    return ""
  params = _load_params(d)
  extra = params.get("projection_ids") or []
  return "MW/LogP" + (f" + {_plurals(len(extra), 'model')}" if extra else " (built-in)")


def summarize_treat(output_dir=None):
  d = _resolve_output_dir(output_dir)
  if not d:
    return ""
  params = _load_params(d)
  n = _n_compounds(d)
  width = _descriptor_feature_width(d, params.get("featurizer_ids", []) or [])
  return f"{n:,} × {width:,} matrix" if (n is not None and width is not None) else "matrix imputed"


def _cv_stats(output_dir):
  """Per-descriptor lazy-qsar CV reports (estimators/*/*/cv_report.json)."""
  import glob

  from zairachem.base.vars import ESTIMATORS_SUBFOLDER

  stats = []
  for f in glob.glob(os.path.join(output_dir, ESTIMATORS_SUBFOLDER, "*", "*", "cv_report.json")):
    try:
      with open(f) as fh:
        r = json.load(fh)
      r["descriptor"] = os.path.basename(os.path.dirname(f))
      stats.append(r)
    except Exception:
      continue
  return stats


def _cv_headline(cv):
  """(mean_oof, best_descriptor, best_oof) or None from a list of cv reports."""
  aucs = [s["oof_auc"] for s in cv if s.get("oof_auc") is not None]
  if not aucs:
    return None
  best = max(cv, key=lambda s: s.get("oof_auc") if s.get("oof_auc") is not None else -1)
  return sum(aucs) / len(aucs), best.get("descriptor"), best.get("oof_auc")


def summarize_estimate(output_dir=None):
  d = _resolve_output_dir(output_dir)
  if not d:
    return ""
  if _is_predict():
    # Predict applies the trained models — there are no freshly trained estimators or CV stats here.
    n = len(_load_params(d).get("featurizer_ids", []) or [])
    return _plurals(n, "descriptor") + " scored" if n else "models applied"
  algos = _estimator_algorithms(d)
  base = _plurals(len(algos), "algorithm") + " trained" if algos else "estimators trained"
  head = _cv_headline(_cv_stats(d))
  if head:
    mean, best, best_auc = head
    base += f" · CV AUROC {mean:.2f} (best {best} {best_auc:.2f})"
  return base


def summarize_pool(output_dir=None):
  d = _resolve_output_dir(output_dir)
  if not d:
    return ""
  rel = _reliability_summary(d)
  if rel:
    parts = [_plurals(rel.get("n_descriptors", 0), "descriptor")]
    ad = rel.get("applicability")
    if ad and ad.get("n_out_of_domain"):
      parts.append(f"{ad['n_out_of_domain']:,} out-of-domain")
    return " · ".join(parts)
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
  "projections": summarize_projections,
  "treat": summarize_treat,
  "estimate": summarize_estimate,
  "pool": summarize_pool,
  "report": summarize_report,
  "finish": summarize_finish,
}


# --- Per-step detail content (rich, always-on), printed borderless under the banner -------------


def _bar(frac, width=12):
  """A calm mini progress bar (filled █ in the default fg vs dim ─) for a 0..1 fraction."""
  try:
    frac = max(0.0, min(1.0, float(frac)))
  except (TypeError, ValueError):
    frac = 0.0
  filled = round(frac * width)
  return f"{'█' * filled}[dim]{'─' * (width - filled)}[/]"


def _active_inactive(output_dir):
  """(n_active, n_inactive) from data.csv's binary column, or None."""
  from zairachem.base.vars import DATA_FILENAME, DATA_SUBFOLDER

  try:
    import pandas as pd

    df = pd.read_csv(os.path.join(output_dir, DATA_SUBFOLDER, DATA_FILENAME))
    if "bin" in df.columns:
      n_act = int(df["bin"].sum())
      return n_act, len(df) - n_act
  except Exception:
    pass
  return None


def _provenance(output_dir):
  """The run's data-provenance dict (per-model n_total/n_from_project/n_computed), or {}."""
  try:
    from zairachem.base.utils.isaura_report import _load_provenance

    return _load_provenance(output_dir)
  except Exception:
    return {}


def _model_width(output_dir, eos):
  """Descriptor column count for a single featurizer model, or None."""
  from zairachem.base.vars import DESCRIPTORS_SUBFOLDER

  try:
    import h5py

    base = os.path.join(output_dir, DESCRIPTORS_SUBFOLDER, eos)
    h5 = os.path.join(base, "raw.h5")
    if not os.path.exists(h5):
      chunk = os.path.join(base, "raw_chunks", "chunk_0000.h5")
      h5 = chunk if os.path.exists(chunk) else None
    if not h5:
      return None
    with h5py.File(h5, "r") as f:
      if "Features" in f:
        return int(f["Features"].shape[0])
      if "Values" in f:
        return int(f["Values"].shape[1])
  except Exception:
    pass
  return None


def _treat_widths(output_dir, featurizer_ids):
  """(columns_in, columns_out) summed across featurizers — raw H5 width vs treated info width.

  Returns None if nothing readable. Derived entirely from artifacts (no extra persistence).
  """
  from zairachem.base.vars import DESCRIPTORS_SUBFOLDER, TREATED_DESC_FILENAME

  ci = co = 0
  ok = False
  for eos in featurizer_ids:
    rin = _model_width(output_dir, eos)
    info = os.path.join(
      output_dir, DESCRIPTORS_SUBFOLDER, eos, TREATED_DESC_FILENAME.replace(".h5", ".json")
    )
    try:
      with open(info) as f:
        cout = int(json.load(f).get("features"))
    except Exception:
      cout = None
    if rin is not None and cout is not None:
      ci += rin
      co += cout
      ok = True
  return (ci, co) if ok else None


def _plots_by_category(output_dir):
  """Group report PNGs into coarse categories by filename keyword. Returns {category: count}."""
  from zairachem.base.vars import REPORT_SUBFOLDER

  buckets = {
    "roc": "ROC",
    "auroc": "ROC",
    "calib": "calibration",
    "dist": "distributions",
    "violin": "distributions",
    "strip": "distributions",
    "proj": "projections",
    "cv": "cross-validation",
    "confusion": "confusion",
    "r2": "regression",
    "obs": "regression",
    "hist": "histograms",
  }
  out = {}
  try:
    p = os.path.join(output_dir, REPORT_SUBFOLDER, "png")
    for fn in os.listdir(p):
      if not fn.endswith(".png"):
        continue
      low = fn.lower()
      cat = next((label for kw, label in buckets.items() if kw in low), "other")
      out[cat] = out.get(cat, 0) + 1
  except Exception:
    return {}
  return out


def _dir_size_mb(output_dir):
  total = 0
  try:
    for root, _dirs, files in os.walk(output_dir):
      for f in files:
        try:
          total += os.path.getsize(os.path.join(root, f))
        except OSError:
          continue
    return total / (1024 * 1024)
  except Exception:
    return None


def _fmt_timing(timing, top=4):
  """Compact 'phase Xs · phase Ys' string from a cv_report timing dict (largest phases first)."""
  if not isinstance(timing, dict):
    return ""
  scalars = [(k, v) for k, v in timing.items() if isinstance(v, (int, float))]
  if not scalars:
    return ""
  items = sorted(scalars, key=lambda kv: -kv[1])[:top]
  return " · ".join(f"{k} {v:.1f}s" for k, v in items)


def _detail_rows(key, output_dir=None):
  """Rich (label, value) detail lines per step, read from the run's artifacts; [] if unavailable."""
  d = _resolve_output_dir(output_dir)
  if not d:
    return []
  params = _load_params(d)
  if key in _LIVE_TABLE_STEPS:
    # These steps keep their live table on screen as the record — no separate detail block.
    return []
  if key == "setup":
    rows = []
    n = _n_compounds(d)
    if params.get("task") == "classification" and _active_inactive(d):
      a, i = _active_inactive(d)
      rows.append(("compounds", f"{n:,} [dim]·[/] {a:,} active [dim]·[/] {i:,} inactive"))
    elif n is not None:
      rows.append(("compounds", f"{n:,}"))
    feats = params.get("featurizer_ids", []) or []
    if feats:
      rows.append(("featurizers", "  ".join(feats)))
    projs = params.get("projection_ids", []) or []
    rows.append((
      "projection",
      "MW/LogP" + (f"  +  {'  '.join(projs)}" if projs else " [dim](built-in)[/]"),
    ))
    store = params.get("contribute_store")
    rows.append(("store", f"on [dim]· project {store}[/]" if store else "[dim]off[/]"))
    return rows
  if key == "pool":
    rows = []
    rel = _reliability_summary(d)
    if rel:
      # Per-sample reliability pooler: show how descriptors were combined. At predict time (no
      # labels → no pooled metrics) these rows are the user's window into what the step did.
      rows.append(("method", "reliability [dim]· per-sample weighted (logit space)[/]"))
      rows.append((
        "weighting",
        f"{rel.get('tier', '?')} [dim]· {_plurals(rel.get('n_descriptors', 0), 'descriptor')}[/]",
      ))
      ad = rel.get("applicability")
      if ad:
        n_ood = ad.get("n_out_of_domain", 0)
        n_tot = rel.get("n_compounds", 0) or 0
        pct = 100.0 * ad.get("frac_out_of_domain", 0.0)
        col = "yellow" if n_ood else "dim"
        rows.append((
          "applicability",
          f"[{col}]{n_ood:,}[/]/{n_tot:,} compounds out-of-domain [dim]({pct:.0f}%)[/]",
        ))
        flagged = [
          f"{name} {cnt:,}"
          for name, cnt in sorted(
            ad.get("per_descriptor_out_of_domain", {}).items(), key=lambda kv: -kv[1]
          )
          if cnt
        ][:4]
        if flagged:
          rows.append(("ood by descriptor", "[dim]" + "  ".join(flagged) + "[/]"))
      mw = rel.get("mean_weights", {})
      if mw:
        top = sorted(mw.items(), key=lambda kv: -kv[1])[:4]
        rows.append((
          "mean weights",
          "  ".join(f"{name} [bold]{w:.2f}[/]" for name, w in top),
        ))
      m = _pooled_metrics(d)
      if m:

        def fmtr(k):
          try:
            return f"{float(m[k]):.3f}"
          except Exception:
            return "—"

        if params.get("task") == "classification":
          rows.append((
            "pooled",
            f"AUROC {fmtr('auroc')} [dim]·[/] acc {fmtr('accuracy')} [dim]·[/] MCC {fmtr('mcc')}",
          ))
        else:
          rows.append(("pooled", f"R² {fmtr('r2')}"))
      return rows
    algos = _estimator_algorithms(d)
    rows.append(("consensus", f"{_plurals(len(algos), 'descriptor model')}" if algos else "pooled"))
    m = _pooled_metrics(d)
    if m:

      def fmt(k):
        try:
          return f"{float(m[k]):.3f}"
        except Exception:
          return "—"

      if params.get("task") == "classification":
        rows.append((
          "pooled",
          f"AUROC {fmt('auroc')} [dim]·[/] acc {fmt('accuracy')} [dim]·[/] MCC {fmt('mcc')}",
        ))
      else:
        rows.append(("pooled", f"R² {fmt('r2')}"))
    return rows
  if key == "report":
    from zairachem.base.vars import REPORT_SUBFOLDER, RESULTS_SUBFOLDER

    rows = []
    cats = _plots_by_category(d)
    if cats:
      total = sum(cats.values())
      breakdown = "  ".join(f"{v} {k}" for k, v in sorted(cats.items(), key=lambda kv: -kv[1]))
      rows.append(("plots", f"{total} [dim]· {breakdown}[/]"))
    outs = [
      f
      for f in ("output.csv",)
      if os.path.exists(os.path.join(d, RESULTS_SUBFOLDER, f))
    ]
    if outs:
      rows.append(("outputs", "  ".join(outs)))
    html_path = os.path.join(d, REPORT_SUBFOLDER, "report.html")
    if os.path.exists(html_path):
      rows.append(("html", _collapse(html_path)))
    return rows
  if key == "finish":
    from zairachem.base.vars import RESULTS_SUBFOLDER

    rows = [("model", _collapse(d))]
    size = _dir_size_mb(d)
    if size is not None:
      rows.append(("size", f"{size:,.1f} MB"))
    present = [
      f
      for f in ("output.csv",)
      if os.path.exists(os.path.join(d, RESULTS_SUBFOLDER, f))
    ]
    from zairachem.base.vars import REPORT_SUBFOLDER

    if os.path.exists(os.path.join(d, REPORT_SUBFOLDER, "report.html")):
      present.append("report.html")
    if present:
      rows.append(("artifacts", "  ".join(present)))
    return rows
  return []


def final_summary_panel(output_dir=None):
  """Render the closing run-summary panel: headline metrics + output location."""
  # The live checklist is transient; stop it first so the panel lands cleanly below the records.
  tracker.stop()
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
