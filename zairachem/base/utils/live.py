"""Live, in-place progress widgets for the ZairaChem CLI.

Self-contained presentation primitives — the per-item live table (:class:`LiveTableMonitor`), the
single-line :class:`LiveProgressBar` / :class:`SetupProgress`, the braille spinner, the host CPU/RAM
caption and the mini :func:`_bar`. They render to the shared console and depend only on it (no
pipeline/summary logic), so the orchestration layer (:mod:`pipeline_tracker`) and the report/setup
steps can build on them without import cycles.

Re-exported from :mod:`zairachem.base.utils.progress` for backwards compatibility.
"""

import contextlib
import threading
import time

from rich.table import Table
from rich.text import Text

from zairachem.base.utils.console import active_color, console, echo


def _bar(frac, width=12):
  """A calm mini progress bar (filled █ in the default fg vs dim ─) for a 0..1 fraction."""
  try:
    frac = max(0.0, min(1.0, float(frac)))
  except (TypeError, ValueError):
    frac = 0.0
  filled = round(frac * width)
  return f"{'█' * filled}[dim]{'─' * (width - filled)}[/]"


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

  Renders a discrete :class:`LiveProgressBar` with ``show_bar=False`` — just a dim percentage + the
  operation name shown dim as the activity (no bar glyph, no bold, no count) — instead of the old
  multi-column *pulsing* bar, which both blinked and clashed with the rest of the UI. The final
  percentage persists as the sub-step's record (like the other steps), rather than vanishing. The
  optional ``description`` on :meth:`update` is appended as the dim sub-phase (e.g. the merge phase).
  """

  def __init__(self, color=None, width=24):
    self.color = color or active_color()
    self.width = width
    self._bar = None
    self._cm = None

  def __enter__(self):
    return self

  def add_task(self, description, total=0):
    # show_bar=False: the Setup sub-steps (validate/standardize SMILES, consistency checks, merge)
    # show just a dim percentage + the operation name — no bar glyph, per request.
    self._bar = LiveProgressBar(
      description,
      total=total,
      color=self.color,
      width=self.width,
      persist=True,
      discrete=True,
      show_bar=False,
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

    The table renders in a single **non-transient** Live region: the finished state is drawn as the
    last frame and left on screen as the step's permanent record. We deliberately do NOT use a
    transient region plus a separate ``console.print(self)`` — those are two independent render
    surfaces, and on fast steps the transient clear could miscount lines and leave a duplicated
    title/header behind before the reprint. One surface, one copy. The table height is kept constant
    across refreshes (no caption; fixed column widths in :meth:`__rich__`) so in-place redraws stay
    clean. Only one ``Live`` region may be active at a time — don't nest.
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
        self, console=console, transient=False, refresh_per_second=8, auto_refresh=True
      ) as live:
        self._live = live
        yield self
        # Draw the finished state as the last frame while Live is still active; with transient=False
        # that frame stays on screen as the step's record — no separate reprint (which would double
        # the title/header on fast steps).
        live.update(self, refresh=True)
    finally:
      self._live = None
      self._on_exit()
