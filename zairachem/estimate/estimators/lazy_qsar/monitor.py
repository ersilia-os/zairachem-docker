"""Live, digested progress for the per-descriptor lazy-qsar training (Estimate step).

Each descriptor trains one ``LazyClassifier.fit()``, which is otherwise a black box on screen. This
module renders a live, in-place table (themed, no emojis) showing every descriptor go
``queued → training: <inner step> → done`` with its algorithms, out-of-fold AUROC and wall-clock,
and a live "current inner step" captured from lazy-qsar's own loguru messages (``Fitting head
2/2: rf``, ``Batch 1/3``, ``calibrate: …``) — without ever surfacing lazy-qsar's raw tables.

The generic live-table machinery (transient Live, fixed-height rows, off-TTY fallback, start/finish/
substep state) lives in :class:`zairachem.base.utils.progress.LiveTableMonitor`; this subclass adds
the estimate-specific columns and the loguru-sink substep capture. The live view is transient: it
animates while training and clears at the end, leaving the durable per-descriptor summary (rendered
by ``base/utils/progress._detail_rows``) as the permanent record.
"""

import contextlib
import re

from zairachem.base.utils.progress import LiveTableMonitor

_RE_HEAD = re.compile(r"[Ff]itting head (\d+)/(\d+):\s*(\S+)")
_RE_BATCH = re.compile(r"[Bb]atch (\d+)/(\d+)")
_RE_WINNER = re.compile(r"[Pp]ortfolio winner:\s*(\S+)")


def _substep_from_message(msg):
  """Digest a lazy-qsar loguru message into a short status, or None if not informative."""
  m = _RE_HEAD.search(msg)
  if m:
    return f"{m.group(3)} (head {m.group(1)}/{m.group(2)})"
  m = _RE_BATCH.search(msg)
  if m:
    return f"batch {m.group(1)}/{m.group(2)}"
  m = _RE_WINNER.search(msg)
  if m:
    return f"selected {m.group(1)}"
  if "calibrat" in msg.lower():
    return "calibrating"
  return None


class TrainingMonitor(LiveTableMonitor):
  """The Estimate step's per-descriptor training table.

  Columns: Descriptor | Status (queued → training:<substep> → done/skipped) | Algorithms |
  OOF AUC | Time. The live substep is captured from lazy-qsar's loguru messages via a temporary
  sink installed for the lifetime of :meth:`live`.

  Parameters
  ----------
  model_ids : list of str
      Descriptors to be trained, in order.
  color : str
      Accent colour (the estimate step's colour).
  result_reader : callable, optional
      ``model_id -> (oof_auc, portfolio)`` consulted on completion to fill the row from the
      descriptor's persisted ``cv_report.json``. Best-effort; may return ``(None, [])``.
  """

  item_label = "Descriptor"
  title = "Training models per descriptor"
  running_verb = "training"
  show_activity = True  # stream lazy-qsar's raw loguru line in a trailing dim column
  # Width of all non-activity columns so Activity gets exactly the leftover terminal width (no column
  # shrink). Fit columns: Descriptor 10 + Status 28 + Algorithms 24 + OOF 9 + Signals 9 + Time 8 = 88
  # content, plus 6 inter-column gaps × 4 padding = 24 → 112. (Predict drops Algorithms+OOF: see below.)
  reserved_width = 112

  def __init__(self, model_ids, color="yellow", result_reader=None, predict=False):
    super().__init__(model_ids, color=color, result_reader=result_reader)
    self._sink_id = None
    # At predict the trained models are only *applied* — there is no training, no OOF AUROC and no
    # algorithm portfolio to report, so relabel and drop those (fit-only) columns.
    self.predict = predict
    if predict:
      self.title = "Applying models per descriptor"
      self.running_verb = "applying"
      # Predict drops Algorithms + OOF: Descriptor 10 + Status 28 + Signals 9 + Time 8 = 55 content,
      # plus 4 inter-column gaps × 4 = 16 → 71. More width is therefore free for Activity.
      self.reserved_width = 71

  # --- estimate-specific columns ---

  def _columns(self, table):
    if not self.predict:
      table.add_column("Algorithms", width=24, no_wrap=True, overflow="ellipsis")
      table.add_column("OOF AUC", justify="right", width=9, no_wrap=True)
    # Pooler reliability signals each descriptor produces (applicability domain, rank→error
    # reliability curve) — lit when present, dim when absent.
    table.add_column("Signals", width=9, no_wrap=True)
    table.add_column("Time", justify="right", width=8, no_wrap=True)

  def _signals_cell(self, s):
    def tok(label, on):
      return f"[{self.color}]{label}[/]" if on else f"[dim]{label}[/]"

    return f"{tok('AD', s['extra'].get('ad'))} {tok('rank', s['extra'].get('rank'))}"

  def _row_cells(self, item_id, s):
    cells = []
    if not self.predict:
      portfolio = s["extra"].get("portfolio") or []
      oof = s["extra"].get("oof")
      cells.append("  ".join(portfolio) if portfolio else "[dim]—[/]")
      cells.append(f"[bold]{oof:.3f}[/]" if isinstance(oof, (int, float)) else "[dim]—[/]")
    cells.append(self._signals_cell(s))
    cells.append(self._fmt_time(s))
    return cells

  def _store_result(self, s, result):
    """Merge the ``result_reader`` dict ({oof, portfolio, ad, rank}) into the row's extra.

    Tolerates the legacy ``(oof_auc, portfolio)`` tuple form for backward compatibility.
    """
    if isinstance(result, dict):
      s["extra"].update(result)
      s["extra"]["portfolio"] = list(result.get("portfolio") or [])
    else:
      oof, portfolio = result
      s["extra"]["oof"] = oof
      s["extra"]["portfolio"] = list(portfolio or [])

  def _plain_finish(self, item_id, s):
    from zairachem.base.utils.console import echo

    el = f"{s['elapsed']:.1f}s" if s and s["elapsed"] is not None else "—"
    sig = []
    if s and s["extra"].get("ad"):
      sig.append("AD")
    if s and s["extra"].get("rank"):
      sig.append("rank")
    sig_txt = ("+" + ",".join(sig)) if sig else ""
    status = s["status"] if s else "done"
    if self.predict:
      echo(f"{status} {item_id}  {sig_txt}  [dim]{el}[/]", kind="info")
      return
    oof = s["extra"].get("oof") if s else None
    oof_txt = f"OOF {oof:.3f}" if isinstance(oof, (int, float)) else ""
    algos = " ".join(s["extra"].get("portfolio") or []) if s else ""
    echo(f"{status} {item_id}  {algos}  {oof_txt}  {sig_txt}  [dim]{el}[/]", kind="info")

  # --- loguru capture (installed for the lifetime of live()) ---

  def _on_log(self, message):
    try:
      raw = message.record["message"]
    except Exception:
      return
    if not self._active:
      return
    # Two views of the same line: a digested substep in the Status column (when the line matches a
    # known pattern) and the raw line, verbatim, in the trailing dim Activity column.
    self.set_activity(self._active, raw)
    try:
      sub = _substep_from_message(raw)
    except Exception:
      sub = None
    if sub:
      self.set_substep(self._active, sub)

  def _on_enter(self):
    from loguru import logger as _loguru

    try:
      self._sink_id = _loguru.add(
        self._on_log,
        level="INFO",
        filter=lambda r: str(r["name"]).startswith("lazyqsar"),
      )
    except Exception:
      self._sink_id = None

  def _on_exit(self):
    from loguru import logger as _loguru

    if self._sink_id is not None:
      with contextlib.suppress(Exception):
        _loguru.remove(self._sink_id)
      self._sink_id = None
