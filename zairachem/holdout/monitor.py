"""Live per-fold table for the held-out validation (Evaluate) step.

One row per fold, showing the split type, train/test molecule counts, the current sub-phase (which
descriptor is fitting, then pooling/scoring), and the held-out AUROC/AUPR once scored. The table title
carries a self-correcting ETA — kept in the title rather than a caption because a toggling caption makes
the transient Live miscount its height (see :meth:`LiveTableMonitor.__rich__`).
"""

from zairachem.base.utils.live import LiveTableMonitor


def _fmt_metric(v):
  try:
    return f"{float(v):.3f}"
  except (TypeError, ValueError):
    return "[dim]—[/]"


def _fmt_duration(seconds):
  """Compact human duration: ``45s`` / ``3m20s`` / ``12m``."""
  if seconds is None:
    return "—"
  seconds = max(0, int(round(seconds)))
  if seconds < 90:
    return f"{seconds}s"
  m, s = divmod(seconds, 60)
  return f"{m}m{s:02d}s" if m < 10 else f"{m}m"


class EvaluateMonitor(LiveTableMonitor):
  """Per-fold live validation table. Columns: Fold | Status | Split | Train/Test | AUROC | AUPR | Time."""

  item_label = "Fold"
  running_verb = "evaluating"
  # Wider than the default so substeps like "model 3/6: eos4u6p" fit without ellipsis.
  status_width = 34

  def __init__(self, item_ids, color="bright_blue", est_seconds=None):
    super().__init__(item_ids, color=color)
    # Per-fold time predictor from the training step (a fold refits the same descriptor stack). Only
    # trusted if meaningfully large — a resumed run whose estimate step was skipped reports ~0s.
    self._est_seconds = est_seconds if (est_seconds and est_seconds > 5) else None

  def _columns(self, table):
    table.add_column("Split", width=13, no_wrap=True)
    table.add_column("Train/Test", justify="right", width=11, no_wrap=True)
    table.add_column("AUROC", justify="right", width=7, no_wrap=True)
    table.add_column("AUPR", justify="right", width=7, no_wrap=True)
    table.add_column("Time", justify="right", width=8, no_wrap=True)

  def _row_cells(self, item_id, s):
    e = s["extra"]
    return [
      e.get("split", "[dim]—[/]"),
      e.get("counts", "[dim]—[/]"),
      _fmt_metric(e.get("auroc")),
      _fmt_metric(e.get("aupr")),
      self._fmt_time(s),
    ]

  @property
  def title(self):
    """Dynamic title with a self-correcting ETA (recomputed each render; always one line)."""
    statuses = [self.state[m] for m in self.order]
    n = len(self.order)
    done = [s for s in statuses if s["status"] in ("done", "skipped")]
    label = f"Held-out validation · {len(done)}/{n} folds"
    if len(done) >= n:
      return label + " · done"
    done_times = [s["elapsed"] for s in done if s["elapsed"] is not None]
    per_fold = (sum(done_times) / len(done_times)) if done_times else self._est_seconds
    if not per_fold:
      return label
    remaining = per_fold * sum(1 for s in statuses if s["status"] == "queued")
    running = next((s for s in statuses if s["status"] == "running"), None)
    if running is not None:
      remaining += max(0.0, per_fold - (self._elapsed(running) or 0.0))
    return f"{label} · ~{_fmt_duration(remaining)} left"
