import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

from zairachem.report import BasePlot
from zairachem.report.colors import rgb as _color
from zairachem.report.fetcher import ResultsFetcher


def _percentile_rank(p):
  """Empirical percentile rank in [0, 1] of each value (ties broken by first occurrence)."""
  p = np.asarray(p, dtype=float)
  order = np.argsort(p, kind="mergesort")
  ranks = np.empty(len(p), dtype=float)
  ranks[order] = np.arange(len(p))
  return ranks / max(1, len(p) - 1)


def _draw_score_violins(ax, values, y, ylabel, show_points=False, max_pts=1000):
  """Vertical violins of a per-sample score split by class (inactive vs active).

  Always overlays a white inner boxplot — IQR box + median line + whiskers to the 5th/95th
  percentiles (with small caps). When ``show_points`` is set, up to ``max_pts`` jittered points per
  class are drawn over a lightened violin. Returns False (unavailable) if neither class has finite
  values."""
  y = np.asarray(y, dtype=int)
  v = np.asarray(values, dtype=float)
  finite = np.isfinite(v)
  present = [
    (c, name, _color(key))
    for c, name, key in ((0, "Inactive", "inactive"), (1, "Active", "active"))
    if np.any((y == c) & finite)
  ]
  if not present:
    return False
  order = [name for _, name, _ in present]
  palette = [col for _, _, col in present]
  data = pd.DataFrame({
    "value": v[finite],
    "cls": np.where(y[finite] == 1, "Active", "Inactive"),
  })
  sns.violinplot(
    x="cls", y="value", data=data, ax=ax, order=order, palette=palette, cut=0, inner=None
  )
  if show_points:
    for coll in ax.collections:  # lighten the violin bodies so the points read on top
      coll.set_alpha(0.35)
    rng = np.random.default_rng(0)
    for i, (c, _, col) in enumerate(present):
      pts = v[(y == c) & finite]
      if len(pts) > max_pts:
        pts = rng.choice(pts, max_pts, replace=False)
      jitter = rng.uniform(-0.18, 0.18, size=len(pts))
      ax.scatter(
        i + jitter, pts, s=7, color=col, alpha=0.6, linewidths=0.3, edgecolors="white", zorder=5
      )
  # Inner white boxplot: whiskers (5th–95th pct, with caps) + IQR box + a median line.
  for i, (c, _, _) in enumerate(present):
    p5, q1, med, q3, p95 = np.percentile(v[(y == c) & finite], [5, 25, 50, 75, 95])
    w, cap = 0.11, 0.05
    ax.plot([i, i], [q3, p95], color="white", lw=1.2, zorder=10)
    ax.plot([i, i], [p5, q1], color="white", lw=1.2, zorder=10)
    ax.plot([i - cap, i + cap], [p95, p95], color="white", lw=1.2, zorder=10)
    ax.plot([i - cap, i + cap], [p5, p5], color="white", lw=1.2, zorder=10)
    ax.add_patch(
      Rectangle((i - w, q1), 2 * w, q3 - q1, fill=False, edgecolor="white", lw=1.4, zorder=11)
    )
    ax.plot([i - w, i + w], [med, med], color="white", lw=1.8, zorder=12)
  ax.set_xticks(range(len(present)))
  ax.set_xticklabels(order)
  ax.set_xlabel("")
  ax.set_ylabel(ylabel)
  return True


class _PooledOofScorePlot(BasePlot):
  """Base for a pooled out-of-fold score distribution (actives vs inactives), shown through one
  score-type lens. The pooled classifier score (``clf_truth_proba``) is, at fit time, the honest
  out-of-fold prediction (each descriptor contributes its OOF probability scattered to compound
  order); this base draws it as vertical class-split violins with an inner white boxplot, optionally
  with jittered points.

  Subclasses set ``stem``/``title``/``ylabel`` and implement ``transform(p, y)`` → per-sample values
  (same length as ``p``), or return ``None`` when the lens has no data yet (e.g. the raw score before
  the pipeline persists it). ``show_points`` toggles the jittered-points variant. Classification only.
  """

  stem = None
  title = ""
  ylabel = ""
  show_points = False

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(3, 2))
    self.name = self.stem
    self.is_available = False
    if not self.has_clf_data():
      return
    ts = self._truth_score(path)
    if ts is None:  # this lens' source score isn't available (e.g. raw before a re-fit persists it)
      return
    y, p = ts
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(y) == 0 or len(set(y.tolist())) < 2:
      return
    vals = self.transform(p, y)
    if vals is None:
      return
    if not _draw_score_violins(
      self.ax, np.asarray(vals, dtype=float), y, self.ylabel, show_points=self.show_points
    ):
      return
    self.ax.set_title(self.title)
    self.is_available = True

  def _truth_score(self, path):
    """``(y_true, score)`` the lens operates on, or ``None`` if unavailable. Default: the pooled
    calibrated OOF probability. Overridden by the raw lens to use the uncalibrated pooled score."""
    return ResultsFetcher(path=path).clf_truth_proba()

  def transform(self, p, y):
    raise NotImplementedError


class OofScoreProbaPlot(_PooledOofScorePlot):
  """Plot of the pooled OOF score as a probability, actives vs inactives."""

  stem = "oof-score-proba"
  title = "Out-of-fold pooled score · probability"
  ylabel = "Probability"

  def transform(self, p, y):
    """Return the pooled score clipped to [0, 1]."""
    return np.clip(p, 0.0, 1.0)


class OofScoreLogitPlot(_PooledOofScorePlot):
  """Plot of the pooled OOF score on the log-odds scale, actives vs inactives."""

  stem = "oof-score-logit"
  title = "Out-of-fold pooled score · log-odds"
  ylabel = "Log-odds (logit)"

  def transform(self, p, y):
    """Return the logit of the pooled score, clipped away from 0 and 1."""
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(pc / (1 - pc))


class OofScoreRankPlot(_PooledOofScorePlot):
  """Plot of the pooled OOF score as a within-run percentile rank, actives vs inactives."""

  stem = "oof-score-rank"
  title = "Out-of-fold pooled score · percentile rank"
  ylabel = "Percentile rank"

  def transform(self, p, y):
    """Return the empirical percentile rank in [0, 1] of each pooled score."""
    return _percentile_rank(p)


class OofScoreLiftPlot(_PooledOofScorePlot):
  """Plot of the pooled OOF score as lift over the active base rate, actives vs inactives."""

  stem = "oof-score-lift"
  title = "Out-of-fold pooled score · lift"
  ylabel = "Lift (× base rate)"

  def transform(self, p, y):
    """Return the pooled score divided by the fraction of actives, or None if there are none."""
    prior = float(np.mean(y))
    return p / prior if prior > 0 else None


class OofScoreRawPlot(_PooledOofScorePlot):
  """Uncalibrated raw-score lens: the pooled OOF score before per-head probability calibration
  (lazy-qsar ≥ 3.4.2 exposes ``oof_raw_``; estimate pools & scatters it to a ``clf_raw`` column).
  Unavailable (blank) for models fit before that, when no pooled raw OOF was persisted."""

  stem = "oof-score-raw"
  title = "Out-of-fold pooled score · raw (uncalibrated)"
  ylabel = "Raw score (uncalibrated)"

  def _truth_score(self, path):
    return ResultsFetcher(path=path).clf_truth_raw()  # None when clf_raw wasn't persisted

  def transform(self, p, y):
    """Return the uncalibrated pooled score clipped to [0, 1]."""
    return np.clip(p, 0.0, 1.0)


# Jittered-points variants (same transforms, points over a lightened violin).
class OofScoreProbaPointsPlot(OofScoreProbaPlot):
  """Jittered-points variant of :class:`OofScoreProbaPlot` (points over a lightened violin)."""

  stem = "oof-score-proba-pts"
  title = "Out-of-fold pooled score · probability (points)"
  show_points = True


class OofScoreLogitPointsPlot(OofScoreLogitPlot):
  """Jittered-points variant of :class:`OofScoreLogitPlot` (points over a lightened violin)."""

  stem = "oof-score-logit-pts"
  title = "Out-of-fold pooled score · log-odds (points)"
  show_points = True


class OofScoreRankPointsPlot(OofScoreRankPlot):
  """Jittered-points variant of :class:`OofScoreRankPlot` (points over a lightened violin)."""

  stem = "oof-score-rank-pts"
  title = "Out-of-fold pooled score · percentile rank (points)"
  show_points = True


class OofScoreLiftPointsPlot(OofScoreLiftPlot):
  """Jittered-points variant of :class:`OofScoreLiftPlot` (points over a lightened violin)."""

  stem = "oof-score-lift-pts"
  title = "Out-of-fold pooled score · lift (points)"
  show_points = True


class OofScoreRawPointsPlot(OofScoreRawPlot):
  """Jittered-points variant of :class:`OofScoreRawPlot` (points over a lightened violin)."""

  stem = "oof-score-raw-pts"
  title = "Out-of-fold pooled score · raw (points)"
  show_points = True
