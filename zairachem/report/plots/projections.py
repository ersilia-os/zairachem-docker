import numpy as np
import seaborn as sns

from zairachem.base.utils.logging import logger
from zairachem.report import BasePlot
from zairachem.report.colors import rgb as _color
from zairachem.report.fetcher import ResultsFetcher
from zairachem.report.plots._common import (
  _draw_training_projection_background,
  _subsample,
)


def _draw_class_density(ax, x, y, color):
  """One class's filled-KDE density surface + translucent points onto ``ax`` (no legend)."""
  x = np.asarray(x, dtype=float)
  y = np.asarray(y, dtype=float)
  if len(x) == 0:
    return
  idx, alpha = _subsample(len(x))
  xs, ys = x[idx], y[idx]
  try:
    if len(xs) >= 5:
      sns.kdeplot(
        x=xs, y=ys, ax=ax, fill=True, color=color, levels=8, thresh=0.05, alpha=0.35, bw_adjust=1.1
      )
  except (ValueError, np.linalg.LinAlgError) as e:
    logger.debug(f"[report] density surface skipped, points only: {e}")
  ax.scatter(xs, ys, color=color, s=8, alpha=alpha, edgecolors="none", zorder=5)


def _projection_xy_by_class(path, projection):
  """``(x, y, bt)`` for a projection keeping only finite-coordinate, labelled rows (bt in {0,1})."""
  bt = np.asarray(ResultsFetcher(path=path).get_actives_inactives(), dtype=float)
  xs = np.asarray(projection["xs"], dtype=float)
  ys = np.asarray(projection["ys"], dtype=float)
  keep = np.isfinite(xs) & np.isfinite(ys) & ~np.isnan(bt)
  return xs[keep], ys[keep], bt[keep].astype(int)


class ProjectionMergedPlot(BasePlot):
  """A 2-D projection with both classes overlaid: per-class density surface + points (no legend).

  ``projection`` is a dict ``{name, title, x_label, y_label, xs, ys}`` from
  ``ResultsFetcher.get_projections``. Coloured by true class (crimson actives, cobalt inactives).
  """

  def __init__(self, ax, path, projection):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.name = "projection-merged-" + projection["name"]
    self.is_available = False
    if not self.has_clf_data():
      return  # coloured by true class — needs ground truth (predict-no-truth uses ProjectionProbaPlot)
    x, y, bt = _projection_xy_by_class(path, projection)
    if len(bt) == 0:
      return
    ax = self.ax
    if self.is_predict():  # backdrop of the trained model's chemical space (applicability domain)
      _draw_training_projection_background(ax, path, projection["name"])
    _draw_class_density(ax, x[bt == 0], y[bt == 0], _color("inactive"))
    _draw_class_density(ax, x[bt == 1], y[bt == 1], _color("active"))
    ax.set_xlabel(projection["x_label"])
    ax.set_ylabel(projection["y_label"])
    ax.set_title(projection["title"])
    self.is_available = True


class ProjectionClassPlot(BasePlot):
  """A 2-D projection for a single class (active or inactive): density surface + points (no legend)."""

  def __init__(self, ax, path, projection, cls):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    noun = "active" if cls == 1 else "inactive"
    self.name = f"projection-{projection['name']}-{noun}"
    self.is_available = False
    if not self.has_clf_data():
      return  # coloured by true class — needs ground truth
    x, y, bt = _projection_xy_by_class(path, projection)
    mask = bt == cls
    if not mask.any():
      return
    ax = self.ax
    if self.is_predict():  # backdrop of the trained model's chemical space (applicability domain)
      _draw_training_projection_background(ax, path, projection["name"])
    _draw_class_density(ax, x[mask], y[mask], _color(noun))
    ax.set_xlabel(projection["x_label"])
    ax.set_ylabel(projection["y_label"])
    ax.set_title(f"{projection['title']} · {noun}s")
    self.is_available = True
