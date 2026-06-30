import os
import pandas as pd

import stylia

from zairachem.base import ZairaBase
from zairachem.base.vars import DATA_FILENAME, DATA_SUBFOLDER, REPORT_SUBFOLDER

# stylia 1.0.1 dropped the TWO_COLUMNS_WIDTH constant; keep the original figure proportions.
TWO_COLUMNS_WIDTH = 7.09
INDIVIDUAL_FIGSIZE = (TWO_COLUMNS_WIDTH / 2, TWO_COLUMNS_WIDTH / 2)

# In stylia 1.0.1 create_figure's width/height are scale factors, not inches, so the old
# inch-based figsizes must be rescaled. This factor maps the individual figure to ~1870 px,
# matching the previous (0.0.2) report output; other figsizes scale proportionally.
FIGSIZE_SCALE = 0.43 / (TWO_COLUMNS_WIDTH / 2)

# Publication-ready figures: the non-branded "article" style (NPG / Nature Publishing Group palette)
# and the "print" format — so plots can be dropped straight into papers. Set once at import.
stylia.set_style("article")
stylia.set_format("print")


class BaseResults(ZairaBase):
  def __init__(self, path):
    ZairaBase.__init__(self)
    if path is None:
      self.path = self.get_output_dir()
    else:
      self.path = path
    self._data_columns = None  # cached column names of data.csv (read once; see _columns())

  def _columns(self):
    """Column names of the run's ``data.csv``, read once and cached. Every plot's availability guard
    (``has_clf_data`` etc.) consults these, so re-reading the CSV per call was pure waste."""
    if self._data_columns is None:
      # Only the header is needed (these are column-name checks), so read zero rows.
      df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME), nrows=0)
      self._data_columns = list(df.columns)
    return self._data_columns

  def has_outcome_data(self):
    return any("clf" in c or "reg" in c for c in self._columns())

  def has_clf_data(self):
    return any("bin" in c and "_skip" not in c and "_aux" not in c for c in self._columns())

  def has_reg_data(self):
    return any("val" in c and "_skip" not in c and "_aux" not in c for c in self._columns())


class BaseTable(BaseResults):
  def __init__(self, path):
    BaseResults.__init__(self, path=path)


class BasePlot(BaseResults):
  def __init__(self, ax, path, figsize=None):
    BaseResults.__init__(self, path=path)
    if ax is None:
      if figsize is None:
        figsize = INDIVIDUAL_FIGSIZE
      _, ax = stylia.create_figure(
        1, 1, width=figsize[0] * FIGSIZE_SCALE, height=figsize[1] * FIGSIZE_SCALE
      )
    self.name = "base"
    self.ax = ax[0]
    # stylia 1.0.1's AxisManager re-applies placeholder axis titles ("X-axis / Units" /
    # "Y-axis / Units") on every ``ax[0]`` access, so clear them only after binding ``self.ax``.
    # Plots that don't set their own labels (categorical heatmaps / horizontal bars) then render
    # clean; plots that call set_xlabel/set_ylabel override these blanks afterwards.
    self.ax.set_xlabel("")
    self.ax.set_ylabel("")

  def save(self):
    if not self.is_available:
      return
    import matplotlib.pyplot as plt

    # PNG only (report/png/). The HTML references these files directly, so a per-figure PDF copy would
    # just double the figure footprint for a download link few use — dropped to keep the report lean.
    png_dir = os.path.join(self.path, REPORT_SUBFOLDER, "png")
    os.makedirs(png_dir, exist_ok=True)
    stylia.save_figure(os.path.join(png_dir, self.name + ".png"))
    plt.close()

  def load(self):
    pass
