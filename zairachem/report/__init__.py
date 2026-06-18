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

  def has_outcome_data(self):
    df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
    for c in list(df.columns):
      if "clf" in c:
        return True
      if "reg" in c:
        return True
    return False

  def has_clf_data(self):
    df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
    for c in list(df.columns):
      if "bin" in c and "_skip" not in c and "_aux" not in c:
        return True
    return False

  def has_reg_data(self):
    df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
    for c in list(df.columns):
      if "val" in c and "_skip" not in c and "_aux" not in c:
        return True
    return False


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

  def save(self):
    if not self.is_available:
      return
    import matplotlib.pyplot as plt

    # Keep the report folder tidy: PNGs in report/png/, PDFs in report/pdf/. Always save both
    # (PNG raster at 600 dpi via stylia, PDF vector) of the same figure.
    report = os.path.join(self.path, REPORT_SUBFOLDER)
    png_dir = os.path.join(report, "png")
    pdf_dir = os.path.join(report, "pdf")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)
    stylia.save_figure(os.path.join(png_dir, self.name + ".png"))
    plt.savefig(os.path.join(pdf_dir, self.name + ".pdf"), bbox_inches="tight")
    plt.close()

  def load(self):
    pass
