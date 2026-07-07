import os
import pandas as pd

import stylia

from zairachem.base import ZairaBase
from zairachem.base.vars import DATA_FILENAME, DATA_SUBFOLDER, REPORT_SUBFOLDER

# stylia 1.0.1 dropped the TWO_COLUMNS_WIDTH constant; keep the original figure proportions.
TWO_COLUMNS_WIDTH = 7.09

# In stylia 1.0.1 create_figure's width/height are scale factors, not inches, so the old
# inch-based figsizes must be rescaled. This factor maps the individual figure to ~1870 px,
# matching the previous (0.0.2) report output; other figsizes scale proportionally.
FIGSIZE_SCALE = 0.43 / (TWO_COLUMNS_WIDTH / 2)

# Raster DPI for the report PNGs. stylia.save_figure hardcodes dpi=600 (print quality), but these PNGs
# are only shown in the HTML grid at small sizes, so 600 dpi produced ~5x larger files for no on-screen
# benefit (a ~12 MB → ~2.4 MB report footprint drop). 200 dpi is retina-crisp at the display size; the
# accompanying per-figure vector PDF remains the publication-quality download.
REPORT_DPI = 200

# Reference grid for figure footprints. Every figure declares an (rows, cols) footprint in 3 cm cells.
# Two distinct quantities (kept separate on purpose):
#   * CELLS_PER_WIDTH — how many 3 cm cells span stylia's "print" full width (≈ 18 cm ÷ 3 cm = 6).
#     This is the SIZING divisor: a footprint maps to width=cols/CELLS_PER_WIDTH,
#     height=rows/CELLS_PER_WIDTH (cells square; both fractions of stylia's full WIDTH).
#   * GRID_COLS × GRID_ROWS — the composite/display reference grid (landscape 10 columns × 6 rows =
#     30 cm wide × 18 cm tall) shown in the report HTML's "About the figures" section and used for the
#     per-card footprint badges. The composite canvas is wider than one print figure on purpose.
CELLS_PER_WIDTH = 6
GRID_COLS = 10
GRID_ROWS = 6
CELL_CM = 3.0

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

  def has_clf_data(self):
    return any("bin" in c and "_skip" not in c and "_aux" not in c for c in self._columns())

  def has_reg_data(self):
    return any("val" in c and "_skip" not in c and "_aux" not in c for c in self._columns())


class BaseTable(BaseResults):
  def __init__(self, path):
    BaseResults.__init__(self, path=path)


class BasePlot(BaseResults):
  def __init__(self, ax, path, cells=None, figsize=None):
    BaseResults.__init__(self, path=path)
    # Footprint on the reference grid as (rows, cols) of 3 cm cells — the source of truth for size.
    self.cells = cells or (2, 2)
    if ax is None:
      if figsize is not None:
        # Legacy inch-like sizing (kept as a fallback); prefer ``cells``.
        _, ax = stylia.create_figure(
          1, 1, width=figsize[0] * FIGSIZE_SCALE, height=figsize[1] * FIGSIZE_SCALE
        )
      else:
        rows, cols = self.cells
        _, ax = stylia.create_figure(
          1, 1, width=cols / CELLS_PER_WIDTH, height=rows / CELLS_PER_WIDTH
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

    # Both a raster PNG (shown in the report + a download link) and a vector PDF (a second download
    # option), written to report/png/ and report/pdf/. The HTML references both directly.
    report_dir = os.path.join(self.path, REPORT_SUBFOLDER)
    png_dir = os.path.join(report_dir, "png")
    pdf_dir = os.path.join(report_dir, "pdf")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)
    # Match stylia.save_figure's layout (tight_layout + bbox_inches="tight") but write the PNG at the
    # report DPI instead of stylia's hardcoded 600 — far faster to encode and smaller on disk, with no
    # visible difference at the HTML display size. The PDF stays vector (publication-quality download).
    plt.tight_layout()
    plt.savefig(
      os.path.join(png_dir, self.name + ".png"),
      dpi=REPORT_DPI,
      transparent=False,
      bbox_inches="tight",
    )
    plt.savefig(os.path.join(pdf_dir, self.name + ".pdf"), transparent=False, bbox_inches="tight")
    plt.close()

  def load(self):
    pass
