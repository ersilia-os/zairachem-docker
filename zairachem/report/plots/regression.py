from sklearn.metrics import (
  mean_absolute_error,
  r2_score,
)

from zairachem.report import BasePlot
from zairachem.report.colors import named_colors
from zairachem.report.fetcher import ResultsFetcher


class IndividualEstimatorsR2Plot(BasePlot):
  """Per-descriptor R² against the transformed target (regression only)."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 4))
    if self.has_reg_data():
      self.name = "r2-individual"
      ax = self.ax
      self.fetcher = ResultsFetcher(path=path)
      tasks = self.fetcher.get_reg_tasks()
      task = tasks[0]
      yt = self.fetcher.get_transformed()
      df_ys = self.fetcher._read_individual_estimator_results(task)
      scores = []
      labels = []
      for yp in list(df_ys.columns):
        scores += [r2_score(yt, list(df_ys[yp]))]
        labels += [yp]
      x = list(range(len(labels)))
      y = scores
      ax.scatter(x, y, color=named_colors.red)
      ax.set_xticks(x)
      ax.set_xticklabels(labels, rotation=90)
      ax.set_ylabel("R2")
      self.is_available = True
    else:
      self.is_available = False


class RegressionPlotTransf(BasePlot):
  """Predicted vs. observed scatter on the transformed activity, with R²/MAE (regression only)."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    if self.has_reg_data():
      self.is_available = True
      self.name = "regression-trans"
      ax = self.ax
      yt = ResultsFetcher(path=path).get_transformed()
      yp = ResultsFetcher(path=path).get_pred_reg_trans()
      ax.scatter(yt, yp, color=named_colors.purple, s=15, alpha=0.7)
      ax.set_xlabel("Observed Activity (Transformed)")
      ax.set_ylabel("Predicted Activity (Transformed)")
      ax.set_title(
        "R2 = {0} | MAE = {1}".format(
          round(r2_score(yt, yp), 3), round(mean_absolute_error(yt, yp), 3)
        )
      )
    else:
      self.is_available = False


class RegressionPlotRaw(BasePlot):
  """Predicted vs. observed scatter on the raw activity, with R²/MAE (regression only)."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    if self.has_reg_data():
      self.is_available = True
      self.name = "regression-raw"
      ax = self.ax
      yt = ResultsFetcher(path=path).get_raw()
      yp = ResultsFetcher(path=path).get_pred_reg_raw()
      ax.scatter(yt, yp, color=named_colors.green, s=15, alpha=0.7)
      ax.set_xlabel("Observed Activity")
      ax.set_ylabel("Predicted Activity")
      ax.set_title(
        "R2 = {0} | MAE = {1}".format(
          round(r2_score(yt, yp), 3), round(mean_absolute_error(yt, yp), 3)
        )
      )
    else:
      self.is_available = False


class Transformation(BasePlot):
  """Scatter of the raw → transformed activity mapping applied before training (regression only)."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    if self.has_reg_data():
      self.is_available = True
      self.name = "transformation"
      ax = self.ax
      yt = ResultsFetcher(path=path).get_raw()
      ytrans = ResultsFetcher(path=path).get_transformed()
      ax.scatter(yt, ytrans, color=named_colors.green, s=15, alpha=0.7)
      ax.set_xlabel("Observed Activity (Raw)")
      ax.set_ylabel("Observed Activity (Transformed)")
      ax.set_title("Continuous data transformation")
    else:
      self.is_available = False
