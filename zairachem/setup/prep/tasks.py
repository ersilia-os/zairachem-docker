import os

import numpy as np
import pandas as pd
import rich_click as click

from zairachem.base import ZairaBase
from zairachem.base.utils.logging import logger
from zairachem.base.vars import (
  METADATA_SUBFOLDER,
  MIN_CLASS,
  MIN_REGRESSION_ROWS,
  PARAMETERS_FILE,
  TASKS_FILENAME,
  VALUES_COLUMN,
  VALUES_FILENAME,
)

from .files import ParametersFile


class ExpectedTaskType(ZairaBase):
  """Lookup of the task type (classification or regression) from the trained ``parameters.json``."""

  def __init__(self, path):
    ZairaBase.__init__(self)
    self.path = path
    if self.is_predict():
      self.trained_path = self.get_trained_dir()
    else:
      self.trained_path = self.get_output_dir()

  def _get_params(self):
    params = ParametersFile(
      path=os.path.join(self.trained_path, METADATA_SUBFOLDER, PARAMETERS_FILE)
    )
    return params.load()

  def get(self):
    """Return the ``task`` recorded in the trained model's ``parameters.json``."""
    params = self._get_params()
    return params["task"]


class RegTasks(object):
  """Validation of a continuous target: the values reach lazy-qsar as they are given.

  lazy-qsar standardises the target internally and reports predictions back in the input's own
  units, so ZairaChem neither transforms nor clips it. A target it cannot fit (non-finite,
  constant, or too few compounds) is rejected here, before any descriptor is computed.
  """

  def __init__(self, data):
    self.values = np.array(data[VALUES_COLUMN], dtype=float)

  def validated(self):
    """The target values, or raise a ``click.ClickException`` saying what is wrong with them."""
    if not np.isfinite(self.values).all():
      n = int((~np.isfinite(self.values)).sum())
      raise click.ClickException(
        f"{n:,} of {len(self.values):,} activity values are empty or not a number. A regression "
        "target must be numeric for every compound."
      )
    if len(self.values) < MIN_REGRESSION_ROWS:
      raise click.ClickException(
        f"Only {len(self.values):,} compounds: a regression model needs at least "
        f"{MIN_REGRESSION_ROWS}."
      )
    if np.ptp(self.values) == 0:
      raise click.ClickException(
        f"Every activity value is {self.values[0]:g}: there is nothing to learn from a constant "
        "target."
      )
    return self.values


class ClfTasks(object):
  """Fit-time binary classification labels, validated for class balance."""

  def __init__(self, data):
    self.values = self.binarize(np.array(data[VALUES_COLUMN]))

  def _has_enough_min_class(self, bins):
    n1 = np.sum(bins)
    n0 = len(bins) - n1
    if n1 < MIN_CLASS or n0 < MIN_CLASS:
      return False
    return True

  def binarize(self, values):
    """Return 0/1 labels from values in {0, 0.5, 1} (0.5 maps to 1).

    Raises a ``click.ClickException`` (one line, exit 1) if any value is not binary or if either
    class has fewer than ``MIN_CLASS`` samples.
    """
    accepted = set([0, 0.5, 1])
    bin_values = []
    for v in values:
      if v not in accepted:
        raise click.ClickException("Data is not binary. Cannot do classification")
      if v > 0:
        bin_values += [1]
      else:
        bin_values += [0]
    if not self._has_enough_min_class(bin_values):
      n1 = int(np.sum(bin_values))
      n0 = len(bin_values) - n1
      raise click.ClickException(
        f"Not enough samples per class for classification: "
        f"{n1} positive / {n0} negative (minimum {MIN_CLASS} each required)."
      )
    return bin_values


class ClfTasksForPrediction(object):
  """Predict-time binary ground-truth labels, without the class-balance requirement."""

  def __init__(self, data):
    self.values = self.binarize(np.array(data[VALUES_COLUMN]))

  def binarize(self, values):
    """Return 0/1 labels from values in {0, 0.5, 1} (0.5 maps to 1); raise on other values."""
    accepted = set([0, 0.5, 1])
    bin_values = []
    for v in values:
      if v not in accepted:
        raise Exception(
          f"Data is not binary (found value {v!r}; expected one of {sorted(accepted)}). "
          "Cannot do classification."
        )
      if v > 0:
        bin_values += [1]
      else:
        bin_values += [0]
    return bin_values


class SingleTasks(ZairaBase):
  """Setup step that turns ``values.csv`` into ``tasks.csv`` for the run's task type at fit."""

  def __init__(self, path):
    ZairaBase.__init__(self)
    self.path = path
    if self.is_predict():
      self.trained_path = self.get_trained_dir()
    else:
      self.trained_path = self.get_output_dir()
    self._task = ExpectedTaskType(path=path).get()

  def _get_params(self):
    params = ParametersFile(
      path=os.path.join(self.trained_path, METADATA_SUBFOLDER, PARAMETERS_FILE)
    )
    return params.load()

  def _get_data(self):
    df = pd.read_csv(os.path.join(self.path, VALUES_FILENAME))
    return df

  def run(self):
    """Write ``tasks.csv`` with the target: ``bin`` (classification) or ``val`` (regression)."""
    df = self._get_data()
    if self._task == "classification":
      self.logger.debug("It is simply a binary classification")
      df["bin"] = ClfTasks(data=df).values
    elif self._task == "regression":
      self.logger.debug("The target is continuous")
      df["val"] = RegTasks(data=df).validated()
    else:
      raise Exception("Task is not classification or regression, cannot proceed")
    df.to_csv(os.path.join(self.path, TASKS_FILENAME), index=False)


class SingleTasksForPrediction(SingleTasks):
  """Predict-time `SingleTasks` that tolerates non-binary ground truth."""

  def __init__(self, path):
    SingleTasks.__init__(self, path=path)

  def run(self):
    """Write ``tasks.csv`` with the target when the ground truth is usable.

    Ground truth that cannot be validated only logs a warning: the run then predicts without it.
    """
    logger.info(f"[tasks] Loading values data from {self.path}")
    df = self._get_data()
    logger.info(f"[tasks] Loaded {len(df)} records, task type: {self._task}")
    if self._task == "classification":
      self.logger.debug("It is simply a binary classification")
      logger.info("[tasks] Processing classification task")
      try:
        ct = ClfTasksForPrediction(data=df)
        df["bin"] = ct.values
      except Exception as e:
        # The ground-truth column isn't binary {0, 0.5, 1}. We can't validate a classifier against
        # raw values (the fit-time binarization cutoff isn't persisted), so warn and predict only:
        # without a `bin` column the report simply omits the performance section. The run continues.
        logger.warning(
          f"[tasks] Ground-truth labels are not binary ({e}); skipping validation, predicting only."
        )
      self._task = "classification"
    elif self._task == "regression":
      self.logger.debug("The target is continuous")
      logger.info("[tasks] Processing regression task")
      try:
        df["val"] = RegTasks(data=df).validated()
      except Exception as e:
        # No usable ground truth: predict without validation, as the classification path does.
        logger.warning(
          f"[tasks] Ground-truth values are unusable ({e}); skipping validation, predicting only."
        )
    out_file = os.path.join(self.path, TASKS_FILENAME)
    logger.info(f"[tasks] Writing tasks to {out_file}")
    df.to_csv(out_file, index=False)
    logger.info("[tasks] Tasks processing complete")
