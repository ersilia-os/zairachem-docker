import os
import pandas as pd
from rich.progress import (
  Progress,
  ProgressColumn,
  SpinnerColumn,
  TextColumn,
  TimeElapsedColumn,
  TimeRemainingColumn,
  MofNCompleteColumn,
)
from rich.progress_bar import ProgressBar
from zairachem.base.vars import (
  STANDARD_COMPOUNDS_FILENAME,
  FOLDS_FILENAME,
  TASKS_FILENAME,
  DATA_FILENAME,
  COMPOUND_IDENTIFIER_COLUMN,
  SMILES_COLUMN,
  STANDARD_SMILES_COLUMN,
)
from zairachem.base.utils.logging import logger


class _PulseBarColumn(ProgressColumn):
  def __init__(
    self,
    bar_width: int = 40,
    style: str = "bar.back",
    complete_style: str = "bar.complete",
    finished_style: str = "bar.finished",
    pulse_style: str = "bar.pulse",
  ) -> None:
    super().__init__()
    self.bar_width = int(bar_width)
    self.style = style
    self.complete_style = complete_style
    self.finished_style = finished_style
    self.pulse_style = pulse_style

  def render(self, task) -> ProgressBar:
    return ProgressBar(
      total=task.total,
      completed=task.completed,
      width=max(1, self.bar_width),
      pulse=not task.finished,
      animation_time=task.get_time(),
      style=self.style,
      complete_style=self.complete_style,
      finished_style=self.finished_style,
      pulse_style=self.pulse_style,
    )


def _create_progress():
  return Progress(
    SpinnerColumn(),
    TextColumn("[bold blue]{task.description}"),
    _PulseBarColumn(bar_width=40),
    MofNCompleteColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    transient=False,
  )


class DataMerger(object):
  def __init__(self, path):
    self.path = path

  def get_standard_smiles(self):
    return pd.read_csv(os.path.join(self.path, STANDARD_COMPOUNDS_FILENAME))

  def get_folds(self):
    return pd.read_csv(os.path.join(self.path, FOLDS_FILENAME))

  def get_tasks(self):
    return pd.read_csv(os.path.join(self.path, TASKS_FILENAME))

  def run(self):
    df1 = self.get_standard_smiles()
    df2 = self.get_folds()
    df = pd.merge(df1, df2, on=[COMPOUND_IDENTIFIER_COLUMN, SMILES_COLUMN, STANDARD_SMILES_COLUMN])
    df = df.drop(columns=[SMILES_COLUMN])
    df = df.rename(columns={STANDARD_SMILES_COLUMN: SMILES_COLUMN})
    df_tsk = self.get_tasks()
    df = df.merge(df_tsk, on=COMPOUND_IDENTIFIER_COLUMN)
    df.to_csv(os.path.join(self.path, DATA_FILENAME), index=False)


class DataMergerForPrediction(object):
  def __init__(self, path):
    self.path = path

  def run(self, has_tasks):
    cpd_file = os.path.join(self.path, STANDARD_COMPOUNDS_FILENAME)
    out_file = os.path.join(self.path, DATA_FILENAME)
    with _create_progress() as progress:
      if not has_tasks:
        task = progress.add_task("Merging data", total=3)
        df = pd.read_csv(cpd_file, usecols=[COMPOUND_IDENTIFIER_COLUMN, STANDARD_SMILES_COLUMN])
        progress.update(task, advance=1, description=f"Loaded {len(df):,} compounds")
        df = df.rename(columns={STANDARD_SMILES_COLUMN: SMILES_COLUMN})
        progress.update(task, advance=1, description="Writing merged data")
        df.to_csv(out_file, index=False)
        progress.update(task, advance=1, description="Merge complete")
      else:
        task = progress.add_task("Merging data", total=5)
        df_cpd = pd.read_csv(cpd_file, usecols=[COMPOUND_IDENTIFIER_COLUMN, STANDARD_SMILES_COLUMN])
        progress.update(task, advance=1, description=f"Loaded {len(df_cpd):,} compounds")
        df_cpd = df_cpd.rename(columns={STANDARD_SMILES_COLUMN: SMILES_COLUMN})
        tsk_file = os.path.join(self.path, TASKS_FILENAME)
        df_tsk = pd.read_csv(tsk_file)
        progress.update(task, advance=1, description=f"Loaded {len(df_tsk):,} task records")
        df = df_cpd.merge(df_tsk, on=COMPOUND_IDENTIFIER_COLUMN)
        progress.update(task, advance=1, description=f"Merged {len(df):,} records")
        df.to_csv(out_file, index=False)
        progress.update(task, advance=1, description="Writing merged data")
        progress.update(task, advance=1, description="Merge complete")
    logger.info(f"[merge] Saved {len(df):,} records to {out_file}")
