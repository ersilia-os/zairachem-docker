import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

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


from zairachem.base.vars import (
  SMILES_COLUMN,
  COMPOUNDS_FILENAME,
  COMPOUND_IDENTIFIER_COLUMN,
  STANDARD_SMILES_COLUMN,
)
from zairachem.setup.tools.chembl_structure.standardizer import (
  standardize_molblock_from_smiles,
)
from zairachem.base.utils.logging import logger

DEFAULT_BATCH_SIZE = 1000
MAX_WORKERS = None


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


def _standardize_single(smi):
  try:
    st_smi = standardize_molblock_from_smiles(smi, get_smiles=True)
  except:
    st_smi = smi
  return st_smi if st_smi is not None else smi


def _standardize_batch(batch_data):
  results = []
  for identifier, smi in batch_data:
    st_smi = _standardize_single(smi)
    if st_smi is not None:
      results.append([identifier, smi, st_smi])
  return results


class ChemblStandardize(object):
  def __init__(self, outdir, batch_size=DEFAULT_BATCH_SIZE, max_workers=MAX_WORKERS):
    self.outdir = outdir
    self.input_file = self.get_input_file()
    self.output_file = self.get_output_file()
    self.batch_size = batch_size
    self.max_workers = max_workers

  def get_output_file(self):
    return os.path.join(self.outdir, COMPOUNDS_FILENAME.split(".")[0] + "_std.csv")

  def get_input_file(self):
    return os.path.join(self.outdir, COMPOUNDS_FILENAME)

  def _run_sequential(self, df):
    R = []
    n_total = len(df)
    with _create_progress() as progress:
      task = progress.add_task("Standardizing molecules", total=n_total)
      for idx, r in enumerate(df.values):
        identifier = r[0]
        smi = r[1]
        st_smi = _standardize_single(smi)
        if st_smi is not None:
          R.append([identifier, smi, st_smi])
        progress.update(task, advance=1)
    return R

  def _run_parallel(self, df):
    n_total = len(df)
    data = [(r[0], r[1]) for r in df.values]
    n_batches = (n_total + self.batch_size - 1) // self.batch_size
    batches = []
    for i in range(n_batches):
      start = i * self.batch_size
      end = min(start + self.batch_size, n_total)
      batches.append(data[start:end])
    logger.info(
      f"[standardize] Processing {n_total:,} molecules in {n_batches:,} batches (parallel)"
    )
    R = []
    with _create_progress() as progress:
      task = progress.add_task("Standardizing molecules", total=n_batches)
      with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
        futures = {executor.submit(_standardize_batch, batch): i for i, batch in enumerate(batches)}
        for future in as_completed(futures):
          batch_idx = futures[future]
          try:
            batch_results = future.result()
            R.extend(batch_results)
            progress.update(task, advance=1)
          except Exception as e:
            logger.error(f"[standardize] Batch {batch_idx} failed: {e}")
            progress.update(task, advance=1)
    return R

  def _run_vectorized(self, df):
    n_total = len(df)
    logger.info(f"[standardize] Processing {n_total:,} molecules")
    R = []
    with _create_progress() as progress:
      task = progress.add_task("Standardizing molecules", total=n_total)
      for r in df.values:
        identifier = r[0]
        smi = r[1]
        st_smi = _standardize_single(smi)
        if st_smi is not None:
          R.append([identifier, smi, st_smi])
        progress.update(task, advance=1)
    return R

  def run(self):
    df = pd.read_csv(self.input_file)
    n_total = len(df)
    logger.info(f"[standardize] Starting standardization of {n_total:,} compounds")
    if n_total > 5000:
      try:
        R = self._run_parallel(df)
      except Exception as e:
        logger.warning(f"[standardize] Parallel processing failed, falling back to sequential: {e}")
        R = self._run_vectorized(df)
    else:
      R = self._run_vectorized(df)
    df = pd.DataFrame(
      R, columns=[COMPOUND_IDENTIFIER_COLUMN, SMILES_COLUMN, STANDARD_SMILES_COLUMN]
    )
    df.to_csv(self.output_file, index=False)
    logger.info(f"[standardize] Saved {len(df):,} standardized compounds to {self.output_file}")
