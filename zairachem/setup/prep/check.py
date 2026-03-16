import csv, json, os
import random
import pandas as pd
from rdkit import DataStructs
from rdkit import Chem
from standardiser import standardise
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
  INPUT_SCHEMA_FILENAME,
  RAW_INPUT_FILENAME,
  MAPPING_FILENAME,
  COMPOUND_IDENTIFIER_COLUMN,
  MAPPING_ORIGINAL_COLUMN,
  MAPPING_DEDUPE_COLUMN,
  VALUES_COLUMN,
  SMILES_COLUMN,
  DATA_SUBFOLDER,
  DATA_FILENAME,
)
from zairachem.base.utils.logging import logger

MAX_CHECK_SAMPLES = 10000


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
    transient=True,
  )


class SetupChecker(object):
  def __init__(self, path):
    for f in os.listdir(path):
      if RAW_INPUT_FILENAME in f:
        self.input_file = os.path.join(path, f)
    self.input_schema = os.path.join(path, DATA_SUBFOLDER, INPUT_SCHEMA_FILENAME)
    self.data_file = os.path.join(path, DATA_SUBFOLDER, DATA_FILENAME)
    self.mapping_file = os.path.join(path, DATA_SUBFOLDER, MAPPING_FILENAME)

  def _get_input_schema(self):
    with open(self.input_schema, "r") as f:
      self.input_schema_dict = json.load(f)

  def remap(self):
    logger.info("[check] Remapping indices")
    dm = pd.read_csv(self.mapping_file)
    dd = pd.read_csv(self.data_file)
    cid_mapping = list(dm[COMPOUND_IDENTIFIER_COLUMN])
    cid_data = list(dd[COMPOUND_IDENTIFIER_COLUMN])
    cid_data_idx = {cid: i for i, cid in enumerate(cid_data)}
    new_idxs = [cid_data_idx.get(cid, "") for cid in cid_mapping]
    orig_idxs = list(dm[MAPPING_ORIGINAL_COLUMN])
    with open(self.mapping_file, "w") as f:
      writer = csv.writer(f, delimiter=",")
      writer.writerow([
        MAPPING_ORIGINAL_COLUMN,
        MAPPING_DEDUPE_COLUMN,
        COMPOUND_IDENTIFIER_COLUMN,
      ])
      for o, u, c in zip(orig_idxs, new_idxs, cid_mapping):
        writer.writerow([o, u, c])
    logger.info(f"[check] Remapped {len(cid_mapping):,} entries")

  def check_smiles(self):
    self._get_input_schema()
    input_smiles_column = self.input_schema_dict["smiles_column"]
    di = pd.read_csv(self.input_file)
    dd = pd.read_csv(self.data_file)
    ismi = list(di[input_smiles_column])
    dsmi = list(dd[SMILES_COLUMN])
    mapping = pd.read_csv(self.mapping_file)
    valid_rows = [
      (oidx, uidx, cid)
      for oidx, uidx, cid in mapping.values
      if str(oidx) != "nan" and str(uidx) != "nan"
    ]
    n_total = len(valid_rows)
    if n_total > MAX_CHECK_SAMPLES:
      logger.info(
        f"[check] Sampling {MAX_CHECK_SAMPLES:,} of {n_total:,} molecules for SMILES check"
      )
      valid_rows = random.sample(valid_rows, MAX_CHECK_SAMPLES)
      n_total = MAX_CHECK_SAMPLES
    else:
      logger.info(f"[check] Checking {n_total:,} molecules for SMILES consistency")
    discrepancies = 0
    with _create_progress() as progress:
      task = progress.add_task("Checking SMILES", total=n_total)
      for oidx, uidx, cid in valid_rows:
        oidx = int(oidx)
        uidx = int(uidx)
        omol = Chem.MolFromSmiles(ismi[oidx])
        umol = Chem.MolFromSmiles(dsmi[uidx])
        if omol is None or umol is None:
          progress.update(task, advance=1)
          continue
        ofp = Chem.RDKFingerprint(omol)
        ufp = Chem.RDKFingerprint(umol)
        sim = DataStructs.FingerprintSimilarity(ofp, ufp)
        if sim < 0.6:
          try:
            omol = standardise.run(omol)
          except:
            progress.update(task, advance=1)
            continue
          ofp = Chem.RDKFingerprint(omol)
          sim = DataStructs.FingerprintSimilarity(ofp, ufp)
          if sim < 0.6:
            logger.warning(f"[bold yellow]Low similarity[/]: {sim} {cid} {ismi[oidx]} {dsmi[uidx]}")
            discrepancies += 1
        progress.update(task, advance=1)
    logger.info(f"[check] Found {discrepancies:,} discrepancies in {n_total:,} checked molecules")
    assert discrepancies < n_total * 0.25

  def check_activity(self):
    self._get_input_schema()
    input_values_column = self.input_schema_dict["values_column"]
    if input_values_column is None:
      logger.info("[check] No values column, skipping activity check")
      return
    di = pd.read_csv(self.input_file)
    dd = pd.read_csv(self.data_file)
    ival = list(di[input_values_column])
    dval = list(dd[VALUES_COLUMN])
    mapping = pd.read_csv(self.mapping_file)
    valid_rows = [
      (oidx, uidx, cid)
      for oidx, uidx, cid in mapping.values
      if str(oidx) != "nan" and str(uidx) != "nan"
    ]
    n_total = len(valid_rows)
    if n_total > MAX_CHECK_SAMPLES:
      logger.info(f"[check] Sampling {MAX_CHECK_SAMPLES:,} of {n_total:,} for activity check")
      valid_rows = random.sample(valid_rows, MAX_CHECK_SAMPLES)
      n_total = MAX_CHECK_SAMPLES
    else:
      logger.info(f"[check] Checking {n_total:,} activity values")
    discrepancies = 0
    with _create_progress() as progress:
      task = progress.add_task("Checking activity", total=n_total)
      for oidx, uidx, cid in valid_rows:
        oidx = int(oidx)
        uidx = int(uidx)
        difference = abs(ival[oidx] - dval[uidx])
        if difference > 0.01:
          logger.warning(
            f"[bold yellow]High activity difference detected[/]: {difference} {cid} {oidx} {uidx}"
          )
          discrepancies += 1
        progress.update(task, advance=1)
    logger.info(f"[check] Found {discrepancies:,} activity discrepancies")

  def run(self):
    self.remap()
    self.check_smiles()
    self.check_activity()
