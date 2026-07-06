import csv, json, os
import random
import pandas as pd
from rdkit import DataStructs
from rdkit import Chem
from zairachem.setup.tools.chembl_structure.standardizer import (
  standardize_molblock_from_smiles,
)
from zairachem.base.utils.progress import SetupProgress
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


class SetupChecker(object):
  def __init__(self, path):
    input_dir = os.path.join(path, DATA_SUBFOLDER)
    for f in os.listdir(input_dir):
      if RAW_INPUT_FILENAME in f:
        self.input_file = os.path.join(input_dir, f)
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
    with SetupProgress() as progress:
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
            std_smi = standardize_molblock_from_smiles(ismi[oidx], get_smiles=True)
            omol = Chem.MolFromSmiles(std_smi) if std_smi else None
          except Exception as e:
            logger.debug(f"[check] standardization failed during similarity check: {e}")
            progress.update(task, advance=1)
            continue
          if omol is None:
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
    # Distinct compounds whose duplicate rows disagreed on activity (collapsed across their rows).
    conflicts = set()
    with SetupProgress() as progress:
      task = progress.add_task("Checking activity", total=n_total)
      for oidx, uidx, cid in valid_rows:
        oidx = int(oidx)
        uidx = int(uidx)
        difference = abs(ival[oidx] - dval[uidx])
        if difference > 0.01:
          discrepancies += 1
          conflicts.add(cid)
        progress.update(task, advance=1)
    logger.info(f"[check] Found {discrepancies:,} activity discrepancies")
    if conflicts:
      # Just a count of compounds whose duplicate rows disagreed on activity. Informational.
      from zairachem.base.utils.console import echo

      echo(
        f"{len(conflicts):,} compound(s) had duplicate rows with differing activities.",
        kind="warning",
      )

  def run(self):
    self.remap()
    self.check_smiles()
    self.check_activity()
