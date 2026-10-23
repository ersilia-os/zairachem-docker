import csv
import json
import os
import random

import pandas as pd
from rdkit import Chem, DataStructs

from zairachem.base.utils.logging import logger
from zairachem.base.utils.progress import SetupProgress
from zairachem.base.vars import (
  COMPOUND_IDENTIFIER_COLUMN,
  DATA_FILENAME,
  DATA_SUBFOLDER,
  INPUT_SCHEMA_FILENAME,
  MAPPING_DEDUPE_COLUMN,
  MAPPING_FILENAME,
  MAPPING_ORIGINAL_COLUMN,
  RAW_INPUT_FILENAME,
  REPLICATES_FILENAME,
  SMILES_COLUMN,
  VALUES_COLUMN,
)
from zairachem.setup.tools.chembl_structure.standardizer import (
  standardize_molblock_from_smiles,
)

MAX_CHECK_SAMPLES = 10000


class SetupChecker(object):
  """Setup step that checks ``data.csv`` against the raw input file through ``mapping.csv``."""

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
    """Rewrite ``mapping.csv`` so its dedupe indices point at rows of the final ``data.csv``.

    Compounds absent from ``data.csv`` (e.g. dropped by standardisation) get an empty index.
    """
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
    """Log molecules whose processed SMILES no longer resembles the input SMILES.

    Compares RDKit fingerprints (Tanimoto < 0.6, retried after ChEMBL standardisation of the input)
    on at most ``MAX_CHECK_SAMPLES`` mapped molecules and logs the number of discrepancies.
    """
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
    if discrepancies > n_total * 0.25:
      logger.warning(f"Number of discrepancies >25% ({discrepancies:,} of {n_total:,})")

  def check_activity(self):
    """Count processed activity values that differ from the raw input by more than 0.01.

    Skipped when the input has no values column; warns with the number of compounds whose duplicate
    rows disagreed on activity. Checks at most ``MAX_CHECK_SAMPLES`` mapped rows.
    """
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
      from zairachem.base.utils.console import echo, plural

      echo(
        f"{plural(len(conflicts), 'compound')} had duplicate rows with differing activities"
        f"{self._disagreement_phrase()}.",
        kind="warning",
      )

  def _disagreement_phrase(self):
    """How far the disagreeing repeats actually were apart, for the warning above, or ``""``.

    A count on its own leaves the only question it raises unanswered. Two measurements that
    differ in the third decimal are a rounding artefact; two that differ by four log units are
    two different experiments, and the spread between them is the floor under any model's error
    on this data — so it is the number that decides whether the count matters.
    """
    try:
      with open(self._replicates_path()) as f:
        summary = json.load(f)
    except (OSError, ValueError):
      return ""
    sd, widest = summary.get("pooled_sd"), summary.get("max_range")
    if sd is None:
      return ""
    phrase = f"; those repeats differ by {sd:.3f} on average"
    if widest is not None:
      phrase += f" and by as much as {widest:.2f}"
    return phrase + ", which is the smallest error a model on this data can have"

  def check_duplicate_structures(self):
    """Warn when two compounds became the same molecule once standardised.

    Deduplication runs on the SMILES as written, standardisation runs after it, and the two do not
    always agree: ``C1=CC=CC=C1C(=O)O`` and ``c1ccccc1C(=O)O`` are two rows going in and one
    molecule coming out. Those rows are then trained on twice, and a random held-out split can put
    one of them in train and its twin in test, where the model will appear to predict a molecule it
    has already seen.

    Counted, not corrected: collapsing them would change the identity of every row downstream, and
    which measurement to keep is the user's call. The count goes to ``replicates.json`` so the
    report can show it.
    """
    dd = pd.read_csv(self.data_file)
    if SMILES_COLUMN not in dd.columns or dd.empty:
      return
    counts = dd[SMILES_COLUMN].value_counts()
    repeated = counts[counts > 1]
    self._record_collisions(int(repeated.sum() - len(repeated)))
    if repeated.empty:
      return
    examples = ", ".join(str(s)[:40] for s in repeated.index[:3])
    logger.warning(
      f"[check] {len(repeated):,} molecules appear more than once in the standardised set "
      f"({int(repeated.sum()):,} rows in total): they were written differently in the input and "
      f"became identical here, e.g. {examples}. They are trained on more than once, and a held-out "
      "split can separate copies of one molecule, which flatters its scores."
    )

  def _replicates_path(self):
    """The run's replicate summary, which sits beside the deduplicated data it describes."""
    return os.path.join(os.path.dirname(self.data_file), REPLICATES_FILENAME)

  def _record_collisions(self, n_extra):
    """Add the count to ``replicates.json`` when the run has one. Best-effort."""
    path = self._replicates_path()
    from zairachem.base import _atomic_json

    try:
      with open(path) as f:
        record = json.load(f)
      record["standardised_collisions"] = n_extra
      _atomic_json(path, record)
    except (OSError, ValueError) as e:
      logger.debug(f"[check] could not record duplicate structures: {e}")

  def run(self):
    """Remap ``mapping.csv``, then run the SMILES, duplicate and activity consistency checks."""
    self.remap()
    self.check_smiles()
    self.check_duplicate_structures()
    self.check_activity()
