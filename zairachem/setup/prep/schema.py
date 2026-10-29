import os

import pandas as pd
from rdkit import Chem

from zairachem.base import ZairaBase
from zairachem.base.utils.logging import logger

_SNIFF_SAMPLE_SIZE = 1000
_MIN_CORRECT = 0.8

#: The values a classification label may take (0.5 counts as positive).
BINARY_LABELS = (0, 0.5, 1)


class InputError(ValueError):
  """The input file cannot be used; the message says what is wrong and what to change."""


def target_kind(values):
  """Tell classification labels from a continuous target.

  Parameters
  ----------
  values : array-like
      The target values; missing ones are ignored.

  Returns
  -------
  tuple of (str, list)
      ``"binary"`` when every value is 0, 0.5 or 1; ``"ambiguous"`` when there are at most two
      distinct values and they are not all among those (1 and 2: a class label, or a number?);
      ``"continuous"`` otherwise. Then the sorted distinct values.
  """
  distinct = sorted(pd.Series(values).dropna().unique().tolist())
  if all(v in BINARY_LABELS for v in distinct):
    return "binary", distinct
  return ("ambiguous" if len(distinct) <= 2 else "continuous"), distinct


def not_binary_message(column, kind, distinct):
  """What to tell the user when classification is given a target that is not 0/1 labels."""
  if kind == "continuous":
    return (
      f"The target '{column}' is continuous ({len(distinct):,} distinct values), but classification "
      "needs 0/1 labels. Use --regression, or omit the flag to detect the task."
    )
  shown = ", ".join(str(v) for v in distinct)
  return (
    f"The target '{column}' has only two values ({shown}), not 0/1. Recode it to 0/1 and use "
    "--classification, or use --regression to model it as a number."
  )


class InputSchema(ZairaBase):
  """Column sniffer that finds the SMILES and activity-value columns of an input CSV."""

  def __init__(self, input_file):
    ZairaBase.__init__(self)
    self.logger = logger
    self.input_file = os.path.abspath(input_file)
    self.df_ = pd.read_csv(self.input_file, nrows=_SNIFF_SAMPLE_SIZE)
    self.columns = [c for c in list(self.df_.columns) if not self.df_[c].isnull().all()]
    self.assigned_columns = set()

  def _prop_correct_smiles(self, col):
    values = list(self.df_[col])[
      :10
    ]  # check only the first 10 smiles, if they are not correct continue
    c = 0
    for v in values:
      try:
        mol = Chem.MolFromSmiles(v)
      except Exception:
        continue
      if mol is not None:
        c += 1
    return float(c) / len(values)

  def _is_smiles_column(self, col):
    if "smiles" in col.lower():
      return True
    if self._prop_correct_smiles(col) > _MIN_CORRECT:
      return True
    else:
      return False

  def find_smiles_column(self):
    """Return, as a one-item list, the first column named like or parsing as SMILES.

    Raises
    ------
    InputError
      If the input has no data rows or no column that looks like SMILES (all-empty columns are
      ignored). It is a ``ValueError``.
    """
    name = os.path.basename(self.input_file)
    if self.df_.empty:
      raise InputError(f"The input '{name}' has no data rows.")
    cols = []
    for col in self.columns:
      if self._is_smiles_column(col):
        cols += [col]
      else:
        continue
    if not cols:
      looked = ", ".join(str(c) for c in self.columns) or "none with any values"
      raise InputError(f"No SMILES column found in '{name}' (columns looked at: {looked}).")
    if len(cols) > 1:
      logger.warning(
        f"More than one column contains smiles as header. Proceeding with first col: {cols[0]}"
      )
    return [cols[0]]

  def _is_values_column(self, col):
    try:
      values = list(self.df_[self.df_[col].notnull()][col])
      c = 0
      for v in values:
        try:
          float(v)
        except Exception:
          continue
        c += 1
      if c == len(values):
        return True
      else:
        return False
    except Exception:
      return False

  def find_values_column(self):
    """Return candidate activity columns: ``bin`` or ``exp`` if present, else all numeric ones.

    Numeric identifier-like columns (``id``, ``Unnamed: 0``) are skipped.
    """
    for col in self.columns:
      if col == "bin":
        return [col]
    for col in self.columns:
      if col == "exp":
        return [col]
    cols = []
    for col in self.columns:
      if self._is_values_column(col):
        if col.lower() == "id":
          continue
        if col == "Unnamed: 0":
          continue
        cols += [col]
      else:
        continue
    return cols

  def _why_not_values(self, col):
    """Short reason a column was not taken as the target, for the message when none is found."""
    series = self.df_[col].dropna()
    if series.empty:
      return "empty"
    if col.lower() == "id" or col == "Unnamed: 0":
      return "an identifier"
    for v in series:
      try:
        float(v)
      except Exception:
        return f"not numeric, e.g. '{v}'"
    return "not used"

  def missing_values_column_message(self, smiles_column):
    """Say that no target column was found, which columns were looked at and why each was passed over.

    Parameters
    ----------
    smiles_column : str
      The SMILES column, which is not a candidate.

    Returns
    -------
    str
      A one-line message for ``InputError``.
    """
    name = os.path.basename(self.input_file)
    others = [c for c in self.df_.columns if c != smiles_column]
    looked = "; ".join(f"{c} ({self._why_not_values(c)})" for c in others)
    return (
      f"No numeric or 0/1 target column found in '{name}'. Columns looked at: "
      f"{looked or 'none besides the SMILES column'}. The target must be numbers: recode text "
      "labels such as active/inactive as 1/0."
    )

  def find_values_column_by_name(self, target_name):
    """The input column whose header matches ``target_name`` (case-insensitive, trimmed), or ``[]``.

    Used at predict time to recognise a ground-truth column *only* when it is named exactly like the
    activity/output column used at fit — so unrelated extra columns are ignored (and never trigger the
    "more than one values column" path).
    """
    if not target_name:
      return []
    want = str(target_name).strip().lower()
    return [c for c in self.columns if str(c).strip().lower() == want]

  def resolve_columns(self, values_column_name=None):
    """Resolve the SMILES and values columns.

    When ``values_column_name`` is given (predict), the values column is matched by that exact name
    only (0 or 1 match by construction — no "more than one" assert). When ``None`` (fit), the values
    column is auto-detected as before.
    """
    smiles_column = self.find_smiles_column()
    assert len(smiles_column) == 1, "No SMILES column found!"
    smiles_column = smiles_column[0]
    if values_column_name is not None:
      values_column = self.find_values_column_by_name(values_column_name)
    else:
      values_column = self.find_values_column()
    self.logger.debug("Values column {0}".format(values_column))
    if not values_column:
      values_column = None
    elif values_column_name is not None:
      values_column = values_column[0]
    else:
      assert len(values_column) == 1, "More than one values column found! {0}".format(values_column)
      values_column = values_column[0]
    data = {}
    data["smiles_column"] = smiles_column
    data["values_column"] = values_column
    return data
