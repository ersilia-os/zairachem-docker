import collections
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from rdkit import Chem

from zairachem.base.utils.console import echo
from zairachem.base.utils.logging import logger
from zairachem.base.utils.progress import SetupProgress
from zairachem.base.vars import (
  COMPOUND_IDENTIFIER_COLUMN,
  COMPOUNDS_FILENAME,
  DATA_SUBFOLDER,
  INPUT_SCHEMA_FILENAME,
  MAPPING_DEDUPE_COLUMN,
  MAPPING_FILENAME,
  MAPPING_ORIGINAL_COLUMN,
  REPLICATES_FILENAME,
  SMILES_COLUMN,
  VALUES_COLUMN,
  VALUES_FILENAME,
)
from zairachem.setup.prep.schema import InputSchema

DEDUPE_BATCH_SIZE = 5000
DEDUPE_MAX_WORKERS = None


def _validate_smiles_batch(batch_data):
  results = []
  for idx, cid, smi in batch_data:
    mol = Chem.MolFromSmiles(smi) if smi else None
    results.append((idx, cid, smi, mol is not None))
  return results


def replicate_summary(groups, task):
  """How often the input measured the same molecule twice, and how much those repeats disagreed.

  Parameters
  ----------
  groups : dict
    ``{compound id: [each measurement the input gave it]}``, before they are collapsed to one.
    Blank cells have already been dropped, so a compound here has only the values it really has.
  task : str
    ``"classification"`` or ``"regression"``.

  Returns
  -------
  dict
    ``task``, ``n_rows``, ``n_compounds``, ``n_repeated`` and ``max_repeats`` always. Regression
    adds the
    pooled within-compound standard deviation (``pooled_sd``), the widest single disagreement
    (``max_range``) and the spread across compounds (``target_sd``); classification adds how many
    repeated compounds were given both labels (``disagreeing``).

  Notes
  -----
  The pooled within-compound standard deviation is the assay's own reproducibility, in the target's
  units and so on the same scale as the model's RMSE. That is what makes it the number an RMSE has
  to be read against: a model predicting within the spread of repeat measurements of one molecule
  is at the limit of its data, and no amount of modelling will improve it.
  """
  sizes = [len(v) for v in groups.values()]
  repeated = [np.asarray(v, dtype=float) for v in groups.values() if len(v) > 1]
  summary = {
    "task": task,
    "n_rows": int(sum(sizes)),
    "n_compounds": len(groups),
    "n_repeated": len(repeated),
    "max_repeats": int(max(sizes)) if sizes else 0,
  }
  if not repeated:
    return summary
  if task == "classification":
    summary["disagreeing"] = int(sum(1 for v in repeated if len(set(v.tolist())) > 1))
    return summary
  dof = int(sum(len(v) - 1 for v in repeated))
  if dof:
    ss = float(sum(np.sum((v - v.mean()) ** 2) for v in repeated))
    summary["pooled_sd"] = float(np.sqrt(ss / dof))
  summary["max_range"] = float(max(np.ptp(v) for v in repeated))
  # The spread across compounds is the signal the model has to find; the spread within one is the
  # noise it has to find it through. Their ratio is what says whether the dataset is learnable.
  across = np.asarray([np.median(v) for v in groups.values()], dtype=float)
  across = across[np.isfinite(across)]
  if across.size > 1:
    summary["target_sd"] = float(across.std(ddof=1))
  return summary


def read_replicates(run_dir):
  """The run's replicate summary, or ``{}`` when it has none (older runs, or no duplicates)."""
  path = os.path.join(run_dir, DATA_SUBFOLDER, REPLICATES_FILENAME)
  try:
    with open(path) as f:
      return json.load(f)
  except (OSError, ValueError):
    return {}


class ModelIdsFile(object):
  """Resolve the ``--eos-ids`` value into ``{featurizer_ids, projection_ids?}``.

  The value may be either a JSON file (``{"featurizer_ids": [...], "projection_ids": [...]}``) or
  one/more Ersilia Model Hub IDs given directly (e.g. ``eos8aa5`` or ``eos8aa5,eos3l5f``). Raw IDs
  are treated as featurizers; the projection then falls back to the default.
  """

  _EOS_RE = re.compile(r"^eos[0-9a-z]{4}$")

  def __init__(self, value):
    self.value = value
    self.path = os.path.abspath(value)
    self.data = None

  @classmethod
  def parse_ids(cls, value, flag="--featurizer-ids"):
    """Parse comma/space-separated Ersilia Model Hub IDs into a validated list.

    Raises ``ValueError`` (naming ``flag``) if any token isn't a valid ``eosXXXX`` id.
    """
    ids = [t for t in re.split(r"[,\s]+", str(value).strip()) if t]
    bad = [t for t in ids if not cls._EOS_RE.match(t)]
    if not ids or bad:
      offenders = ", ".join(bad) if bad else "(empty)"
      raise ValueError(
        f"{flag} '{value}' is not valid Ersilia Model Hub IDs (offending: {offenders}). "
        f"Pass IDs like 'eos8aa5' or 'eos8aa5,eos3l5f'."
      )
    return ids

  def load(self):
    """Return the model-ID dict, read from the JSON file or parsed from the raw IDs (cached)."""
    if self.data is not None:
      return self.data
    if os.path.isfile(self.path):
      with open(self.path, "r") as f:
        self.data = json.load(f)
      return self.data
    # Not a file: parse the value as comma/space-separated featurizer model IDs.
    self.data = {"featurizer_ids": self.parse_ids(self.value, flag="--featurizer-ids")}
    return self.data


class ParametersFile(object):
  """Lazily loaded JSON run-parameters file (e.g. ``metadata/parameters.json``)."""

  def __init__(self, path):
    self.path = os.path.abspath(path)
    self.params = None

  def load(self):
    """Return the parameters dict, reading the JSON file on first call only."""
    if self.params is None:
      with open(self.path, "r") as f:
        self.params = json.load(f)
    return self.params


class SingleFile(InputSchema):
  """Fit-time reader that turns a single input CSV into compound, value and mapping tables."""

  #: Repeat noise above this share of the spread across compounds leaves little to learn.
  NOISY_DATA_RATIO = 0.8

  def __init__(self, input_file, params, path):
    InputSchema.__init__(self, input_file)
    self.path = path  # the run root, so get_output_dir()/get_trained_dir() resolve the session
    self.params = params
    self.df = pd.read_csv(input_file)

  def _make_identifiers(self):
    # Preserve input order (dict.fromkeys dedups, keeping first occurrence) so compound_id
    # assignment — and therefore data.csv row order — is deterministic across runs.
    # set() iteration order varies per process (string hash randomization).
    all_smiles = list(
      dict.fromkeys(self.df[self.df[self.smiles_column].notnull()][self.smiles_column])
    )
    smiles2identifier = {}
    n = len(str(len(all_smiles)))
    for i, smi in enumerate(all_smiles):
      identifier = "CID{0}".format(str(i).zfill(n))
      smiles2identifier[smi] = identifier
    identifiers = []
    for smi in list(self.df[self.smiles_column]):
      if smi in smiles2identifier:
        identifiers += [smiles2identifier[smi]]
      else:
        identifiers += [None]
    return identifiers

  def normalize_dataframe(self):
    """Return one row per input row with compound id, SMILES and activity value.

    Resolves the SMILES and values columns and assigns a ``CID…`` identifier per unique SMILES
    (empty SMILES get None).
    """
    resolved_columns = self.resolve_columns()
    self.smiles_column = resolved_columns["smiles_column"]
    self.values_column = resolved_columns["values_column"]
    identifiers = self._make_identifiers()
    df = pd.DataFrame({COMPOUND_IDENTIFIER_COLUMN: identifiers})
    df[SMILES_COLUMN] = self.df[self.smiles_column]
    df[VALUES_COLUMN] = self.df[self.values_column]
    assert df.shape[0] == self.df.shape[0]
    return df

  def _validate_smiles_sequential(self, data):
    results = []
    n_total = len(data)
    with SetupProgress() as progress:
      task = progress.add_task("Validating SMILES", total=n_total)
      for idx, cid, smi in data:
        mol = Chem.MolFromSmiles(smi) if smi else None
        results.append((idx, cid, smi, mol is not None))
        progress.update(task, advance=1)
    return results

  def _validate_smiles_parallel(self, data):
    n_total = len(data)
    n_batches = (n_total + DEDUPE_BATCH_SIZE - 1) // DEDUPE_BATCH_SIZE
    batches = []
    for i in range(n_batches):
      start = i * DEDUPE_BATCH_SIZE
      end = min(start + DEDUPE_BATCH_SIZE, n_total)
      batches.append(data[start:end])
    logger.info(f"[dedupe] Validating {n_total:,} SMILES in {n_batches:,} batches (parallel)")
    results = []
    with SetupProgress() as progress:
      task = progress.add_task("Validating SMILES", total=n_batches)
      with ProcessPoolExecutor(max_workers=DEDUPE_MAX_WORKERS) as executor:
        futures = {
          executor.submit(_validate_smiles_batch, batch): i for i, batch in enumerate(batches)
        }
        for future in as_completed(futures):
          batch_idx = futures[future]
          try:
            batch_results = future.result()
            results.extend(batch_results)
            progress.update(task, advance=1)
          except Exception as e:
            logger.error(f"[dedupe] Batch {batch_idx} failed: {e}")
            progress.update(task, advance=1)
    return results

  def dedupe(self, df, path):
    """Drop invalid SMILES, write ``mapping.csv`` to ``path`` and return the unique compounds.

    The returned frame has one row per compound id (sorted) with its SMILES; ``mapping.csv`` links
    each input row to its deduplicated row index and compound id (empty for invalid SMILES).
    """
    n_total = df.shape[0]
    logger.info(f"[dedupe] Starting deduplication of {n_total:,} compounds")
    data = [
      (i, r[0], r[1]) for i, r in enumerate(df[[COMPOUND_IDENTIFIER_COLUMN, SMILES_COLUMN]].values)
    ]
    if n_total > 10000:
      try:
        validated = self._validate_smiles_parallel(data)
      except Exception as e:
        logger.warning(f"[dedupe] Parallel validation failed, falling back to sequential: {e}")
        validated = self._validate_smiles_sequential(data)
    else:
      validated = self._validate_smiles_sequential(data)
    mapping = collections.defaultdict(list)
    cid2smiles = {}
    for idx, cid, smi, is_valid in validated:
      if not is_valid:
        continue
      mapping[cid].append(idx)
      cid2smiles[cid] = smi
    unique_cids = sorted(set(mapping.keys()))
    unique_cids_idx = {k: i for i, k in enumerate(unique_cids)}
    idx_to_cid = {x: k for k, v in mapping.items() for x in v}
    R = []
    for i in range(n_total):
      if i in idx_to_cid:
        cid = idx_to_cid[i]
        R.append([i, unique_cids_idx[cid], cid])
      else:
        R.append([i, None, None])
    dfm = pd.DataFrame(
      R,
      columns=[
        MAPPING_ORIGINAL_COLUMN,
        MAPPING_DEDUPE_COLUMN,
        COMPOUND_IDENTIFIER_COLUMN,
      ],
    )
    dfm.to_csv(os.path.join(path, MAPPING_FILENAME), index=False)
    R = [[cid, cid2smiles[cid]] for cid in unique_cids]
    dfc = pd.DataFrame(R, columns=[COMPOUND_IDENTIFIER_COLUMN, SMILES_COLUMN])
    logger.info(f"[dedupe] Deduplication complete: {n_total:,} -> {len(dfc):,} unique compounds")
    return dfc

  def compounds_table(self, df, path):
    """Return the deduplicated compounds table, writing ``mapping.csv`` to ``path``."""
    dfc = self.dedupe(df, path)
    return dfc

  def _write_replicates(self, path):
    """Persist the replicate summary and say in one line what the duplicates were."""
    summary = getattr(self, "replicates", None)
    if not summary:
      return
    from zairachem.base import _atomic_json

    _atomic_json(os.path.join(path, REPLICATES_FILENAME), summary)
    if not summary["n_repeated"]:
      return
    message = (
      f"[setup] {summary['n_rows']:,} rows cover {summary['n_compounds']:,} compounds; "
      f"{summary['n_repeated']:,} were measured more than once "
      f"(up to {summary['max_repeats']}) and their values were averaged"
    )
    sd = summary.get("pooled_sd")
    if sd is not None:
      message += f"; repeats of one compound differ by {sd:.3f} on average"
    disagreeing = summary.get("disagreeing")
    if disagreeing:
      message += f"; {disagreeing:,} were given both labels"
    logger.info(message + ".")
    self._warn_if_noise_rivals_signal(summary)

  def _warn_if_noise_rivals_signal(self, summary):
    """Say so before a run spends an hour on descriptors it cannot use.

    The spread across compounds is the signal a model has to find; the spread between repeats of
    one compound is the noise it has to find it through. When the second approaches the first there
    is very little left to learn, and that is worth knowing at setup rather than from a poor R2 an
    hour later.
    """
    noise, signal = summary.get("pooled_sd"), summary.get("target_sd")
    if not noise or not signal or signal <= 0:
      return
    ratio = noise / signal
    if ratio < self.NOISY_DATA_RATIO:
      return
    echo(
      f"Repeat measurements of the same compound differ by {noise:.3f}, almost as much as the "
      f"target varies across compounds ({signal:.3f}). There is little signal here for any model "
      "to find; check that the rows being merged are really the same measurement.",
      kind="warning",
    )

  def values_table(self, df):
    """Return one activity value per compound id: the median over duplicate rows.

    For classification the median is rounded back to a 0/1 label (a 0.5 tie becomes 1). The spread
    of those duplicate rows is kept (:attr:`replicates`) before they are collapsed, since it is the
    only measurement of the assay's own reproducibility the pipeline ever sees.
    """
    dfv = pd.DataFrame({COMPOUND_IDENTIFIER_COLUMN: df[COMPOUND_IDENTIFIER_COLUMN]})
    dfv[VALUES_COLUMN] = df[VALUES_COLUMN]
    dedupe = collections.defaultdict(list)
    for r in dfv[
      [
        COMPOUND_IDENTIFIER_COLUMN,
        VALUES_COLUMN,
      ]
    ].values:
      dedupe[r[0]] += [(r[1])]
    # Aggregate duplicate compounds by the median of their activity values. For CLASSIFICATION the
    # values are binary labels, so round the median back to a 0/1 label (a tie of 0.5 → 1); for
    # REGRESSION keep the continuous median as-is.
    task = (self.params or {}).get("task")
    # A blank cell is not a measurement. Taking the median across it turns a compound that WAS
    # measured into a missing value, which then fails the whole run on the next step; drop the
    # blanks within each compound first and keep its real values. A compound with nothing but
    # blanks stays missing, which is what it is.
    dedupe = {k: [x for x in v if x == x] or [float("nan")] for k, v in dedupe.items()}
    self.replicates = replicate_summary(dedupe, task)
    R = []
    for k, v in dedupe.items():
      m = float(np.median(list(v)))
      if task == "classification":
        m = 1 if m >= 0.5 else 0
      R += [[k, m]]
    dfv = pd.DataFrame(
      R,
      columns=[
        COMPOUND_IDENTIFIER_COLUMN,
        VALUES_COLUMN,
      ],
    )
    return dfv

  def input_schema(self):
    """Return the input file path and resolved SMILES/values column names as a dict."""
    sc = {
      "input_file": self.input_file,
      "smiles_column": self.smiles_column,
      "values_column": self.values_column,
    }
    return sc

  def process(self):
    """Write ``compounds.csv``, ``values.csv``, ``mapping.csv`` and ``input_schema.json``.

    All files go to the data subfolder of the run's output directory.
    """
    path = os.path.join(self.get_output_dir(), DATA_SUBFOLDER)
    df = self.normalize_dataframe()
    dfc = self.compounds_table(df, path)
    dfc.to_csv(os.path.join(path, COMPOUNDS_FILENAME), index=False)
    dfv = self.values_table(df)
    dfv.to_csv(os.path.join(path, VALUES_FILENAME), index=False)
    self._write_replicates(path)
    schema = self.input_schema()
    with open(os.path.join(path, INPUT_SCHEMA_FILENAME), "w") as f:
      json.dump(schema, f, indent=4)


class SingleFileForPrediction(SingleFile):
  """Predict-time `SingleFile` that reads ground truth only from the fit-time values column."""

  def __init__(self, input_file, params, path):
    SingleFile.__init__(self, input_file, params, path)
    self.trained_path = self.get_trained_dir()

  def get_trained_values_column(self):
    """Return the values column name recorded in the trained model's ``input_schema.json``."""
    with open(os.path.join(self.trained_path, DATA_SUBFOLDER, INPUT_SCHEMA_FILENAME), "r") as f:
      return json.load(f)["values_column"]

  def normalize_dataframe(self):
    """Return compound ids and SMILES, plus values when a ground-truth column is found.

    The ground-truth column must be named like the fit-time values column; sets ``self.has_tasks``.
    """
    # Recognise a ground-truth column ONLY when it is named exactly like the fit-time activity column
    # (case-insensitive, trimmed). Any other extra columns in the input are ignored — so prediction
    # never depends on, or crashes over, unrelated numeric columns.
    trained_values_column = self.get_trained_values_column() if self.params is not None else None
    resolved_columns = self.resolve_columns(values_column_name=trained_values_column)
    self.smiles_column = resolved_columns["smiles_column"]
    self.values_column = resolved_columns["values_column"]
    identifiers = self._make_identifiers()
    df = pd.DataFrame({COMPOUND_IDENTIFIER_COLUMN: identifiers})
    df[SMILES_COLUMN] = self.df[self.smiles_column]
    assert df.shape[0] == self.df.shape[0]

    if self.values_column is not None:
      df[VALUES_COLUMN] = self.df[self.values_column]
      self.has_tasks = True
      self.logger.info(
        f"[setup] Ground-truth column '{self.values_column}' found — predictions will be validated."
      )
    else:
      self.has_tasks = False

    return df

  def input_schema(self):
    """Return the input file path and resolved SMILES/values column names as a dict."""
    sc = {
      "input_file": self.input_file,
      "smiles_column": self.smiles_column,
      "values_column": self.values_column,
    }
    return sc

  def process(self):
    """Write compounds, mapping and ``input_schema.json``, plus ``values.csv`` if any are labelled.

    If the ground-truth column is entirely blank, it is dropped (``values_column`` becomes None).
    """
    path = os.path.join(self.get_output_dir(), DATA_SUBFOLDER)
    df = self.normalize_dataframe()
    dfc = self.dedupe(df, path)
    dfc.to_csv(os.path.join(path, COMPOUNDS_FILENAME), index=False)
    if self.has_tasks:
      # Validate only on labelled compounds — rows with a blank ground-truth value still get a
      # prediction but contribute no truth. If nothing is labelled, fall back to predict-only.
      labeled = df[df[VALUES_COLUMN].notnull()]
      if len(labeled):
        dfv = self.values_table(labeled)
        dfv.to_csv(os.path.join(path, VALUES_FILENAME), index=False)
        self._write_replicates(path)
      else:
        # A ground-truth column was matched but every value is blank → nothing to validate against.
        # Drop it from the persisted schema too, so ``values_column`` is None iff there are no tasks
        # (the schema is the durable source of truth; values.csv is an intermediate removed by clean).
        self.has_tasks = False
        self.values_column = None
    schema = self.input_schema()
    with open(os.path.join(path, INPUT_SCHEMA_FILENAME), "w") as f:
      json.dump(schema, f, indent=4)
