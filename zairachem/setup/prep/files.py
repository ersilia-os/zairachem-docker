import collections, json, os, re
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
from rdkit import Chem
from zairachem.base.utils.progress import SetupProgress


from zairachem.setup.prep.schema import InputSchema
from zairachem.base.utils.logging import logger
from zairachem.base.vars import (
  DATA_SUBFOLDER,
  COMPOUNDS_FILENAME,
  VALUES_FILENAME,
  MAPPING_FILENAME,
  INPUT_SCHEMA_FILENAME,
  MAPPING_ORIGINAL_COLUMN,
  MAPPING_DEDUPE_COLUMN,
  COMPOUND_IDENTIFIER_COLUMN,
  SMILES_COLUMN,
  VALUES_COLUMN,
)

DEDUPE_BATCH_SIZE = 5000
DEDUPE_MAX_WORKERS = None


def _validate_smiles_batch(batch_data):
  results = []
  for idx, cid, smi in batch_data:
    mol = Chem.MolFromSmiles(smi) if smi else None
    results.append((idx, cid, smi, mol is not None))
  return results


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
    if self.data is not None:
      return self.data
    if os.path.isfile(self.path):
      with open(self.path, "r") as f:
        self.data = json.load(f)
      return self.data
    # Not a file: parse the value as comma/space-separated featurizer model IDs.
    self.data = {"featurizer_ids": self.parse_ids(self.value, flag="--featurizer-ids")}
    return self.data

  def get_featurizer_ids(self):
    return self.load()["featurizer_ids"]

  def get_projection_ids(self):
    return self.load()["projection_ids"]


class ParametersFile(object):
  def __init__(self, path):
    self.path = os.path.abspath(path)
    self.params = None

  def load(self):
    if self.params is None:
      with open(self.path, "r") as f:
        self.params = json.load(f)
    return self.params


class SingleFile(InputSchema):
  def __init__(self, input_file, params):
    InputSchema.__init__(self, input_file)
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
    dfc = self.dedupe(df, path)
    return dfc

  def values_table(self, df):
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
    sc = {
      "input_file": self.input_file,
      "smiles_column": self.smiles_column,
      "values_column": self.values_column,
    }
    return sc

  def process(self):
    path = os.path.join(self.get_output_dir(), DATA_SUBFOLDER)
    df = self.normalize_dataframe()
    dfc = self.compounds_table(df, path)
    dfc.to_csv(os.path.join(path, COMPOUNDS_FILENAME), index=False)
    dfv = self.values_table(df)
    dfv.to_csv(os.path.join(path, VALUES_FILENAME), index=False)
    schema = self.input_schema()
    with open(os.path.join(path, INPUT_SCHEMA_FILENAME), "w") as f:
      json.dump(schema, f, indent=4)


class SingleFileForPrediction(SingleFile):
  def __init__(self, input_file, params):
    SingleFile.__init__(self, input_file, params)
    self.trained_path = self.get_trained_dir()

  def get_trained_values_column(self):
    with open(os.path.join(self.trained_path, DATA_SUBFOLDER, INPUT_SCHEMA_FILENAME), "r") as f:
      return json.load(f)["values_column"]

  def normalize_dataframe(self):
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
    sc = {
      "input_file": self.input_file,
      "smiles_column": self.smiles_column,
      "values_column": self.values_column,
    }
    return sc

  def process(self):
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
      else:
        self.has_tasks = False
    schema = self.input_schema()
    with open(os.path.join(path, INPUT_SCHEMA_FILENAME), "w") as f:
      json.dump(schema, f, indent=4)
