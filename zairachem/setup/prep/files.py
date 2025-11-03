import collections, json, os
import pandas as pd
import numpy as np
from rdkit import Chem
from zairachem.setup.prep.schema import InputSchema
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


class ModelIdsFile(object):
  def __init__(self, path):
    self.path = os.path.abspath(path)
    self.data = None

  def load(self):
    if self.data is None:
      with open(self.path, "r") as f:
        self.data = json.load(f)
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
    all_smiles = list(set(list(self.df[self.df[self.smiles_column].notnull()][self.smiles_column])))
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

  def dedupe(self, df, path):
    mapping = collections.defaultdict(list)
    cid2smiles = {}
    for i, r in enumerate(df[[COMPOUND_IDENTIFIER_COLUMN, SMILES_COLUMN]].values):
      cid = r[0]
      smi = r[1]
      mol = Chem.MolFromSmiles(smi)
      if mol is None:
        continue
      mapping[cid] += [i]
      cid2smiles[cid] = smi
    unique_cids = sorted(set(mapping.keys()))
    unique_cids_idx = dict((k, i) for i, k in enumerate(unique_cids))
    mapping = dict((x, k) for k, v in mapping.items() for x in v)
    R = []
    for i in range(df.shape[0]):
      if i in mapping:
        cid = mapping[i]
        R += [[i, unique_cids_idx[cid], cid]]
      else:
        R += [[i, None, None]]
    dfm = pd.DataFrame(
      R,
      columns=[
        MAPPING_ORIGINAL_COLUMN,
        MAPPING_DEDUPE_COLUMN,
        COMPOUND_IDENTIFIER_COLUMN,
      ],
    )
    dfm.to_csv(os.path.join(path, MAPPING_FILENAME), index=False)
    R = []
    for cid in unique_cids:
      R += [[cid, cid2smiles[cid]]]
    dfc = pd.DataFrame(R, columns=[COMPOUND_IDENTIFIER_COLUMN, SMILES_COLUMN])
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
    R = []
    for k, v in dedupe.items():
      v = np.median([x for x in v])
      R += [[k,v]]
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
    resolved_columns = self.resolve_columns()
    self.smiles_column = resolved_columns["smiles_column"]
    identifiers = self._make_identifiers()
    df = pd.DataFrame({COMPOUND_IDENTIFIER_COLUMN: identifiers})
    self.values_column = resolved_columns["values_column"]
    if self.values_column is not None and self.params is not None:
      trained_values_column = self.get_trained_values_column()
      if self.values_column != trained_values_column:
        self.logger.warning(
          "Inconsistent values column, {0} vs {1}".format(self.values_column, trained_values_column)
        )
    df = pd.DataFrame({COMPOUND_IDENTIFIER_COLUMN: identifiers})
    df[SMILES_COLUMN] = self.df[self.smiles_column]
    assert df.shape[0] == self.df.shape[0]

    if self.values_column is not None and self.params is not None:
      df[VALUES_COLUMN] = self.df[self.values_column]
      self.has_tasks = True
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
      dfv = self.values_table(df)
      dfv.to_csv(os.path.join(path, VALUES_FILENAME), index=False)
    schema = self.input_schema()
    with open(os.path.join(path, INPUT_SCHEMA_FILENAME), "w") as f:
      json.dump(schema, f, indent=4)
