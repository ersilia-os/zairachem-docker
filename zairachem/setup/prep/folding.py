import os, random
import pandas as pd
import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.cluster import KMeans

from zairachem.setup.prep.utils import Fingerprinter

from zairachem.base.vars import (
  STANDARD_COMPOUNDS_FILENAME,
  STANDARD_SMILES_COLUMN,
  FOLDS_FILENAME,
)


class RandomFolds(object):
  def __init__(self, outdir):
    self.outdir = outdir
    self.input_file = self.get_input_file()

  def get_input_file(self):
    return os.path.join(self.outdir, STANDARD_COMPOUNDS_FILENAME)

  def random_k_fold_split(self, df, k=5, random_seed=42):
    random.seed(random_seed)
    shuffled_indices = np.random.permutation(len(df))
    folds = np.array_split(shuffled_indices, k)
    return folds

  def run(self):
    df = pd.read_csv(self.input_file)
    folds = self.random_k_fold_split(df)
    for fold_number, indices in enumerate(folds):
      df.loc[indices, "fld_rnd"] = fold_number
    return df["fld_rnd"].tolist()


class ScaffoldFolds(object):
  def __init__(self, outdir):
    self.outdir = outdir
    self.input_file = self.get_input_file()

  def get_input_file(self):
    return os.path.join(self.outdir, STANDARD_COMPOUNDS_FILENAME)

  def _compute_scaffold(self, smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
      return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, isomericSmiles=include_chirality)

  def _group_by_scaffold(self, smiles_list):
    scaffold_dict = defaultdict(list)
    for idx, smiles in enumerate(smiles_list):
      scaffold = self._compute_scaffold(smiles)
      if scaffold:
        scaffold_dict[scaffold].append(idx)
    return scaffold_dict

  def scaffold_k_fold_split(self, smiles_list, k=5, random_seed=42):
    random.seed(random_seed)
    scaffold_dict = self._group_by_scaffold(smiles_list)
    scaffold_items = sorted(scaffold_dict.items(), key=lambda x: len(x[1]), reverse=True)
    folds = [[] for _ in range(k)]
    fold_sizes = [0] * k
    for _, indices in scaffold_items:
      smallest_fold = min(range(k), key=lambda x: fold_sizes[x])
      folds[smallest_fold].extend(indices)
      fold_sizes[smallest_fold] += len(indices)
    return folds, fold_sizes

  def run(self):
    df = pd.read_csv(self.input_file)
    smiles_list = df[STANDARD_SMILES_COLUMN]
    folds, fold_sizes = self.scaffold_k_fold_split(smiles_list)
    for fold_number, indices in enumerate(folds):
      df.loc[indices, "fld_scf"] = fold_number
    return df["fld_scf"].tolist()


class ClusterFolds(object):
  def __init__(self, outdir):
    self.outdir = outdir
    self.input_file = self.get_input_file()

  def get_input_file(self):
    return os.path.join(self.outdir, STANDARD_COMPOUNDS_FILENAME)

  def cluster_k_fold_split(self, smiles_list, k=5, random_seed=42):
    fp_generator = Fingerprinter()
    fingerprints = fp_generator.calculate(smiles_list)
    kmeans = KMeans(n_clusters=k, random_state=random_seed)
    clusters = kmeans.fit_predict(fingerprints)
    return clusters

  def run(self):
    df = pd.read_csv(self.input_file)
    smiles_list = df[STANDARD_SMILES_COLUMN]
    folds = self.cluster_k_fold_split(smiles_list)
    return folds


class FoldEnsemble(object):
  def __init__(self, outdir):
    self.outdir = outdir
    self.input_file = self.get_input_file()
    self.output_file = self.get_output_file()

  def get_output_file(self):
    return os.path.join(self.outdir, FOLDS_FILENAME)

  def get_input_file(self):
    return os.path.join(self.outdir, STANDARD_COMPOUNDS_FILENAME)

  def run(self):
    df = pd.read_csv(self.input_file)
    fld_scf = ScaffoldFolds(self.outdir).run()
    fld_clt = ClusterFolds(self.outdir).run()
    fld_rnd = RandomFolds(self.outdir).run()
    df = df.assign(fld_rnd=fld_rnd, fld_scf=fld_scf, fld_clt=fld_clt)
    df.to_csv(self.get_output_file(), index=False)
