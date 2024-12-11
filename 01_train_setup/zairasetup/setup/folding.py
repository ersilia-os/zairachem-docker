import os
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.cluster import KMeans

from . import STANDARD_COMPOUNDS_FILENAME, STANDARD_SMILES_COLUMN, FOLDS_FILENAME

class RandomFolds(object):
    def __init__(self, outdir):
        self.outdir = outdir
        self.input_file = self.get_input_file()

    def get_input_file(self):
        return os.path.join(self.outdir, STANDARD_COMPOUNDS_FILENAME)
    
    def random_k_fold_split(self, k=5, random_seed=42):
        df = pd.read_csv(self.input_file)
        random.seed(random_seed)
        shuffled_indices = np.random.permutation(len(df))
        folds = np.array_split(shuffled_indices,k)
        return folds
    
    def run(self):
        df = pd.read_csv(self.input_file)
        folds = self.random_k_fold_split()
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
    
    def _group_by_scaffold(self):
        df = pd.read_csv(self.input_file)
        smiles_list = df[STANDARD_SMILES_COLUMN]
        scaffold_dict = defaultdict(list)
        for idx, smiles in enumerate(smiles_list):
            scaffold = self._compute_scaffold(smiles)
            if scaffold:
                scaffold_dict[scaffold].append(idx)
        return scaffold_dict
    
    def scaffold_k_fold_split(self, k=5, random_seed=42):
        random.seed(random_seed)
        scaffold_dict = self._group_by_scaffold()
        scaffold_items =  sorted(scaffold_dict.items(), key=lambda x: len(x[1]), reverse=True)
        folds = [[] for _ in range(k)]
        fold_sizes = [0] * k
        for _, indices in scaffold_items:
            smallest_fold = min(range(k), key=lambda x: fold_sizes[x])
            folds[smallest_fold].extend(indices)
            fold_sizes[smallest_fold] += len(indices)
        return folds, fold_sizes
    
    def run(self):
        df = pd.read_csv(self.input_file)
        folds, fold_sizes= self.scaffold_k_fold_split()
        for fold_number, indices in enumerate(folds):
            df.loc[indices, "fld_scf"] = fold_number
        return df["fld_scf"].tolist()

class ClusterFolds(object):
    def __init__(self, outdir):
        self.outdir = outdir
        self.input_file = self.get_input_file()

    def get_input_file(self):
        return os.path.join(self.outdir, STANDARD_COMPOUNDS_FILENAME)

    def _compute_ecfp4_fingerprints(self):
        df = pd.read_csv(self.input_file)
        smiles_list=df[STANDARD_SMILES_COLUMN]
        fingerprints = []
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024)
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = mfpgen.GetFingerprint(mol)
                fingerprints.append(np.array(fp))
            else:
                fingerprints.append(None)
        return np.array([fp for fp in fingerprints if fp is not None])

    def _cluster_k_fold_split(self, k=5, random_seed=42):
        fingerprints = self._compute_ecfp4_fingerprints()
        kmeans = KMeans(n_clusters=k, random_state=random_seed)
        clusters = kmeans.fit_predict(fingerprints)
        return clusters

    def run(self):
        folds = self._cluster_k_fold_split()
        return folds


class FoldEnsemble(object):
    def __init__(self,outdir):
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