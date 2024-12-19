import os
import pandas as pd
import json
import numpy as np
import h5py
# Internal imports

from zairabase import ZairaBase
from zairabase.vars import (
    COMPOUND_IDENTIFIER_COLUMN,
    PARAMETERS_FILE,
    SMILES_COLUMN,
    DATA_SUBFOLDER, 
    DATA_FILENAME,
    ESTIMATORS_SUBFOLDER,
    DESCRIPTORS_SUBFOLDER,
    POOL_SUBFOLDER,
    RESULTS_UNMAPPED_FILENAME,
    INPUT_SCHEMA_FILENAME,
    MAPPING_FILENAME,
)

class ResultsIterator(ZairaBase): #TODO SAME AS ESTIMATOR
    def __init__(self, path):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path

    def _read_model_ids(self):
        with open(
            os.path.join(self.path, DESCRIPTORS_SUBFOLDER, "done_eos.json"), "r"
        ) as f:
            model_ids = list(json.load(f))
        return model_ids

    def iter_relpaths(self):
        estimators_folder = os.path.join(self.path, ESTIMATORS_SUBFOLDER)
        model_ids = self._read_model_ids()
        rpaths = []
        for est_fam in os.listdir(estimators_folder):
            if os.path.isdir(os.path.join(estimators_folder, est_fam)):
                focus_folder = os.path.join(estimators_folder, est_fam)
                for d in os.listdir(focus_folder):
                    if d in model_ids:
                        rpaths += [[est_fam, d]]
        for rpath in rpaths:
            yield rpath

    def iter_abspaths(self):
        for rpath in self.iter_relpaths:
            yield "/".join([self.path] + rpath)

class XGetter(ZairaBase):
    def __init__(self, path):
        ZairaBase.__init__(self)
        self.path = path
        self.logger.debug(self.path)
        self.X = []
        self.columns = []

    @staticmethod
    def _read_results_file(file_path):
        df = pd.read_csv(file_path)
        df = df[
            [
                c
                for c in list(df.columns)
                if c not in [SMILES_COLUMN, COMPOUND_IDENTIFIER_COLUMN]
            ]
        ]
        return df

    def _get_manifolds(self):
        pca_file = os.path.join(self.path, DESCRIPTORS_SUBFOLDER, "pca.h5")
        if os.path.exists(pca_file):
            with h5py.File(
                os.path.join(self.path, DESCRIPTORS_SUBFOLDER, "pca.h5"), "r"
            ) as f:
                X_ = f["Values"][:]
                self.X += [X_]
                for i in range(X_.shape[1]):
                    self.columns += ["pca-{0}".format(i)]
        umap_file = os.path.join(self.path, DESCRIPTORS_SUBFOLDER, "umap.h5")
        if os.path.exists(umap_file):
            with h5py.File(
                os.path.join(self.path, DESCRIPTORS_SUBFOLDER, "umap.h5"), "r"
            ) as f:
                X_ = f["Values"][:]
                self.X += [X_]
                for i in range(X_.shape[1]):
                    self.columns += ["umap-{0}".format(i)]

    def _get_results(self):
        prefixes = []
        dfs = []
        for rpath in ResultsIterator(path=self.path).iter_relpaths():
            print(rpath)
            prefixes += ["-".join(rpath)]
            file_name = "/".join(
                [self.path, ESTIMATORS_SUBFOLDER] + rpath + [RESULTS_UNMAPPED_FILENAME]
            )
            print(file_name)
            dfs += [self._read_results_file(file_name)]
        for i in range(len(dfs)):
            df = dfs[i]
            prefix = prefixes[i]
            self.X += [np.array(df)]
            self.columns += ["{0}-{1}".format(prefix, c) for c in list(df.columns)]
        self.logger.debug(
            "Number of columns: {0} ... from {1} estimators".format(
                len(self.columns), len(dfs)
            )
        )

    def get(self):
        self._get_manifolds()
        self._get_results()
        X = np.hstack(self.X)
        df = pd.DataFrame(X, columns=self.columns)
        df.to_csv(os.path.join(self.path, POOL_SUBFOLDER, DATA_FILENAME), index=False)
        return df

class BasePooler(ZairaBase):
    def __init__(self, path):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        self.logger.debug(self.path)
        self.task = self._get_task()

    def _get_task(self):
        with open(os.path.join(self.path, DATA_SUBFOLDER, PARAMETERS_FILE), "r") as f:
            task = json.load(f)["task"]
        return task
    
    def _get_compound_ids(self):
        df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
        cids = list(df["compound_id"])
        return cids

    def _get_X(self):
        df = XGetter(path=self.path).get()
        return df

    def _get_X_clf(self, df):
        return df[[c for c in list(df.columns) if "clf" in c and "_bin" not in c]]

    def _get_X_reg(self, df):
        return df[[c for c in list(df.columns) if "reg" in c]]

    def _get_y(self, task):
        df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
        return np.array(df[task])

    def _get_Y_col(self):
        if self.task == "classification":
            Y_col = "bin"
        if self.task == "regression":
            Y_col = "val"
        return Y_col

    def _get_y(self): 
        df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
        Y_col = self._get_Y_col()
        return np.array(df[Y_col])
    
    def _filter_out_bin(self, df):
        columns = list(df.columns)
        columns = [c for c in columns if "_bin" not in c]
        return df[columns]

    def _filter_out_manifolds(self, df):
        columns = list(df.columns)
        columns = [c for c in columns if "umap-" not in c and "pca-" not in c]
        return df[columns]

    def _filter_out_unwanted_columns(self, df):
        df = self._filter_out_manifolds(df)
        df = self._filter_out_bin(df)
        return df

    def _get_total_time_budget_sec(self):
        with open(os.path.join(self.path, DATA_SUBFOLDER, PARAMETERS_FILE), "r") as f:
            time_budget = json.load(f)["time_budget"]
        return int(time_budget) * 60 + 1
    
    def _estimate_time_budget(self): #TODO CONFIRM TIME TO USE
        elapsed_time = self.get_elapsed_time()
        print("Elapsed time: {0}".format(elapsed_time))
        total_time_budget = self._get_total_time_budget_sec()
        print("Total time budget: {0}".format(total_time_budget))
        available_time = total_time_budget - elapsed_time
        # Assuming classification and regression will be done
        available_time = available_time / 2.0
        # Substract retraining and subsequent tasks
        available_time = available_time * 0.8
        available_time = int(available_time) + 1
        print("Available time: {0}".format(available_time))
        return available_time

class BaseOutcomeAssembler(ZairaBase):
    def __init__(self, path=None):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        if self.is_predict():
            self.trained_path = self.get_trained_dir()
        else:
            self.trained_path = self.path

    def _get_mappings(self):
        return pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, MAPPING_FILENAME))

    def _get_compounds(self):
        return pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))[
            [COMPOUND_IDENTIFIER_COLUMN, SMILES_COLUMN]
        ]

    def _get_original_input_size(self):
        with open(
            os.path.join(self.path, DATA_SUBFOLDER, INPUT_SCHEMA_FILENAME), "r"
        ) as f:
            schema = json.load(f)
        file_name = schema["input_file"]
        return pd.read_csv(file_name).shape[0]

    def _remap(self, df, mappings):
        n = self._get_original_input_size()
        ncol = df.shape[1]
        R = [[None] * ncol for _ in range(n)]
        for m in mappings.values:
            i, j = m[0], m[1]
            if np.isnan(j):
                continue
            R[i] = list(df.iloc[int(j)])
        return pd.DataFrame(R, columns=list(df.columns))