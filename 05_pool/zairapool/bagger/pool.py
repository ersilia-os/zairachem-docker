import os
import pandas as pd
import numpy as np

from ..base import BasePooler
from .bagger import BaggerRegressor, BaggerClassifier

from zairabase import ZairaBase
from zairabase.vars import POOL_SUBFOLDER

class Fitter(BasePooler):
    def __init__(self, path):
        BasePooler.__init__(self, path=path)
        self.trained_path = os.path.join(
            self.get_output_dir(), POOL_SUBFOLDER
        )

    def run(self, time_budget_sec=None):
        self.reset_time()
        if time_budget_sec is None:
            time_budget_sec = self._estimate_time_budget()
        else:
            time_budget_sec = time_budget_sec
        valid_idxs = self.get_validation_indices(path=self.path)
        cids = self._get_compound_ids()
        df_X = self._get_X()
        df_X = self._filter_out_unwanted_columns(df_X)
        df_Y = self._get_y()
        cids = [cids[idx] for idx in valid_idxs] # compound ids only for validation
        if self.task == "regression":
            df_X_reg = self._get_X_reg(df_X)
            X_reg = pd.DataFrame(df_X_reg).reset_index(drop=True)
            Y_reg = pd.DataFrame(df_Y).reset_index(drop=True)
            if X_reg.shape[1] > 0:
                reg = BaggerRegressor(path=self.trained_path)
                reg.fit(X_reg.iloc[valid_idxs], Y_reg.iloc[valid_idxs])
                Y_reg_hat = reg.predict(X_reg.iloc[valid_idxs]).reshape(-1, 1)
                results = pd.DataFrame({"reg": Y_reg_hat.flatten().tolist()})
            else:
                reg = None
        if self.task == "classification":
            df_X_clf = self._get_X_clf(df_X)
            X_clf = pd.DataFrame(df_X_clf).reset_index(drop=True)
            Y_clf = pd.DataFrame(df_Y).reset_index(drop=True)
            if X_clf.shape[1] > 0:
                clf = BaggerClassifier(path=self.trained_path)
                clf.fit(X_clf.iloc[valid_idxs], Y_clf.iloc[valid_idxs])
                Y_clf_hat = clf.predict(X_clf.iloc[valid_idxs]).reshape(-1, 1)
                B_clf_hat = np.zeros(Y_clf_hat.shape, dtype=int)
                B_clf_hat[Y_clf_hat > 0.5] = 1
                results = pd.DataFrame({"clf": Y_clf_hat.flatten().tolist(), "clf_bin": B_clf_hat.flatten().tolist()})
            else:
                clf = None
        columns = results.columns.tolist()
        results["compound_id"] = cids
        results = results[["compound_id"] + columns]
        self.update_elapsed_time()
        return results

class Predictor():
    def __init__(self, path):
        BasePooler.__init__(self, path=path)
        self.trained_path = os.path.join(
            self.get_trained_dir(), POOL_SUBFOLDER
        )

    def run(self):
        self.reset_time()
        df = self._get_X()
        df = self._filter_out_unwanted_columns(df)
        cids = self._get_compound_ids() # compound ids only for validation
        if self.task == "regression":
            df_X_reg = self._get_X_reg(df)
            X_reg = pd.DataFrame(df_X_reg).reset_index(drop=True)
            if X_reg.shape[1] > 0:
                reg = BaggerRegressor(path=self.trained_path)
                Y_reg_hat = reg.predict(X_reg).reshape(-1, 1)
                results = pd.DataFrame({"reg": Y_reg_hat})
            else:
                reg = None
        if self.task == "classification":
            df_X_clf = self._get_X_clf(df)
            X_clf = pd.DataFrame(df_X_clf).reset_index(drop=True)
            if X_clf.shape[1] > 0:
                clf = BaggerClassifier(path=self.trained_path)
                Y_clf_hat = clf.predict(X_clf).reshape(-1, 1)
                B_clf_hat = np.zeros(Y_clf_hat.shape, dtype=int)
                B_clf_hat[Y_clf_hat > 0.5] = 1
                results = pd.DataFrame({"clf": Y_clf_hat, "clf_bin": B_clf_hat})
            else:
                clf = None
        columns = results.columns.tolist()
        results["compound_id"] = cids
        results = results[["compound_id"] + columns]
        self.update_elapsed_time()
        return results

class Bagger(ZairaBase):
    def __init__(self, path=None):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        if not self.is_predict():
            self.estimator = Fitter(path=self.path)
        else:
            self.estimator = Predictor(path=self.path)

    def run(self, time_budget_sec=None):
        if time_budget_sec is not None:
            self.time_budget_sec = int(time_budget_sec)
        else:
            self.time_budget_sec = None
        if not self.is_predict():
            results = self.estimator.run(time_budget_sec=self.time_budget_sec)
        else:
            results = self.estimator.run()
        return results