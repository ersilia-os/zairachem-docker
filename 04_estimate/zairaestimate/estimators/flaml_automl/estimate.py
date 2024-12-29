import os
import json
import pandas as pd
import numpy as np
import collections
import joblib

from zairabase import ZairaBase
from zairabase.vars import (
    DESCRIPTORS_SUBFOLDER,
    DATA_SUBFOLDER,
    DATA_FILENAME,
    ESTIMATORS_SUBFOLDER,
    Y_HAT_FILE
)

from ..base import BaseEstimatorIndividual
from .flaml import FlamlClassifier, FlamlRegressor
from . import ESTIMATORS_FAMILY_SUBFOLDER


ESTIMATORS = ["rf", "extra_tree"]  # TODO CONFIRM THESE ONLY

class Fitter(BaseEstimatorIndividual):
    def __init__(self, path, model_id, is_simple):
        BaseEstimatorIndividual.__init__(self, path=path, estimator=ESTIMATORS_FAMILY_SUBFOLDER, model_id=model_id)
        self.trained_path = os.path.join(
            self.get_output_dir(), ESTIMATORS_SUBFOLDER, ESTIMATORS_FAMILY_SUBFOLDER
        )
        self.is_simple = is_simple

    def _get_flds(self): #TODO ARE WE USING FOLDS?
        # for now only auxiliary folds are used
        col = [f for f in self.schema["folds"] if "_aux" in f][0]
        df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
        return np.array(df[col])

    def run_heavy(self, time_budget_sec=None):
        self.reset_time()
        if time_budget_sec is None:
            time_budget_sec = self._estimate_time_budget()
        else:
            time_budget_sec = time_budget_sec
        groups = self._get_flds()
        tasks = collections.OrderedDict()
        X = self._get_X()
        y = self._get_y()
        t = "reg" if self.task == "regression" else "clf"
        if self.task == "regression":
            model = FlamlRegressor()
            model.fit_heavy(
                X,
                y,
                time_budget=time_budget_sec,
                estimators=ESTIMATORS,
                groups=groups,
                predict_by_group=False,
            )
            file_name = os.path.join(self.trained_path, self.model_id, t + ".joblib")
            model.save(file_name)
            tasks[t] = model.results
        if self.task == "classification":
            model = FlamlClassifier()
            model.fit_heavy(
                X,
                y,
                time_budget=time_budget_sec,
                estimators=ESTIMATORS,
                groups=groups,
                predict_by_group=False,
            )
            file_name = os.path.join(self.trained_path, self.model_id, t + ".joblib")
            model.save(file_name)
            tasks[t] = model.results
        self.update_elapsed_time()
        return tasks

    def run_simple(self, time_budget_sec):
        self.reset_time()
        if time_budget_sec is None:
            time_budget_sec = self._estimate_time_budget()
        else:
            time_budget_sec = time_budget_sec
        tasks = collections.OrderedDict()
        X = self._get_X()
        train_idxs = self.get_train_indices(path=self.path)
        valid_idxs = self.get_validation_indices(path=self.path)
        y = self._get_y()
        t = "reg" if self.task == "regression" else "clf"
        if self.task == "regression":
            model = FlamlRegressor()
            model.fit_simple(
                X[train_idxs],
                y[train_idxs],
                time_budget=time_budget_sec,
                estimators=ESTIMATORS,
            )
            file_name = os.path.join(self.trained_path, self.model_id, t + ".joblib")
            model.save(file_name)
            model = model.load(file_name)
            tasks[t] = model.run(X, y)
            _valid_task = model.run(X[valid_idxs], y[valid_idxs])
            tasks[t]["valid"] = _valid_task["main"]
        if self.task == "classification":
            model = FlamlClassifier()
            model.fit_simple(
                X[train_idxs],
                y[train_idxs],
                time_budget=time_budget_sec,
                estimators=ESTIMATORS,
            )
            file_name = os.path.join(self.trained_path, self.model_id, t + ".joblib")
            model.save(file_name)
            model = model.load(file_name)
            tasks[t] = model.run(X, y)
            _valid_task = model.run(X[valid_idxs], y[valid_idxs])
            tasks[t]["valid"] = _valid_task["main"]
        self.update_elapsed_time()
        return tasks

    def run(self, time_budget_sec=None):
        if self.is_simple:
            return self.run_simple(time_budget_sec=time_budget_sec)
        else:
            return self.run_heavy(time_budget_sec=time_budget_sec)


class Predictor(BaseEstimatorIndividual):
    def __init__(self, path, model_id):
        BaseEstimatorIndividual.__init__(self, path=path, estimator=ESTIMATORS_FAMILY_SUBFOLDER, model_id=model_id)
        self.trained_path = os.path.join(
            self.get_trained_dir(), ESTIMATORS_SUBFOLDER, ESTIMATORS_FAMILY_SUBFOLDER
        )

    def run(self):
        self.reset_time()
        tasks = collections.OrderedDict()
        X = self._get_X()
        t = "reg" if self.task == "regression" else "clf"
        if self.task == "regression":
            y = self._get_y()
            model = FlamlRegressor()
            file_name = os.path.join(self.trained_path, self.model_id, t + ".joblib")
            model = model.load(file_name)
            tasks[t] = model.run(X, y)
        if self.task == "classification":
            y = self._get_y()
            model = FlamlClassifier()
            file_name = os.path.join(self.trained_path, self.model_id, t + ".joblib")
            model = model.load(file_name)
            tasks[t] = model.run(X, y)
        self.update_elapsed_time()
        return tasks


class IndividualEstimator(ZairaBase):
    def __init__(self, path=None, model_id=None, is_simple=True):
        ZairaBase.__init__(self)
        self.model_id = model_id
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        if not self.is_predict():
            self.estimator = Fitter(
                path=self.path, model_id=self.model_id, is_simple=is_simple
            )
        else:
            self.estimator = Predictor(path=self.path, model_id=self.model_id)

    def run(self, time_budget_sec=None):
        if time_budget_sec is not None:
            self.time_budget_sec = int(time_budget_sec)
        else:
            self.time_budget_sec = None
        if not self.is_predict():
            results = self.estimator.run(time_budget_sec=self.time_budget_sec)
        else:
            results = self.estimator.run()
        joblib.dump(
            results,
            os.path.join(
                self.path,
                ESTIMATORS_SUBFOLDER,
                ESTIMATORS_FAMILY_SUBFOLDER,
                self.model_id,
                Y_HAT_FILE,
            ),
        )


class Estimator(ZairaBase):
    def __init__(self, path=None):
        ZairaBase.__init__(self)
        self.path = path

    def _get_model_ids(self):
        if self.path is None:
            path = self.get_output_dir()
        else:
            path = self.path
        if self.is_predict():
            path_trained = self.get_trained_dir()
        else:
            path_trained = path
        with open(
            os.path.join(path_trained, DESCRIPTORS_SUBFOLDER, "done_eos.json"), "r"
        ) as f:
            model_ids = list(json.load(f))
        #TODO do we need to check that the descriptors that must be treated should be treated?
        return model_ids

    def run(self, time_budget_sec=None):
        model_ids = self._get_model_ids()
        if time_budget_sec is not None:
            tbs = max(int(time_budget_sec / len(model_ids)), 1)
        else:
            tbs = None
        for model_id in model_ids:
            estimator = IndividualEstimator(path=self.path, model_id=model_id)
            estimator.run(time_budget_sec=tbs)
