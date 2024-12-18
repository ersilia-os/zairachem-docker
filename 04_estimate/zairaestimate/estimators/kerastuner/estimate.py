import os
import numpy as np
import pandas as pd
import h5py
import collections
import json
import joblib

from zairabase import ZairaBase

from zairabase.vars import (
    DESCRIPTORS_SUBFOLDER,
    ESTIMATORS_SUBFOLDER,
)

from ..base import BaseEstimatorIndividual
from .kerastuner_ import KerasTunerClassifier, KerasTunerRegressor
from . import ESTIMATORS_FAMILY_SUBFOLDER

TUNER_PROJECT_NAME = "kerastuner"

class Fitter(BaseEstimatorIndividual):
    def __init__(self, path, model_id):
        BaseEstimatorIndividual.__init__(self, path=path, estimator= ESTIMATORS_FAMILY_SUBFOLDER, model_id=model_id)
        self.trained_path = os.path.join(
            self.get_output_dir(), ESTIMATORS_SUBFOLDER, ESTIMATORS_FAMILY_SUBFOLDER
        )

    def run(self, time_budget_sec=None):
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
        save_path = os.path.join(self.trained_path, self.model_id,  TUNER_PROJECT_NAME)
        t = "reg" if self.task == "regression" else "clf"
        if self.task == "regression":
            model = KerasTunerRegressor()
            model.fit(
                X[train_idxs],
                y[train_idxs],
                save_path
            )
            model.save(save_path)
            model = model.load(save_path)
            tasks[t] = model.run(X, y)
            _valid_task = model.run(X[valid_idxs], y[valid_idxs])
            tasks[t]["valid"] = _valid_task["main"]
        if self.task == "classification":
            model = KerasTunerClassifier(X[train_idxs], y[train_idxs])
            model.fit(
                save_path
            )
            model.save(save_path)
            model = model.load(save_path)
            tasks[t] = model.run(X, y)
            _valid_task = model.run(X[valid_idxs], y[valid_idxs])
            tasks[t]["valid"] = _valid_task["main"]
        self.update_elapsed_time()
        return tasks

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
        t = self.task
        if self.task == "regression":
            y = self._get_y()
            model = KerasTunerRegressor()
            file_name = os.path.join(self.trained_path, self.model_id, t + ".joblib")
            model = model.load(file_name)
            tasks[t] = model.run(X, y)
        if self.task == "classification":
            y = self._get_y()
            model = KerasTunerClassifier()
            file_name = os.path.join(self.trained_path, self.model_id, t + ".joblib")
            model = model.load(file_name)
            tasks[t] = model.run(X, y)
        self.update_elapsed_time()
        return tasks


class IndividualEstimator(ZairaBase):
    def __init__(self, path=None, model_id=None):
        ZairaBase.__init__(self)
        self.model_id = model_id
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        if not self.is_predict():
            self.estimator = Fitter(
                path=self.path, model_id=self.model_id
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

