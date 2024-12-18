import os
import numpy as np
import pandas as pd

from zairabase import ZairaBase
from ..base import BaseEstimator, BaseOutcomeAssembler
from ..automl.classic import ClassicEstimator

from . import ESTIMATORS_FAMILY_SUBFOLDER
from zairabase.vars import (
    DATA_SUBFOLDER,
    ESTIMATORS_SUBFOLDER,
    DATA_FILENAME,
    RESULTS_UNMAPPED_FILENAME, 
    RESULTS_MAPPED_FILENAME,
    SMILES_COLUMN
)

class Fitter(BaseEstimator):
    def __init__(self, path):
        BaseEstimator.__init__(self, path=path)
        self.trained_path = os.path.join(
            self.get_output_dir(), ESTIMATORS_SUBFOLDER, ESTIMATORS_FAMILY_SUBFOLDER
        )

    def _get_smiles(self):
        df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
        return df[[SMILES_COLUMN]]

    def _get_y(self): 
        df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
        Y_col = self.get_Y_col()
        return df[[Y_col]], Y_col

    def run(self, time_budget_sec=None):
        self.reset_time()
        if time_budget_sec is None:
            time_budget_sec = self._estimate_time_budget()
        else:
            time_budget_sec = time_budget_sec
        train_idxs = self.get_train_indices(path=self.path)
        df_smiles = self._get_smiles()
        df_Y, Y_col = self._get_y()
        df = pd.concat([df_smiles, df_Y], axis=1)
        self.logger.debug("Starting Classic estimation")
        estimator = ClassicEstimator(save_path=self.trained_path, task=self.task, Y_col=Y_col)
        self.logger.debug("Fitting")
        estimator.fit(data=df.iloc[train_idxs, :])
        estimator.save()
        estimator = estimator.load()
        df_smiles = self._get_smiles()
        df_Y, _ = self._get_y()
        df = pd.concat([df_smiles, df_Y], axis=1)
        results = estimator.run(df)
        self.update_elapsed_time()
        return results


class Predictor(BaseEstimator):
    def __init__(self, path):
        BaseEstimator.__init__(self, path=path)
        self.trained_path = os.path.join(
            self.get_trained_dir(), ESTIMATORS_SUBFOLDER, ESTIMATORS_FAMILY_SUBFOLDER
        )

    def run(self):
        self.reset_time()
        df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))[
            [SMILES_COLUMN]
        ]
        model = ClassicEstimator(save_path=self.trained_path).load()
        results = model.run(df)
        self.update_elapsed_time()
        return results


class Assembler(BaseOutcomeAssembler):
    def __init__(self, path=None):
        BaseOutcomeAssembler.__init__(self, path=path)

    def run(self, df):
        df_c = self._get_compounds()
        df_y = df
        df = pd.concat([df_c, df_y], axis=1)
        df.to_csv(
            os.path.join(
                self.path,
                ESTIMATORS_SUBFOLDER,
                ESTIMATORS_FAMILY_SUBFOLDER,
                RESULTS_UNMAPPED_FILENAME,
            ),
            index=False,
        )
        mappings = self._get_mappings()
        df = self._remap(df, mappings)
        df.to_csv(
            os.path.join(
                self.path,
                ESTIMATORS_SUBFOLDER,
                ESTIMATORS_FAMILY_SUBFOLDER,
                RESULTS_MAPPED_FILENAME,
            ),
            index=False,
        )


class Estimator(ZairaBase):
    def __init__(self, path=None):
        ZairaBase.__init__(self)
        self.logger.debug(path)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        path_ = os.path.join(
            self.path, ESTIMATORS_SUBFOLDER, ESTIMATORS_FAMILY_SUBFOLDER
        )
        if not os.path.exists(path_):
            os.makedirs(path_, exist_ok=True)
        if not self.is_predict():
            self.logger.debug("Starting Classic fitter")
            self.estimator = Fitter(path=self.path)
        else:
            self.logger.debug("Starting Classic predictor")
            self.estimator = Predictor(path=self.path)
        self.assembler = Assembler(path=self.path)

    def run(self, time_budget_sec=None):
        if time_budget_sec is not None:
            self.time_budget_sec = int(time_budget_sec)
        else:
            self.time_budget_sec = None
        if not self.is_predict():
            self.logger.debug("Mode: fit")
            results = self.estimator.run()
        else:
            self.logger.debug("Mode: predict")
            results = self.estimator.run()
        self.assembler.run(results)
