import os
import json
import pandas as pd

from zairabase import ZairaBase
from zairabase.utils.pipeline import PipelineStep
from zairabase.vars import DATA_SUBFOLDER, DATA_FILENAME, PARAMETERS_FILE


from .flaml_automl.pipe import FlamlAutoMLPipeline
from .kerastuner.pipe import KerasTunerPipeline
#from .from_manifolds.pipe import ManifoldPipeline
from .evaluate import SimpleEvaluator

MOLMAP_DATA_SIZE_LIMIT = 10000

class EstimatorPipeline(ZairaBase):
    def __init__(self, path):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        self.output_dir = os.path.abspath(self.path)
        assert os.path.exists(self.output_dir)
        self.params = self._load_params()
        self.get_estimators()
        self.data_size = self._get_data_size() #TODO clean up if not needed

    def _get_data_size(self):
        data = pd.read_csv(
            os.path.join(self.get_trained_dir(), DATA_SUBFOLDER, DATA_FILENAME)
        )
        return data.shape[0]

    def _load_params(self):
        with open(os.path.join(self.path, DATA_SUBFOLDER, PARAMETERS_FILE), "r") as f:
            params = json.load(f)
        return params

    def get_estimators(self):
        self.logger.debug("Getting estimators")
        self._estimators_to_use = set()
        for x in self.params["estimators"]:
            self._estimators_to_use.update([x])

    def _flaml_estimator_pipeline(self, time_budget_sec):
        if "flaml" not in self._estimators_to_use:
            return
        step = PipelineStep("flaml", self.output_dir)
        if not step.is_done():
            self.logger.debug("Running flaml estimator pipeline")
            p = FlamlAutoMLPipeline(path=self.path)
            p.run(time_budget_sec=time_budget_sec)
            step.update()

    def _kerastuner_estimator_pipeline(self, time_budget_sec):
        if "kerastuner" not in self._estimators_to_use:
            return
        step = PipelineStep("kerastuner", self.output_dir)
        if not step.is_done():
            self.logger.debug("Running kerastuner estimator pipeline")
            p = KerasTunerPipeline(path=self.path)
            p.run(time_budget_sec=time_budget_sec)
            step.update()

    def _manifolds_pipeline(self, time_budget_sec):
        if "autogluon-manifolds" not in self._estimators_to_use:
            return
        step = PipelineStep("manifolds_pipeline", self.output_dir)
        if not step.is_done():
            self.logger.debug("Running manifolds estimator pipeline")
            p = ManifoldPipeline(path=self.path)
            p.run(time_budget_sec=time_budget_sec)
            step.update()

    def _simple_evaluation(self):
        self.logger.debug("Simple evaluation")
        step = PipelineStep("simple_evaluation", self.output_dir)
        if not step.is_done():
            SimpleEvaluator(path=self.path).run()
            step.update()

    def run(self, time_budget_sec=None):
        self._flaml_estimator_pipeline(time_budget_sec)
        self._kerastuner_estimator_pipeline(time_budget_sec)
        #self._manifolds_pipeline(time_budget_sec)
        self._simple_evaluation()
