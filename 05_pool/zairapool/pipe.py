import os
import json
import importlib

from zairabase import ZairaBase
from zairabase.utils.pipeline import PipelineStep
from zairabase.vars import DATA_SUBFOLDER, PARAMETERS_FILE, DESCRIPTORS_SUBFOLDER
from zairabase.vars import ENSEMBLE_MODE


class PoolerPipeline(ZairaBase):
    def __init__(self, path):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        self.output_dir = os.path.abspath(self.path)
        assert os.path.exists(self.output_dir)
        self.params = self._load_params()
        self.estimators = self.get_estimators()
        self.descriptors = self.get_descriptors()

    def _load_params(self):
        with open(os.path.join(self.path, DATA_SUBFOLDER, PARAMETERS_FILE), "r") as f:
            params = json.load(f)
        return params

    def get_estimators(self):
        self.logger.debug("Getting estimators")
        self._estimators_used = set()
        for x in self.params["estimators"]:
            self._estimators_used.update([x])

    def get_descriptors(self):
        self.logger.debug("Getting descriptors")
        with open(
            os.path.join(self.path, DESCRIPTORS_SUBFOLDER, "done_eos.json"), "r"
        ) as f:
            model_ids = list(json.load(f))
        return model_ids
    
    def _pool_pipeline(self, time_budget_sec):
        step = PipelineStep("pool", self.output_dir)
        if not step.is_done():
            self.logger.debug("Pooling")
            if ENSEMBLE_MODE == "bagging":
                pooler = importlib.import_module(".pipe", package="zairapool.bagger")
                bagger = pooler.BaggerPipeline(self.path)
                bagger.run(time_budget_sec = time_budget_sec)
            elif ENSEMBLE_MODE == "blending":
                pooler = importlib.import_module(".blender.pipe")
            else:
                pooler = None
            step.update()

    def run(self, time_budget_sec=None):
        self._pool_pipeline(time_budget_sec)



