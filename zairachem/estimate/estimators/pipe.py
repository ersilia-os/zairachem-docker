import json, os
import pandas as pd

from zairachem.base import ZairaBase
from zairachem.base.utils.pipeline import PipelineStep
from zairachem.base.utils.logging import logger
from zairachem.base.vars import DATA_SUBFOLDER, DATA_FILENAME, PARAMETERS_FILE
from zairachem.estimate.estimators.evaluate import SimpleEvaluator
from zairachem.estimate.estimators.lazy_qsar.pipe import LazyQsarAutoMLPipeline

MOLMAP_DATA_SIZE_LIMIT = 10000


class EstimatorPipeline(ZairaBase):
  def __init__(self, path):
    ZairaBase.__init__(self)
    self.logger = logger
    if path is None:
      self.path = self.get_output_dir()
    else:
      self.path = path
    self.output_dir = os.path.abspath(self.path)
    assert os.path.exists(self.output_dir)
    self.params = self._load_params()
    self.get_estimators()
    self.data_size = self._get_data_size()  # TODO clean up if not needed

  def _get_data_size(self):
    data = pd.read_csv(os.path.join(self.get_trained_dir(), DATA_SUBFOLDER, DATA_FILENAME))
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

  def _lq_random_forest_estimator_pipeline(self, time_budget_sec):
    self.logger.info(f"Selected estimators to use:{self._estimators_to_use}")
    if "lazyqsar" not in self._estimators_to_use:
      return
    step = PipelineStep("random_forest", self.output_dir)
    if not step.is_done():
      self.logger.debug("Running lazyqsar pipeline for the selected estimators!")
      p = LazyQsarAutoMLPipeline(path=self.path)
      p.run(time_budget_sec=time_budget_sec)
      step.update()

  def _simple_evaluation(self):
    self.logger.debug("Simple evaluation is started")
    step = PipelineStep("simple_evaluation", self.output_dir)
    if not step.is_done():
      SimpleEvaluator(path=self.path).run()
      step.update()

  def run(self, time_budget_sec=None):
    self._lq_random_forest_estimator_pipeline(time_budget_sec)
    self._simple_evaluation()
