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
    self.data_size = self._get_data_size()  # TODO clean up if not needed

  def _get_data_size(self):
    data = pd.read_csv(os.path.join(self.get_trained_dir(), DATA_SUBFOLDER, DATA_FILENAME))
    return data.shape[0]

  def _load_params(self):
    with open(os.path.join(self.path, DATA_SUBFOLDER, PARAMETERS_FILE), "r") as f:
      params = json.load(f)
    return params


  def _lazyqsar_estimator_pipeline(self):
    step = PipelineStep("lazy-qsar", self.output_dir)
    if not step.is_done():
      self.logger.debug("Running lazyqsar pipeline")
      p = LazyQsarAutoMLPipeline(path=self.path)
      p.run()
      step.update()

  def _simple_evaluation(self):
    self.logger.debug("Simple evaluation is started")
    step = PipelineStep("simple_evaluation", self.output_dir)
    if not step.is_done():
      SimpleEvaluator(path=self.path).run()
      step.update()
    else:
      logger.warning(
        "[yellow]Estimation setup for requested inferece is already done. Skippign this step![/]"
      )

  def run(self):
    self._lazyqsar_estimator_pipeline()
    self._simple_evaluation()
