import os

from zairachem.base import ZairaBase
from zairachem.base.utils.pipeline import PipelineStep
from zairachem.base.utils.logging import logger
from zairachem.base.utils.matrices import DEFAULT_CHUNK_SIZE
from zairachem.estimate.estimators.evaluate import SimpleEvaluator
from zairachem.estimate.estimators.lazy_qsar.pipe import LazyQsarAutoMLPipeline


class EstimatorPipeline(ZairaBase):
  def __init__(self, path, batch_size=None):
    ZairaBase.__init__(self)
    self.logger = logger
    self.path = path
    self.output_dir = os.path.abspath(self.path)
    self.batch_size = batch_size or DEFAULT_CHUNK_SIZE
    assert os.path.exists(self.output_dir)
    self.params = self._load_params()

  def _rank_references(self):
    """Prepare lazy-qsar's rank reference for the descriptors this fit trains (after screening).

    Classification fits only: the regression estimator is a stub, and predict reads the reference
    from the trained model itself.
    """
    if self.is_predict() or self.params.get("task") != "classification":
      return
    from zairachem.base.utils.descriptors import effective_descriptors
    from zairachem.describe.descriptors import rank_reference

    rank_reference.prepare(self.path, effective_descriptors(self.path), batch_size=self.batch_size)

  def _lazyqsar_estimator_pipeline(self):
    step = PipelineStep("lazy-qsar", self.output_dir)
    if not step.is_done():
      self._rank_references()
      logger.info(f"[estimator] Running lazyqsar pipeline with batch_size={self.batch_size}")
      p = LazyQsarAutoMLPipeline(path=self.path, batch_size=self.batch_size)
      p.run()
      step.update()

  def _simple_evaluation(self):
    self.logger.debug("Simple evaluation is started")
    step = PipelineStep("simple_evaluation", self.output_dir)
    if not step.is_done():
      SimpleEvaluator(path=self.path).run()
      step.update()
    else:
      logger.info("Estimation already done — skipping.")

  def run(self):
    self._lazyqsar_estimator_pipeline()
    self._simple_evaluation()
