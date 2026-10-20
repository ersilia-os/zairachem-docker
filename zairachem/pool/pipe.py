import os

from zairachem.base import ZairaBase
from zairachem.base.utils.logging import logger
from zairachem.base.utils.matrices import DEFAULT_CHUNK_SIZE
from zairachem.base.utils.pipeline import PipelineStep
from zairachem.pool.reliability_pooler.pipe import ReliabilityPoolerPipeline


class PoolerPipeline(ZairaBase):
  """Combine the per-descriptor predictions into the final pooled prediction.

  Parameters
  ----------
  path : str
    Run folder.
  batch_size : int, optional
    Rows per chunk when reading the per-descriptor results.
  """

  def __init__(self, path, batch_size=None):
    ZairaBase.__init__(self)
    self.path = path
    self.output_dir = os.path.abspath(self.path)
    self.batch_size = batch_size or DEFAULT_CHUNK_SIZE
    assert os.path.exists(self.output_dir)
    self.params = self._load_params()
    self.descriptors = self.get_descriptors()

  def get_descriptors(self):
    """Return the descriptors the pooled model uses (the screened selection when there is one)."""
    self.logger.debug("Getting individual descriptors")
    from zairachem.base.utils.descriptors import effective_descriptors

    return effective_descriptors(self.path)

  def _pool_pipeline(self):
    step = PipelineStep("pool", self.output_dir)
    if not step.is_done():
      # The legacy "bagger" pooler was removed; a model whose parameters still ask for it must be
      # refitted rather than silently pooled differently from how it was trained.
      if (self.params or {}).get("pooler", "reliability") == "bagger":
        raise RuntimeError(
          "This model was trained with the legacy 'bagger' pooler, which has been removed. "
          "Refit it with the current ZairaChem."
        )
      logger.info(f"[pool] Running 'reliability' pipeline with batch_size={self.batch_size}")
      ReliabilityPoolerPipeline(self.path, batch_size=self.batch_size).run()
    else:
      self.logger.info("Pooling already done — skipping.")

  def run(self):
    """Pool the per-descriptor predictions, skipping the step if already done."""
    self._pool_pipeline()
