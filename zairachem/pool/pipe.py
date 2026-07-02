import json, os

from zairachem.base import ZairaBase
from zairachem.base.utils.pipeline import PipelineStep
from zairachem.base.utils.logging import logger
from zairachem.base.utils.matrices import DEFAULT_CHUNK_SIZE
from zairachem.pool.bagger.pipe import BaggerPipeline
from zairachem.pool.reliability_pooler.pipe import ReliabilityPoolerPipeline
from zairachem.base.vars import (
  DESCRIPTORS_SUBFOLDER,
)


class PoolerPipeline(ZairaBase):
  def __init__(self, path, batch_size=None):
    ZairaBase.__init__(self)
    if path is None:
      self.path = self.get_output_dir()
    else:
      self.path = path
    self.output_dir = os.path.abspath(self.path)
    self.batch_size = batch_size or DEFAULT_CHUNK_SIZE
    assert os.path.exists(self.output_dir)
    self.params = self._load_params()
    self.descriptors = self.get_descriptors()

  def get_descriptors(self):
    self.logger.debug("Getting individual descriptors")
    with open(os.path.join(self.path, DESCRIPTORS_SUBFOLDER, "done_eos.json"), "r") as f:
      model_ids = list(json.load(f))
    return model_ids

  def _pool_pipeline(self):
    step = PipelineStep("pool", self.output_dir)
    if not step.is_done():
      # Default to the per-sample reliability pooler; "bagger" restores the legacy global-weight
      # ensemble (the only option for regression). Selectable via parameters.json "pooler".
      pooler_name = (self.params or {}).get("pooler", "reliability")
      if self.params and str(self.params.get("task")) == "regression":
        pooler_name = "bagger"
      logger.info(f"[pool] Running '{pooler_name}' pipeline with batch_size={self.batch_size}")
      if pooler_name == "bagger":
        pipeline = BaggerPipeline(self.path, batch_size=self.batch_size)
      else:
        pipeline = ReliabilityPoolerPipeline(self.path, batch_size=self.batch_size)
      pipeline.run()
    else:
      self.logger.info("Pooling already done — skipping.")

  def run(self):
    self._pool_pipeline()
