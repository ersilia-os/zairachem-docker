import json, os

from zairachem.base import ZairaBase
from zairachem.base.utils.pipeline import PipelineStep
from zairachem.pool.bagger.pipe import BaggerPipeline
from zairachem.base.vars import (
  DATA_SUBFOLDER,
  DESCRIPTORS_SUBFOLDER,
  PARAMETERS_FILE,
)


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
    self.descriptors = self.get_descriptors()

  def _load_params(self):
    with open(os.path.join(self.path, DATA_SUBFOLDER, PARAMETERS_FILE), "r") as f:
      params = json.load(f)
    return params

  def get_descriptors(self):
    self.logger.debug("Getting individual descriptors")
    with open(os.path.join(self.path, DESCRIPTORS_SUBFOLDER, "done_eos.json"), "r") as f:
      model_ids = list(json.load(f))
    return model_ids

  def _pool_pipeline(
    self,
  ):
    step = PipelineStep("pool", self.output_dir)
    if not step.is_done():
      bagger = BaggerPipeline(self.path)
      bagger.run()
    else:
      self.logger.warning(
        "[yellow]Pooling setup for requested inferece is already done. Skippign this step![/]"
      )

  def run(self):
    self._pool_pipeline()
