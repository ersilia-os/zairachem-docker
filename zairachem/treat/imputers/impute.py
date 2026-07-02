import os

from zairachem.base import ZairaBase
from zairachem.base.utils.logging import logger
from zairachem.base.utils.matrices import DEFAULT_CHUNK_SIZE
from zairachem.base.utils.pipeline import PipelineStep
from zairachem.treat.imputers.treated import TreatedDescriptors


class Imputer(ZairaBase):
  """Treat step: apply the reference transformers to the descriptors only.

  Projections (the 2-D embeddings shown in the report) are computed by the separate Projections
  step (``treat.imputers.manifolds.Manifolds``); they are shown as-is and are not transformed here.
  """

  def __init__(self, path, batch_size=None):
    ZairaBase.__init__(self)
    if path is None:
      self.path = self.get_output_dir()
    else:
      self.path = path
    self.output_dir = os.path.abspath(self.path)
    self.batch_size = batch_size or DEFAULT_CHUNK_SIZE
    assert os.path.exists(self.output_dir)

  def _treated_descriptions(self):
    step = PipelineStep("treated_descriptions", self.output_dir)
    if not step.is_done():
      logger.info(f"[imputer] Using chunk size: {self.batch_size}")
      TreatedDescriptors(chunk_size=self.batch_size).run()
      step.update()

  def run(self):
    self.reset_time()
    self._treated_descriptions()
    self.update_elapsed_time()
