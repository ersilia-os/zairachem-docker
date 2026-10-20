from zairachem.base.utils.matrices import DEFAULT_CHUNK_SIZE

from .assemble import BaggerAssembler
from .pool import Bagger


class BaggerPipeline:
  """Pool-step pipeline for the legacy bagger: pool the descriptors, then assemble results."""

  def __init__(self, path, batch_size=None):
    self.path = path
    self.batch_size = batch_size or DEFAULT_CHUNK_SIZE
    self.bagger = Bagger(path, batch_size=self.batch_size)
    self.assembler = BaggerAssembler(path)

  def run(self):
    """Pool the per-descriptor predictions with the bagger and write the results tables."""
    results = self.bagger.run()
    self.assembler.run(results)
