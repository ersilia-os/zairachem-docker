from zairachem.base.utils.matrices import DEFAULT_CHUNK_SIZE

from .assemble import ReliabilityPoolerAssembler
from .pool import ReliabilityPooler


class ReliabilityPoolerPipeline:
  def __init__(self, path, batch_size=None):
    self.path = path
    self.batch_size = batch_size or DEFAULT_CHUNK_SIZE
    self.pooler = ReliabilityPooler(path, batch_size=self.batch_size)
    self.assembler = ReliabilityPoolerAssembler(path)

  def run(self):
    results = self.pooler.run()
    self.assembler.run(results)
