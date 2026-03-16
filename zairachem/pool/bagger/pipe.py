from zairachem.base.utils.matrices import DEFAULT_CHUNK_SIZE
from .pool import Bagger
from .assemble import BaggerAssembler


class BaggerPipeline:
  def __init__(self, path, batch_size=None):
    self.path = path
    self.batch_size = batch_size or DEFAULT_CHUNK_SIZE
    self.bagger = Bagger(path, batch_size=self.batch_size)
    self.assembler = BaggerAssembler(path)

  def run(self):
    results = self.bagger.run()
    self.assembler.run(results)
