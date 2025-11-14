from .pool import Bagger
from .assemble import BaggerAssembler


class BaggerPipeline:
  def __init__(self, path):
    self.path = path
    self.bagger = Bagger(path)
    self.assembler = BaggerAssembler(path)

  def run(self):
    results = self.bagger.run()
    self.assembler.run(results)
