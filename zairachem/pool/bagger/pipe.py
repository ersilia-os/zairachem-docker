from .pool import Bagger
from .assemble import BaggerAssembler


class BaggerPipeline:
  def __init__(self, path):
    self.path = path
    self.bagger = Bagger(path)
    self.assembler = BaggerAssembler(path)

  def run(self, time_budget_sec=None):
    results = self.bagger.run(time_budget_sec=time_budget_sec)
    self.assembler.run(results)
