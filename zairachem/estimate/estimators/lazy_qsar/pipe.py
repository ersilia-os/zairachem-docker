from zairachem.estimate.estimators.lazy_qsar.estimate import Estimator
from zairachem.estimate.estimators.lazy_qsar.assemble import OutcomeAssembler


class LazyQsarAutoMLPipeline(object):
  def __init__(self, path):
    self.e = Estimator(path=path)
    self.a = OutcomeAssembler(path=path)

  def run(self, time_budget_sec=None):
    self.e.run(time_budget_sec=time_budget_sec)
    self.a.run()
