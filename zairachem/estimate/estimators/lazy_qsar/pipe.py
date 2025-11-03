from zairachem.estimate.estimators.lazy_qsar.estimate import Estimator
from zairachem.estimate.estimators.lazy_qsar.assemble import OutcomeAssembler


class LazyQsarAutoMLPipeline(object):
  def __init__(self, path):
    self.e = Estimator(path=path)
    self.a = OutcomeAssembler(path=path)

  def run(self):
    self.e.run()
    self.a.run()
