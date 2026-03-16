from zairachem.base.utils.matrices import DEFAULT_CHUNK_SIZE
from zairachem.estimate.estimators.lazy_qsar.estimate import Estimator
from zairachem.estimate.estimators.lazy_qsar.assemble import OutcomeAssembler


class LazyQsarAutoMLPipeline(object):
  def __init__(self, path, batch_size=None):
    self.batch_size = batch_size or DEFAULT_CHUNK_SIZE
    self.e = Estimator(path=path, batch_size=self.batch_size)
    self.a = OutcomeAssembler(path=path)

  def run(self):
    self.e.run()
    self.a.run()
