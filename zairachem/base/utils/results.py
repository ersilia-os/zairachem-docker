"""Iterate a run's per-descriptor estimator result folders.

Shared by the estimate, pool and report steps (previously copy-pasted in each). Yields the
``[estimator_family, descriptor_id]`` relative paths under ``estimators/`` for the descriptors that
were actually produced (listed in ``descriptors/done_eos.json``).
"""

import os

from zairachem.base import ZairaBase
from zairachem.base.vars import ESTIMATORS_SUBFOLDER


class ResultsIterator(ZairaBase):
  def __init__(self, path):
    ZairaBase.__init__(self)
    if path is None:
      self.path = self.get_output_dir()
    else:
      self.path = path

  def _read_model_ids(self):
    # Honour --max-descriptors pre-screening: use the selected subset when present, else every
    # featurizer in done_eos.json.
    from zairachem.base.utils.descriptors import effective_descriptors

    return effective_descriptors(self.path)

  def iter_relpaths(self):
    estimators_folder = os.path.join(self.path, ESTIMATORS_SUBFOLDER)
    model_ids = self._read_model_ids()
    rpaths = []
    for est_fam in os.listdir(estimators_folder):
      if os.path.isdir(os.path.join(estimators_folder, est_fam)):
        focus_folder = os.path.join(estimators_folder, est_fam)
        for d in os.listdir(focus_folder):
          if d in model_ids:
            rpaths += [[est_fam, d]]
    for rpath in rpaths:
      yield rpath

  def iter_abspaths(self):
    for rpath in self.iter_relpaths():
      yield "/".join([self.path] + rpath)
