import json
import os

from zairachem.base import ZairaBase
from zairachem.base.vars import DESCRIPTORS_SUBFOLDER


class DescriptorBase(ZairaBase):
  """Base class for the treat step, holding the run and trained model dirs and run mode."""

  def __init__(self, path):
    ZairaBase.__init__(self)
    self.path = path
    self.trained_path = self.get_trained_dir()
    self._is_predict = self.is_predict()
    self._is_train = self.is_train()

  def done_eos_iter(self):
    """Yield the eos_ids of the featurizers that the describe step completed (``done_eos.json``)."""
    with open(os.path.join(self.path, DESCRIPTORS_SUBFOLDER, "done_eos.json"), "r") as f:
      data = json.load(f)
    for eos_id in data:
      yield eos_id
