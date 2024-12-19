import json
import os

from zairabase import ZairaBase
from zairabase.vars import DESCRIPTORS_SUBFOLDER, ESTIMATORS_SUBFOLDER

class ResultsIterator(ZairaBase): #TODO SAME AS ESTIMATOR
    def __init__(self, path):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path

    def _read_model_ids(self):
        with open(
            os.path.join(self.path, DESCRIPTORS_SUBFOLDER, "done_eos.json"), "r"
        ) as f:
            model_ids = list(json.load(f))
        return model_ids

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
        for rpath in self.iter_relpaths:
            yield "/".join([self.path] + rpath)