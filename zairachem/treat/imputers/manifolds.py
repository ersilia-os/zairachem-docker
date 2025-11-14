import csv, joblib, json, os
import numpy as np
import pandas as pd
from zairachem.describe.descriptors.utils import get_model_url
from zairachem.treat.imputers import DescriptorBase
from zairachem.base.utils.matrices import Hdf5, Data
from zairachem.base.utils.utils import fetch_schema_from_github, post
from zairachem.base.vars import (
  DATA_FILENAME,
  DATA_SUBFOLDER,
  DESCRIPTORS_SUBFOLDER,
  PARAMETERS_FILE,
  ERSILIA_DATA_FILENAME,
  DEFAULT_PROJECTIONS,
)

MAX_COMPONENTS = 4


class Manifolds(DescriptorBase):
  def __init__(self):
    DescriptorBase.__init__(self)
    self.path = self.get_output_dir()
    self.input_file = os.path.join(self.path, DATA_SUBFOLDER, ERSILIA_DATA_FILENAME)
    self.url = get_model_url(DEFAULT_PROJECTIONS[0])
    params_file = os.path.join(self.trained_path, "data", PARAMETERS_FILE)
    with open(params_file, "r") as f:
      self.reference_eos_id = json.load(f)["featurizer_ids"][0]

  def _load_data(self):
    with open(self.input_file, "r") as f:
      reader = csv.reader(f)
      h = next(reader)
      return ([row[0] for row in reader], h)

  def load(self, file_name):
    return joblib.load(file_name)

  def save(self, obj, file_name):
    joblib.dump(obj, file_name)

  def finalize(self, algo_name):
    algo_path = os.path.join(self.trained_path, algo_name + ".joblib")

    if not self._is_predict:
      self.save(self.X, algo_path)
    else:
      self.X = self.load(algo_path)

    file_name = os.path.join(self.path, DESCRIPTORS_SUBFOLDER, algo_name + ".h5")
    data = Data()
    data.set(inputs=self.inputs, values=self.X, features=None)
    Hdf5(file_name).save(data)
    data.save_info(file_name.split(".")[0] + ".json")

  def group_by_prefix(self, names):
    groups = {}
    for name in names:
      prefix, *_ = name.split("_", 1)
      groups.setdefault(prefix, []).append(name)
    return groups

  def run(self):
    self.inputs = self._load_data()[0]
    data = post(self.inputs, self.url)
    cols = fetch_schema_from_github(DEFAULT_PROJECTIONS[0])[0]
    algos = self.group_by_prefix(cols)
    if not self._is_predict:
      self.train_idxs = self.get_train_indices(path=self.path)
      self.trained_path = os.path.join(self.path, DESCRIPTORS_SUBFOLDER)
    else:
      self.trained_path = os.path.join(self.trained_path, DESCRIPTORS_SUBFOLDER)
    for algo, dims in algos.items():
      self.X = [[d[dim] for dim in dims] for d in data]
      self.finalize(algo)
