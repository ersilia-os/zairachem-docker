import joblib, os, random
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from umap import UMAP
from lol import LOL

# Bugfix to deal with large data in UMAP #TODO check
# from dill import extend
# extend(use_dill=False)


from zairachem.treat.imputers import DescriptorBase
from zairachem.base.utils.matrices import Hdf5, Data
from zairachem.base.vars import (
  DATA_FILENAME,
  DATA_SUBFOLDER,
  DESCRIPTORS_SUBFOLDER,
  RAW_DESC_FILENAME,
  PARAMETERS_FILE
)


MAX_COMPONENTS = 4


class Pca(object):
  def __init__(self):
    self._name = "pca"

  def fit(self, X, y=None):
    n_components = np.min([MAX_COMPONENTS, X.shape[0], X.shape[1]])
    self.pca = PCA(n_components=n_components, whiten=True)
    self.pca.fit(X)

  def transform(self, X):
    return self.pca.transform(X)

  def save(self, file_name):
    joblib.dump(self, file_name)

  def load(self, file_name):
    return joblib.load(file_name)


class Umap(object):
  def __init__(self, max_components=500, max_samples=1000):
    self._name = "umap"
    self._max_components = max_components
    self._max_samples = max_samples

  def _get_pca(self, X):
    pca = PCA(n_components=0.9, whiten=True)
    pca.fit(X)
    Xt = pca.transform(X)
    n_components = int(np.min([Xt.shape[1], self._max_components]))
    pca = PCA(n_components=n_components, whiten=True)
    return pca

  def fit(self, X, y=None):
    if X.shape[0] > self._max_samples:
      idxs = np.array(
        random.sample([i for i in range(X.shape[0])], self._max_samples),
        dtype=int,
      )
      X = X[idxs]
    self.pca = self._get_pca(X)
    self.pca.fit(X)
    Xt = self.pca.transform(X)
    self.reducer = UMAP(densmap=False)
    self.reducer.fit(Xt)

  def transform(self, X):
    try:
      Xt = self.pca.transform(X)
    except:
      Xt = X
    Xt = self.reducer.transform(Xt)
    return Xt

  def save(self, file_name):
    joblib.dump(self, file_name)

  def load(self, file_name):
    return joblib.load(file_name)


class LolliPop(object):
  def __init__(self):
    self._name = "lolp"

  def fit(self, X, y):
    n_components = np.min([MAX_COMPONENTS, X.shape[0], X.shape[1]])
    self.lmao = LOL(n_components=n_components, svd_solver="full")
    self.lmao.fit(X, y)

  def transform(self, X):
    return self.lmao.transform(X)

  def save(self, file_name):
    joblib.dump(self, file_name)

  def load(self, file_name):
    return joblib.load(file_name)


class Manifolds(DescriptorBase):
  def __init__(self):
    DescriptorBase.__init__(self)
    self.y = self._get_y_aux()
    params_file = os.path.join(self.trained_path, "data", PARAMETERS_FILE)
    with open(params_file, "r") as f:
      self.reference_eos_id = json.load(f)["featurizer_ids"][0]


  def _get_y_aux(self):  # TODO USE REAL BIN not AUX
    if not self.is_predict():
      y = np.array(pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))["bin"])
      return y
    else:
      return None

  def _algo(self, algo):
    algo_path = os.path.join(self.trained_path, algo._name + ".joblib")
    if not self._is_predict:
      algo.fit(X=self.X[self.train_idxs], y=self.y[self.train_idxs])
      algo.save(algo_path)
    else:
      algo = algo.load(algo_path)
    Xp = algo.transform(self.X)
    file_name = os.path.join(self.path, DESCRIPTORS_SUBFOLDER, algo._name + ".h5")
    data = Data()
    data.set(inputs=self.inputs, values=Xp, features=None)
    Hdf5(file_name).save(data)
    data.save_info(file_name.split(".")[0] + ".json")

  def run(self):
    print("REF: ", self.reference_eos_id)
    file_name = os.path.join(
      self.path, DESCRIPTORS_SUBFOLDER, self.reference_eos_id, RAW_DESC_FILENAME
    )
    data = Hdf5(file_name).load()
    self.inputs = data.inputs()
    self.X = data.values()
    if not self._is_predict:
      self.train_idxs = self.get_train_indices(path=self.path)
      self.trained_path = os.path.join(self.path, DESCRIPTORS_SUBFOLDER)
    else:
      self.trained_path = os.path.join(self.trained_path, DESCRIPTORS_SUBFOLDER)
    algo = Pca()
    self._algo(algo)
    algo = Umap()
    self._algo(algo)
    algo = LolliPop()
    self._algo(algo)
