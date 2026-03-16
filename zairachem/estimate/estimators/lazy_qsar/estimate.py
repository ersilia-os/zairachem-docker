import collections, joblib, json, os, gc
import numpy as np
from lazyqsar.agnostic import LazyBinaryClassifier
from zairachem.estimate.estimators.lazy_qsar.utils import make_classification_report
from zairachem.base import ZairaBase
from zairachem.base.utils.logging import logger
from zairachem.base.utils.matrices import DEFAULT_CHUNK_SIZE
from zairachem.base.vars import (
  DESCRIPTORS_SUBFOLDER,
  ESTIMATORS_SUBFOLDER,
  Y_HAT_FILE,
)
from zairachem.estimate.estimators.lazy_qsar import ESTIMATORS_FAMILY_SUBFOLDER
from zairachem.estimate.estimators.base import BaseEstimatorIndividual


class Fitter(BaseEstimatorIndividual):
  def __init__(self, path, model_id, is_simple, batch_size=None):
    BaseEstimatorIndividual.__init__(
      self,
      path=path,
      estimator=ESTIMATORS_FAMILY_SUBFOLDER,
      model_id=model_id,
      batch_size=batch_size,
    )
    self.trained_path = os.path.join(
      self.get_output_dir(), ESTIMATORS_SUBFOLDER, ESTIMATORS_FAMILY_SUBFOLDER
    )
    self.is_simple = is_simple

  def run(self):
    self.reset_time()
    tasks = collections.OrderedDict()
    X = self._get_X()
    if X is None:
      logger.warning(f"[lazyqsar:fit] Skipping {self.model_id}: no descriptor data available")
      self.update_elapsed_time()
      return tasks
    train_idxs = self.get_train_indices(path=self.path)
    y = self._get_y()
    t = "reg" if self.task == "regression" else "clf"
    if self.task == "classification":
      logger.info(f"[lazyqsar:fit] Training on {len(train_idxs)} samples, {X.shape[1]} features")
      model = LazyBinaryClassifier()
      model.fit(
        X=X[train_idxs],
        y=y[train_idxs],
      )
      model_folder = os.path.join(self.trained_path, self.model_id, t)
      model.save(model_folder, onnx=True)
      logger.info(f"[lazyqsar:fit] Model saved to {model_folder}")
      train_preds = self._predict_batched(model, X)
      tasks[t] = make_classification_report(y, train_preds)
    self.update_elapsed_time()
    del X
    gc.collect()
    return tasks

  def _predict_batched(self, model, X):
    n_samples = X.shape[0]
    if n_samples <= self.batch_size * 2:
      return model.predict_proba(X=X)[:, 1]
    logger.info(f"[lazyqsar:predict] Batched prediction on {n_samples} samples")
    preds = np.empty(n_samples, dtype=np.float32)
    for start in range(0, n_samples, self.batch_size):
      end = min(start + self.batch_size, n_samples)
      batch_preds = model.predict_proba(X=X[start:end])[:, 1]
      preds[start:end] = batch_preds
      logger.debug(f"[lazyqsar:predict] Processed {start}-{end}/{n_samples}")
    return preds


class Predictor(BaseEstimatorIndividual):
  def __init__(self, path, model_id, batch_size=None):
    BaseEstimatorIndividual.__init__(
      self,
      path=path,
      estimator=ESTIMATORS_FAMILY_SUBFOLDER,
      model_id=model_id,
      batch_size=batch_size,
    )
    self.trained_path = os.path.join(
      self.get_trained_dir(), ESTIMATORS_SUBFOLDER, ESTIMATORS_FAMILY_SUBFOLDER
    )

  def run(self):
    self.reset_time()
    tasks = collections.OrderedDict()
    X = self._get_X()
    if X is None:
      logger.warning(f"[lazyqsar:predict] Skipping {self.model_id}: no descriptor data available")
      self.update_elapsed_time()
      return tasks
    y = self._get_y()
    t = "reg" if self.task == "regression" else "clf"
    if self.task == "classification":
      model_folder = os.path.join(self.trained_path, self.model_id, t)
      model = LazyBinaryClassifier.load(model_folder)
      logger.info(f"[lazyqsar:predict] Loaded model from {model_folder}")
      preds = self._predict_batched(model, X)
      tasks[t] = make_classification_report(y, preds)
    self.update_elapsed_time()
    del X
    gc.collect()
    return tasks

  def _predict_batched(self, model, X):
    n_samples = X.shape[0]
    if n_samples <= self.batch_size * 2:
      return model.predict_proba(X=X)[:, 1]
    logger.info(f"[lazyqsar:predict] Batched prediction on {n_samples} samples")
    preds = np.empty(n_samples, dtype=np.float32)
    for start in range(0, n_samples, self.batch_size):
      end = min(start + self.batch_size, n_samples)
      batch_preds = model.predict_proba(X=X[start:end])[:, 1]
      preds[start:end] = batch_preds
      logger.debug(f"[lazyqsar:predict] Processed {start}-{end}/{n_samples}")
    return preds


class ChunkedPredictor(BaseEstimatorIndividual):
  def __init__(self, path, model_id, batch_size=None):
    BaseEstimatorIndividual.__init__(
      self,
      path=path,
      estimator=ESTIMATORS_FAMILY_SUBFOLDER,
      model_id=model_id,
      batch_size=batch_size,
    )
    self.trained_path = os.path.join(
      self.get_trained_dir(), ESTIMATORS_SUBFOLDER, ESTIMATORS_FAMILY_SUBFOLDER
    )

  def run(self):
    self.reset_time()
    tasks = collections.OrderedDict()
    shape = self._get_X_shape()
    if shape is None:
      logger.warning(
        f"[lazyqsar:chunked_predict] Skipping {self.model_id}: no descriptor data available"
      )
      self.update_elapsed_time()
      return tasks
    y = self._get_y()
    t = "reg" if self.task == "regression" else "clf"
    if self.task == "classification":
      model_folder = os.path.join(self.trained_path, self.model_id, t)
      model = LazyBinaryClassifier.load(model_folder)
      logger.info(f"[lazyqsar:chunked_predict] Loaded model from {model_folder}")
      n_samples = shape[0]
      preds = np.empty(n_samples, dtype=np.float32)
      for start, end, chunk in self._iter_X():
        batch_preds = model.predict_proba(X=chunk)[:, 1]
        preds[start:end] = batch_preds
        logger.debug(f"[lazyqsar:chunked_predict] Processed {start}-{end}/{n_samples}")
        del chunk
        gc.collect()
      tasks[t] = make_classification_report(y, preds)
    self.update_elapsed_time()
    return tasks


class IndividualEstimator(ZairaBase):
  def __init__(self, path=None, model_id=None, is_simple=True, batch_size=None):
    ZairaBase.__init__(self)
    self.model_id = model_id
    self.batch_size = batch_size or DEFAULT_CHUNK_SIZE
    if path is None:
      self.path = self.get_output_dir()
    else:
      self.path = path
    if not self.is_predict():
      self.estimator = Fitter(
        path=self.path, model_id=self.model_id, is_simple=is_simple, batch_size=self.batch_size
      )
    else:
      self.estimator = Predictor(path=self.path, model_id=self.model_id, batch_size=self.batch_size)

  def run(self):
    if not self.is_predict():
      results = self.estimator.run()
    else:
      results = self.estimator.run()
    joblib.dump(
      results,
      os.path.join(
        self.path,
        ESTIMATORS_SUBFOLDER,
        ESTIMATORS_FAMILY_SUBFOLDER,
        self.model_id,
        Y_HAT_FILE,
      ),
    )


class Estimator(ZairaBase):
  def __init__(self, path=None, batch_size=None):
    ZairaBase.__init__(self)
    self.path = path
    self.batch_size = batch_size or DEFAULT_CHUNK_SIZE

  def _get_model_ids(self):
    if self.path is None:
      path = self.get_output_dir()
    else:
      path = self.path
    if self.is_predict():
      path_trained = self.get_trained_dir()
    else:
      path_trained = path
    with open(os.path.join(path_trained, DESCRIPTORS_SUBFOLDER, "done_eos.json"), "r") as f:
      model_ids = list(json.load(f))
    return model_ids

  def run(self):
    model_ids = self._get_model_ids()
    for model_id in model_ids:
      logger.info(f"[lazyqsar] Processing model {model_id}")
      estimator = IndividualEstimator(path=self.path, model_id=model_id, batch_size=self.batch_size)
      estimator.run()
      gc.collect()
