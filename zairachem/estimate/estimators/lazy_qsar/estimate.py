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
    shape = self._get_X_shape()
    if shape is None:
      logger.warning(f"[lazyqsar:fit] Skipping {self.model_id}: no descriptor data available")
      self.update_elapsed_time()
      return tasks
    train_idxs = self.get_train_indices(path=self.path)
    y = self._get_y()
    t = "reg" if self.task == "regression" else "clf"
    if self.task == "classification":
      train_set = set(train_idxs)
      logger.info(f"[lazyqsar:fit] Loading training subset: {len(train_idxs)} of {shape[0]} samples")
      X_train_parts = []
      for start, end, chunk in self._iter_X():
        mask = [i - start for i in train_idxs if start <= i < end]
        if mask:
          X_train_parts.append(chunk[mask])
        del chunk
        gc.collect()
      X_train = np.concatenate(X_train_parts, axis=0)
      del X_train_parts
      gc.collect()
      logger.info(f"[lazyqsar:fit] Training on {X_train.shape[0]} samples, {X_train.shape[1]} features")
      model = LazyBinaryClassifier()
      model.fit(X=X_train, y=y[train_idxs])
      del X_train
      gc.collect()
      model_folder = os.path.join(self.trained_path, self.model_id, t)
      model.save(model_folder, onnx=True)
      logger.info(f"[lazyqsar:fit] Model saved to {model_folder}")
      n_samples = shape[0]
      preds = np.empty(n_samples, dtype=np.float32)
      for start, end, chunk in self._iter_X():
        batch_preds = model.predict_proba(X=chunk)[:, 1]
        preds[start:end] = batch_preds
        del chunk
        gc.collect()
      tasks[t] = make_classification_report(y, preds)
    self.update_elapsed_time()
    gc.collect()
    return tasks


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
    shape = self._get_X_shape()
    if shape is None:
      logger.warning(f"[lazyqsar:predict] Skipping {self.model_id}: no descriptor data available")
      self.update_elapsed_time()
      return tasks
    y = self._get_y()
    t = "reg" if self.task == "regression" else "clf"
    if self.task == "classification":
      model_folder = os.path.join(self.trained_path, self.model_id, t)
      model = LazyBinaryClassifier.load(model_folder)
      logger.info(f"[lazyqsar:predict] Loaded model from {model_folder}, predicting {shape[0]} samples chunk-by-chunk")
      n_samples = shape[0]
      preds = np.empty(n_samples, dtype=np.float32)
      for start, end, chunk in self._iter_X():
        batch_preds = model.predict_proba(X=chunk)[:, 1]
        preds[start:end] = batch_preds
        logger.debug(f"[lazyqsar:predict] Processed {start}-{end}/{n_samples}")
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
      self.estimator = Predictor(
        path=self.path, model_id=self.model_id, batch_size=self.batch_size
      )

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
