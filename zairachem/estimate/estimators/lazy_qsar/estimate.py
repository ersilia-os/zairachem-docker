import collections, joblib, json, lazyqsar, os
from zairachem.estimate.estimators.lazy_qsar.utils import make_classification_report
from zairachem.base import ZairaBase
from zairachem.base.vars import (
  DESCRIPTORS_SUBFOLDER,
  ESTIMATORS_SUBFOLDER,
  Y_HAT_FILE,
)
from zairachem.estimate.estimators.lazy_qsar import ESTIMATORS_FAMILY_SUBFOLDER
from zairachem.estimate.estimators.base import BaseEstimatorIndividual


ESTIMATORS = ["random_forest"]  # TODO CONFIRM THESE ONLY


class Fitter(BaseEstimatorIndividual):
  def __init__(self, path, model_id, is_simple):
    BaseEstimatorIndividual.__init__(
      self, path=path, estimator=ESTIMATORS_FAMILY_SUBFOLDER, model_id=model_id
    )
    self.trained_path = os.path.join(
      self.get_output_dir(), ESTIMATORS_SUBFOLDER, ESTIMATORS_FAMILY_SUBFOLDER
    )
    self.is_simple = is_simple

  def run_simple(self, time_budget_sec):
    self.reset_time()
    if time_budget_sec is None:
      time_budget_sec = self._estimate_time_budget()
    else:
      time_budget_sec = time_budget_sec
    tasks = collections.OrderedDict()
    X = self._get_X()
    train_idxs = self.get_train_indices(path=self.path)
    valid_idxs = self.get_validation_indices(path=self.path)
    y = self._get_y()
    t = "reg" if self.task == "regression" else "clf"
    if self.task == "classification":
      model = lazyqsar.LazyBinaryClassifier()
      model.fit(
        X[train_idxs],
        y[train_idxs],
      )
      model_folder = os.path.join(self.trained_path, self.model_id, t)
      model.save_model(model_folder)
      model = model.load_model(model_folder)
      train_preds = model.predict_proba(X)
      tasks[t] = make_classification_report(y, train_preds)
      valid_preds = model.predict_proba(X[valid_idxs])
      tasks[t]["valid"] = make_classification_report(y[valid_idxs], valid_preds)["main"]

    self.update_elapsed_time()
    return tasks

  def run(self, time_budget_sec=None):
    return self.run_simple(time_budget_sec=time_budget_sec)


class Predictor(BaseEstimatorIndividual):
  def __init__(self, path, model_id):
    BaseEstimatorIndividual.__init__(
      self, path=path, estimator=ESTIMATORS_FAMILY_SUBFOLDER, model_id=model_id
    )
    self.trained_path = os.path.join(
      self.get_trained_dir(), ESTIMATORS_SUBFOLDER, ESTIMATORS_FAMILY_SUBFOLDER
    )

  def run(self):
    self.reset_time()
    tasks = collections.OrderedDict()
    X = self._get_X()
    t = "reg" if self.task == "regression" else "clf"
    if self.task == "classification":
      model = lazyqsar.LazyBinaryClassifier()
      model_folder = os.path.join(self.trained_path, self.model_id, t)
      model = model.load_model(model_folder)
      tasks[t] = model.predict_proba(X)
    self.update_elapsed_time()
    return tasks


class IndividualEstimator(ZairaBase):
  def __init__(self, path=None, model_id=None, is_simple=True):
    ZairaBase.__init__(self)
    self.model_id = model_id
    if path is None:
      self.path = self.get_output_dir()
    else:
      self.path = path
    if not self.is_predict():
      self.estimator = Fitter(path=self.path, model_id=self.model_id, is_simple=is_simple)
    else:
      self.estimator = Predictor(path=self.path, model_id=self.model_id)

  def run(self, time_budget_sec=None):
    if time_budget_sec is not None:
      self.time_budget_sec = int(time_budget_sec)
    else:
      self.time_budget_sec = None
    if not self.is_predict():
      results = self.estimator.run(time_budget_sec=self.time_budget_sec)
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
  def __init__(self, path=None):
    ZairaBase.__init__(self)
    self.path = path

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

  def run(self, time_budget_sec=None):
    model_ids = self._get_model_ids()
    if time_budget_sec is not None:
      tbs = max(int(time_budget_sec / len(model_ids)), 1)
    else:
      tbs = None
    for model_id in model_ids:
      estimator = IndividualEstimator(path=self.path, model_id=model_id)
      estimator.run(time_budget_sec=tbs)
