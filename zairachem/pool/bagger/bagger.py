import os
import numpy as np
import joblib
import collections
from sklearn.linear_model import LogisticRegressionCV, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.metrics import roc_curve, auc, r2_score


class WeightSchemes(object):
  def __init__(self, df_X, df_y, task_type):
    self.df_X = df_X
    self.df_y = df_y
    self.task_type = task_type
    if task_type == "classification":
      self.is_clf = True
      self.score = "roc_auc"
    else:
      self.is_clf = False
      self.score = "r2"
    self.columns = list(self.df_X.columns)
    self._weights = collections.defaultdict(dict)

  def distance_to_leads(self):
    performances = {}
    y = np.array(self.df_y).ravel()
    for col in self.columns:
      y_hat = list(self.df_X[col])
      if self.score == "roc_auc":
        fpr, tpr, _ = roc_curve(y, y_hat)
        performances[col] = auc(fpr, tpr)
      else:
        performances[col] = r2_score(y, y_hat)
    perfs = [v for k, v in performances.items()]
    ref = np.percentile(perfs, 66)
    weights = {}
    for c in self.columns:
      weights[c] = 1 - np.abs(ref - performances[c])
    self._weights["distance_to_leads"] = weights

  def importance(self):
    if self.is_clf:
      mdl = RandomForestClassifier()
    else:
      mdl = RandomForestRegressor()
    mdl.fit(np.array(self.df_X), np.array(self.df_y).ravel())
    weights = {}
    for imp, c in zip(mdl.feature_importances_, self.columns):
      weights[c] = imp
    self._weights["importance"] = weights

  def save(self, filename):
    d = {}
    for k, v in self._weights.items():
      d[k] = v
    results = {"columns": self.columns, "weights": d}
    joblib.dump(results, filename)


class BaggerClassifier(object):
  def __init__(self, path, mode="weighting"):
    assert mode in ["weighting", "median", "model"]
    self.path = path
    if not os.path.exists(self.path):
      os.makedirs(self.path, exist_ok=True)
    self.mode = mode

  def _get_model_filename(self, n):
    return os.path.join(self.path, "clf-{0}.joblib".format(n))

  def _fit_just_median(self, df_X, df_y):
    return np.median(np.array(df_X), axis=1)

  def _fit_weighting(self, df_X, df_y):
    y = np.array(df_y).ravel()
    cols = list(df_X.columns)
    X = np.array(df_X)
    p25 = np.percentile(X.ravel(), 25)
    p50 = np.percentile(X.ravel(), 50)
    p75 = np.percentile(X.ravel(), 75)
    scale = (p25, p50, p75)
    for c in cols:
      X = np.array(df_X[c]).reshape(-1, 1)
      mdl0 = PowerTransformer()
      mdl0.fit(X)
      X = mdl0.transform(X)
      mdl1 = LogisticRegressionCV()
      mdl1.fit(X, y)
      filename = self._get_model_filename(c)
      joblib.dump((mdl0, mdl1), filename)
    filename = self._get_model_filename("overall")
    joblib.dump(scale, filename)
    filename = self._get_model_filename("weighting")
    ws = WeightSchemes(df_X, df_y, "classification")
    ws.distance_to_leads()
    ws.importance()
    ws.save(filename)
    return self._predict_weighting(df_X)

  def _fit_model(self, df_X, df_y):
    y = np.array(df_y).ravel()
    cols = list(df_X.columns)
    for c in cols:
      X = np.array(df_X[c]).reshape(-1, 1)
      mdl = LogisticRegressionCV()
      mdl.fit(X, y)
      filename = self._get_model_filename(c)
      joblib.dump(mdl, filename)
    return self._predict_model(df_X)

  def _predict_just_median(self, df_X):
    return np.median(np.array(df_X), axis=1)

  def _predict_weighting(self, df_X):
    cols = list(df_X.columns)
    Y_hat = []
    for c in cols:
      filename = self._get_model_filename(c)
      if os.path.exists(filename):
        mdl0, mdl1 = joblib.load(filename)
        X = np.array(df_X[c]).reshape(-1, 1)
        X = mdl0.transform(X)
        y_hat = mdl1.predict_proba(X)[:, 1]
        Y_hat += [y_hat]
    Y_hat = np.array(Y_hat).T
    filename = self._get_model_filename("overall")
    filename = self._get_model_filename("weighting")
    weights = joblib.load(filename)
    wvals = weights["weights"]
    y_hats = []
    for k, v in wvals.items():
      w = np.array([v[c] for c in cols])
      if np.sum(w) == 0:
        w = w + 1
      y_hats += [np.average(Y_hat, axis=1, weights=w)]
    y_hats += [np.mean(Y_hat, axis=1)]
    y_hats = np.array(y_hats)
    y_hat = np.mean(y_hats, axis=0)
    return y_hat

  def _predict_model(self, df_X):
    cols = list(df_X.columns)
    Y_hat = []
    for c in cols:
      filename = self._get_model_filename(c)
      if os.path.exists(filename):
        mdl = joblib.load(filename)
        X = np.array(df_X[c]).reshape(-1, 1)
        y_hat = mdl.predict_proba(X)[:, 1]
        Y_hat += [y_hat]
    Y_hat = np.array(Y_hat).T
    return np.median(Y_hat, axis=1)

  def fit(self, df_X, df_y):
    if self.mode == "weighting":
      return self._fit_weighting(df_X, df_y)
    if self.mode == "median":
      return self._fit_just_median(df_X, df_y)
    if self.mode == "model":
      return self._fit_model(df_X, df_y)

  def predict(self, df_X):
    if self.mode == "weighting":
      return self._predict_weighting(df_X)
    if self.mode == "median":
      return self._predict_just_median(df_X)
    if self.mode == "model":
      return self._predict_model(df_X)


class BaggerRegressor(object):
  def __init__(self, path, mode="scaling"):
    assert mode in ["scaling", "median", "model"]
    self.path = path
    if not os.path.exists(self.path):
      os.makedirs(self.path, exist_ok=True)
    self.mode = mode

  def _get_model_filename(self, n):
    return os.path.join(self.path, "reg-{0}.joblib".format(n))

  def _fit_just_median(self, df_X, df_y):
    return np.median(np.array(df_X), axis=1)

  def _fit_scaling(self, df_X, df_y):
    cols = list(df_X.columns)
    X = np.array(df_X)
    p25 = np.percentile(X.ravel(), 25)
    p50 = np.percentile(X.ravel(), 50)
    p75 = np.percentile(X.ravel(), 75)
    scale = (p25, p50, p75)
    for c in cols:
      X = np.array(df_X[c]).reshape(-1, 1)
      mdl = RobustScaler()
      mdl.fit(X)
      filename = self._get_model_filename(c)
      joblib.dump(mdl, filename)
    filename = self._get_model_filename("overall")
    joblib.dump(scale, filename)
    filename = self._get_model_filename("weighting")
    ws = WeightSchemes(df_X, df_y, "classification")
    ws.distance_to_leads()
    ws.importance()
    ws.save(filename)
    return self._predict_scaling(df_X)

  def _fit_model(self, df_X, df_y):
    y = np.array(df_y).ravel()
    cols = list(df_X.columns)
    for c in cols:
      X = np.array(df_X[c]).reshape(-1, 1)
      mdl = LinearRegression()
      mdl.fit(X, y)
      filename = self._get_model_filename(c)
      joblib.dump(mdl, filename)
    return self._predict_model(df_X)

  def _predict_just_median(self, df_X):
    return np.median(np.array(df_X), axis=1)

  def _predict_scaling(self, df_X):
    cols = list(df_X.columns)
    Y_hat = []
    for c in cols:
      filename = self._get_model_filename(c)
      if os.path.exists(filename):
        mdl = joblib.load(filename)
        X = np.array(df_X[c]).reshape(-1, 1)
        y_hat = mdl.transform(X).ravel()
        Y_hat += [y_hat]
    Y_hat = np.array(Y_hat).T
    filename = self._get_model_filename("overall")
    scale = joblib.load(filename)
    iqr = scale[-1] - scale[0]
    med = scale[1]
    Y_hat = Y_hat * iqr + med
    filename = self._get_model_filename("weighting")
    weights = joblib.load(filename)
    wvals = weights["weights"]
    y_hats = []
    for k, v in wvals.items():
      w = np.array([v[c] for c in cols])
      if np.sum(w) == 0:
        w = w + 1
      y_hats += [np.average(Y_hat, axis=1, weights=w)]
    y_hats += [np.mean(Y_hat, axis=1)]
    y_hat = np.mean(y_hats, axis=0)
    return y_hat

  def _predict_model(self, df_X):
    cols = list(df_X.columns)
    Y_hat = []
    for c in cols:
      filename = self._get_model_filename(c)
      if os.path.exists(filename):
        mdl = joblib.load(filename)
        X = np.array(df_X[c]).reshape(-1, 1)
        y_hat = mdl.predict(X)
        Y_hat += [y_hat]
    Y_hat = np.array(Y_hat).T
    return np.mean(Y_hat, axis=1)

  def fit(self, df_X, df_y):
    if self.mode == "scaling":
      return self._fit_scaling(df_X, df_y)
    if self.mode == "median":
      return self._fit_just_median(df_X, df_y)
    if self.mdoe == "model":
      return self._fit_model(df_X, df_y)

  def predict(self, df_X):
    if self.mode == "scaling":
      return self._predict_scaling(df_X)
    if self.mode == "median":
      return self._predict_just_median(df_X)
    if self.mode == "model":
      return self._predict_model(df_X)
