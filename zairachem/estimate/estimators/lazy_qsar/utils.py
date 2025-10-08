import collections
from sklearn.metrics import roc_curve, auc


def make_classification_report(y, y_proba, threshold=0.5):
  y_hat = y_proba
  b_hat = (y_hat > threshold).astype(int)
  results = collections.OrderedDict()
  results["main"] = {"idxs": None, "y": y, "y_hat": y_hat, "b_hat": b_hat}
  return results
