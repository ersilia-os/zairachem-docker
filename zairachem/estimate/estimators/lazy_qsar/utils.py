import collections


def make_classification_report(y, y_proba, y_rank=None, y_ad=None, threshold=0.5):
  y_hat = y_proba
  b_hat = (y_hat > threshold).astype(int)
  results = collections.OrderedDict()
  main = {"idxs": None, "y": y, "y_hat": y_hat, "b_hat": b_hat}
  # Per-sample reliability signals consumed by the reliability pooler. Optional: the pooler
  # degrades gracefully when a descriptor omits them (e.g. tiny datasets with no OOF).
  if y_rank is not None:
    main["r_hat"] = y_rank
  if y_ad is not None:
    main["a_hat"] = y_ad
  results["main"] = main
  return results
