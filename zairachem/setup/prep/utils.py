import collections

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

ALPHA_DETECT_TAIL_CENTER = 0.05
ALPHA_REPEATS = 0.01
NOISE_LEVEL = 1e-6
RADIUS = 3
NBITS = 2048


class Fingerprinter(object):
  """Morgan fingerprint calculator (radius 3, 2048 bits) for SMILES lists."""

  def __init__(self):
    pass

  def _compute_ecfp4_fingerprints(self, smiles_list):
    fingerprints = []
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=NBITS)
    for smiles in smiles_list:
      mol = Chem.MolFromSmiles(smiles)
      if mol:
        fp = mfpgen.GetFingerprint(mol)
        fingerprints.append(np.array(fp))
      else:
        fingerprints.append(None)
    return np.array([fp for fp in fingerprints if fp is not None])

  def calculate(self, smiles_list):
    """Return a fingerprint matrix for the SMILES list, omitting unparseable molecules."""
    X = self._compute_ecfp4_fingerprints(smiles_list)
    return X


class SmoothenY(object):
  """Smoother that spreads heavily repeated regression values (e.g. censored) into a range."""

  def __init__(self, smiles_list, y):
    self.smiles_list = smiles_list
    self.y = np.array(y)

  @staticmethod
  def detect_repeats(y):
    """Return the values repeated more than max(1% of ``y``, 3) times."""
    max_repeats = max(len(y) * ALPHA_REPEATS, 3)
    counts = collections.defaultdict(int)
    for y_ in y:
      counts[float(y_)] += 1
    repeats = []
    for k, v in counts.items():
      if v > max_repeats:
        repeats += [k]
    return repeats

  @staticmethod
  def get_nonrepeat_idxs(y, repeats):
    """Return the indices of ``y`` whose value is not in ``repeats``."""
    repeats = set(repeats)
    idxs = []
    for i, v in enumerate(y):
      v = float(v)
      if v not in repeats:
        idxs += [i]
    return np.array(idxs)

  @staticmethod
  def detect_tail_center_region(y):
    """Return the 5th and 95th percentiles of ``y`` as ``(lower, upper)`` bounds."""
    alpha = ALPHA_DETECT_TAIL_CENTER
    lb = np.percentile(y, alpha * 100)
    ub = np.percentile(y, (1 - alpha) * 100)
    return lb, ub

  @staticmethod
  def get_boundaries(y, repeats, lb, ub):
    """Return, per repeated value, the ``(low, high)`` interval its occurrences are spread over.

    Central values span the neighbouring non-repeated values; values in a tail (below ``lb`` or
    above ``ub``) extend outwards from the repeated value.
    """
    alpha = 0.01
    boundaries = {}
    for r in repeats:
      ysu = y[y > r]
      ysl = y[y < r]
      if r >= lb and r <= ub:
        ulim = np.percentile(ysu, alpha * 100)
        llim = np.percentile(ysl, (1 - alpha) * 100)
        t = (llim, ulim)
      elif r < lb:
        ulim = np.percentile(ysu, alpha * 100)
        llim = r - (ulim - r)
        t = (llim, r)
      elif r > ub:
        llim = np.percentile(ysl, alpha * 100)
        ulim = r + (r - llim)
        t = (r, ulim)
      else:
        pass
      boundaries[r] = t
    return boundaries

  @staticmethod
  def get_repeats_idxs(y, repeats):
    """Return a dict mapping each repeated value to the indices of ``y`` holding it."""
    repeats = set(repeats)
    repeats_idxs = collections.defaultdict(list)
    for i, v in enumerate(y):
      if v in repeats:
        repeats_idxs[v] += [i]
    return repeats_idxs

  def run(self):
    """Return ``y`` with repeated values replaced by evenly spaced values in their interval.

    Order within each repeated group follows a random forest trained on Morgan fingerprints of the
    non-repeated molecules; a tiny Gaussian noise is then added to all values.
    """
    y = self.y
    smiles_list = self.smiles_list
    repeats = self.detect_repeats(y)
    idxs = self.get_nonrepeat_idxs(y, repeats)
    repeats_idxs = self.get_repeats_idxs(y, repeats)
    lb, ub = self.detect_tail_center_region(y[idxs])
    boundaries = self.get_boundaries(y[idxs], repeats, lb, ub)
    fps = Fingerprinter()
    X = fps.calculate(smiles_list)
    X_f = X[idxs]
    y_f = y[idxs]
    # sklearn only for regression smoothing; imported here, off the classification path.
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import QuantileTransformer

    ranker = QuantileTransformer(output_distribution="uniform")
    ranker.fit(y_f.reshape(-1, 1))
    y_f = ranker.transform(y_f.reshape(-1, 1)).ravel()
    mdl = RandomForestRegressor()
    mdl.fit(X_f, y_f)
    repeats_preds = {}
    for k, ridxs in repeats_idxs.items():
      X_r = X[ridxs]
      repeats_preds[k] = mdl.predict(X_r)
    repeats_ranks = {}
    for k, v in repeats_preds.items():
      repeats_ranks[k] = np.argsort(v)
    repeats_values = {}
    for k, v in repeats_ranks.items():
      sorted_values = np.linspace(boundaries[k][0], boundaries[k][1], len(v))
      values = np.zeros(len(v))
      for i, x in enumerate(v):
        values[x] = sorted_values[i]
      repeats_values[k] = values
    for k, ridxs in repeats_idxs.items():
      v = repeats_values[k]
      y[ridxs] = v
    noise = np.random.normal(loc=0, scale=np.std(y) * NOISE_LEVEL, size=len(y))
    y = y + noise
    return y
