"""RDKit-native molecular train/test splitters for held-out validation.

Every group-based strategy reduces to the same shape: assign an integer *group id* per molecule, then
place whole groups on one side of an 80/20 boundary — so scaffolds / fingerprint clusters never
straddle train and test. For each seeded schema a pool of candidate splits is generated and the most
balanced, mutually independent ones are selected (:func:`select_folds`); independence is measured as
low pairwise Jaccard overlap of the held-out sets, so the chosen folds cover different chemical space.

The functions here are pure (they take ``smiles`` + ``labels`` and return row-index arrays) so they can
be unit-tested without the rest of the pipeline. Classification only.
"""

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

from zairachem.base.utils.logging import logger

#: Seeded (randomized) strategies. ``scaffold_det`` is handled separately (single, deterministic).
STRATEGIES = ("random", "scaffold", "butina")

#: User-selectable fold schemas (``--evaluate``), in canonical execution/display order. Includes the
#: deterministic ``scaffold_det`` alongside the seeded families.
FOLD_SCHEMAS = ("random", "scaffold", "scaffold_det", "butina")

# ECFP4-equivalent Morgan fingerprint (radius 2, 2048 bits), matching the repo's other Morgan uses
# (setup/prep/utils.py).
_MORGAN_RADIUS = 2
_MORGAN_NBITS = 2048
# Butina works on a Tanimoto *distance* threshold; 0.6 distance (0.4 similarity) is a common cluster
# granularity for scaffold-like separation.
_BUTINA_CUTOFF = 0.6
_DEFAULT_TEST_FRACTION = 0.2
# Seeded candidates generated per schema before the best k are selected.
_CANDIDATE_POOL = 50


def split(smiles, labels, strategy, seed, test_fraction=_DEFAULT_TEST_FRACTION):
  """Draw one seeded 80/20 split for ``strategy``.

  Parameters
  ----------
  smiles : list of str
    Molecule SMILES, row-aligned with ``labels``.
  labels : array-like of int
    Binary class labels (0/1).
  strategy : {"random", "scaffold", "butina"}
    Splitting strategy.
  seed : int
    Random seed controlling the partition.
  test_fraction : float, optional
    Target held-out fraction (default 0.2).

  Returns
  -------
  tuple of numpy.ndarray or None
    ``(train_idx, test_idx)`` sorted index arrays, or ``None`` if the split does not place both
    classes in each side.
  """
  labels = np.asarray(labels)
  if strategy == "random":
    train_idx, test_idx = _split_random(labels, seed, test_fraction)
  elif strategy == "scaffold":
    train_idx, test_idx = _assign_by_groups(_groups_scaffold(smiles), seed, test_fraction)
  elif strategy == "butina":
    train_idx, test_idx = _assign_by_groups(
      _groups_butina(smiles, _BUTINA_CUTOFF), seed, test_fraction
    )
  else:
    raise ValueError(f"Unknown split strategy '{strategy}' (expected one of {STRATEGIES}).")
  if train_idx is None or not _both_classes(labels, train_idx, test_idx):
    return None
  return train_idx, test_idx


def scaffold_deterministic(smiles, labels, test_fraction=_DEFAULT_TEST_FRACTION):
  """DeepChem-style deterministic scaffold split (no seed).

  Bemis–Murcko scaffold groups are sorted largest-first and greedily assigned to train until adding
  the next group would exceed the train fraction; every remaining group goes to test. Stable across
  runs (ties broken by first appearance).

  Returns
  -------
  tuple of numpy.ndarray
    ``(train_idx, test_idx)`` sorted index arrays.
  """
  groups = _groups_scaffold(smiles)
  members = {}
  for i, g in enumerate(groups):
    members.setdefault(int(g), []).append(i)
  ordered = sorted(members.values(), key=lambda m: (-len(m), m[0]))
  train_cutoff = (1.0 - test_fraction) * len(smiles)
  train_idx, test_idx = [], []
  for m in ordered:
    if len(train_idx) + len(m) <= train_cutoff:
      train_idx.extend(m)
    else:
      test_idx.extend(m)
  return np.array(sorted(train_idx)), np.array(sorted(test_idx))


def select_folds(
  smiles, labels, strategy, k, pool=_CANDIDATE_POOL, test_fraction=_DEFAULT_TEST_FRACTION
):
  """Generate a candidate pool for ``strategy`` and pick the best ``k`` folds.

  Candidates that fail the both-classes guardrail are dropped. The first pick is the best-balanced
  candidate (train/test active rate closest to the overall rate, plus a small held-out-size penalty);
  each subsequent pick minimizes the maximum pairwise Jaccard overlap of held-out sets against those
  already chosen, so the selected folds are as mutually independent as possible.

  Returns
  -------
  list of dict
    Up to ``k`` dicts with keys ``seed``, ``train_idx``, ``test_idx`` (lists), ``balance``,
    ``mean_jaccard``. Empty if no valid candidate exists.
  """
  labels = np.asarray(labels)
  p = float(labels.mean())
  candidates = []
  for seed in range(pool):
    res = split(smiles, labels, strategy, seed, test_fraction)
    if res is None:
      continue
    train_idx, test_idx = res
    balance = abs(labels[train_idx].mean() - p) + abs(labels[test_idx].mean() - p)
    size_penalty = abs(len(test_idx) / len(labels) - test_fraction)
    candidates.append({
      "seed": seed,
      "train_idx": train_idx,
      "test_idx": test_idx,
      "balance": float(balance),
      "quality": float(balance + size_penalty),
      "test_set": frozenset(int(i) for i in test_idx),
    })
  if not candidates:
    logger.warning(
      f"[evaluate] strategy '{strategy}' produced no valid split (class balance); skipping."
    )
    return []

  candidates.sort(key=lambda c: c["quality"])
  chosen = [candidates[0]]
  chosen_seeds = {candidates[0]["seed"]}
  while len(chosen) < k and len(chosen) < len(candidates):
    best, best_score = None, None
    for c in candidates:
      if c["seed"] in chosen_seeds:
        continue
      max_jaccard = max(_jaccard(c["test_set"], s["test_set"]) for s in chosen)
      score = max_jaccard + c["quality"]  # both lower-is-better
      if best_score is None or score < best_score:
        best, best_score = c, score
    if best is None:
      break
    chosen.append(best)
    chosen_seeds.add(best["seed"])

  if len(chosen) < k:
    logger.warning(
      f"[evaluate] strategy '{strategy}': only {len(chosen)} of {k} requested folds are class-valid."
    )
  folds = []
  for c in chosen:
    others = [s for s in chosen if s["seed"] != c["seed"]]
    mean_jaccard = (
      float(np.mean([_jaccard(c["test_set"], s["test_set"]) for s in others])) if others else 0.0
    )
    folds.append({
      "seed": c["seed"],
      "train_idx": [int(i) for i in c["train_idx"]],
      "test_idx": [int(i) for i in c["test_idx"]],
      "balance": c["balance"],
      "mean_jaccard": mean_jaccard,
    })
  return folds


def build_fold_definitions(
  smiles, labels, repeats=3, schemas=None, test_fraction=_DEFAULT_TEST_FRACTION
):
  """Build the held-out fold menu, ready to serialize to ``metadata/splits.json``.

  ``schemas`` selects which families to include (subset of :data:`FOLD_SCHEMAS`; ``None`` = all). Each
  requested seeded family (random / scaffold / butina) contributes ``repeats`` folds chosen for balance
  and mutual independence; ``scaffold_det`` is a single deterministic fold. Fold names are stable
  (``random_00`` …). Lowering ``repeats`` or trimming ``schemas`` are the fidelity-preserving ways to
  cut compute — every fold still refits the exact production pipeline. Folds are emitted in canonical
  order (random → scaffold → scaffold_det → butina), which JSON preserves downstream.

  Returns
  -------
  dict
    Ordered mapping ``fold_name -> {strategy, seed, train_idx, test_idx, ...}`` (JSON-serializable).
  """
  labels = np.asarray(labels)
  wanted = set(schemas) if schemas else set(FOLD_SCHEMAS)
  folds = {}

  def _add_seeded(strategy, k):
    if strategy not in wanted or k < 1:
      return
    for i, fold in enumerate(
      select_folds(smiles, labels, strategy, k, test_fraction=test_fraction)
    ):
      folds[f"{strategy}_{i:02d}"] = {"strategy": strategy, **fold}

  _add_seeded("random", repeats)
  _add_seeded("scaffold", repeats)
  if "scaffold_det" in wanted:
    tr, te = scaffold_deterministic(smiles, labels, test_fraction)
    if not _both_classes(labels, tr, te):
      logger.warning(
        "[evaluate] deterministic scaffold split has a class entirely on one side; "
        "its held-out metrics may be undefined."
      )
    folds["scaffold_det"] = {
      "strategy": "scaffold_det",
      "seed": None,
      "train_idx": [int(i) for i in tr],
      "test_idx": [int(i) for i in te],
    }
  _add_seeded("butina", repeats)
  return folds


# --- internal helpers -------------------------------------------------------------------------


def _split_random(labels, seed, test_fraction):
  """Stratified random 80/20 split (both classes guaranteed by construction)."""
  from sklearn.model_selection import StratifiedShuffleSplit

  try:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_idx, test_idx = next(sss.split(np.zeros(len(labels)), labels))
  except ValueError:
    return None, None
  return np.array(sorted(train_idx)), np.array(sorted(test_idx))


def _groups_scaffold(smiles):
  """Group id per molecule by Bemis–Murcko scaffold; unparseable SMILES become singletons."""
  scaffold_to_id = {}
  groups = np.empty(len(smiles), dtype=int)
  next_id = 0
  for i, smi in enumerate(smiles):
    mol = Chem.MolFromSmiles(smi) if smi else None
    key = None
    if mol is not None:
      try:
        key = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
      except Exception:
        key = None
    if key is None:
      key = ("__invalid__", i)  # unique per row → its own singleton group
    if key not in scaffold_to_id:
      scaffold_to_id[key] = next_id
      next_id += 1
    groups[i] = scaffold_to_id[key]
  return groups


def _groups_butina(smiles, cutoff):
  """Group id per molecule by Butina clustering of Morgan fingerprints (Tanimoto distance)."""
  gen = rdFingerprintGenerator.GetMorganGenerator(radius=_MORGAN_RADIUS, fpSize=_MORGAN_NBITS)
  fps = []
  for smi in smiles:
    mol = Chem.MolFromSmiles(smi) if smi else None
    fps.append(gen.GetFingerprint(mol) if mol is not None else None)
  valid_idx = [i for i, fp in enumerate(fps) if fp is not None]
  valid_fps = [fps[i] for i in valid_idx]
  n = len(valid_fps)
  # Condensed lower-triangle distance list (1 - Tanimoto), the input Butina expects.
  dists = []
  for i in range(1, n):
    sims = DataStructs.BulkTanimotoSimilarity(valid_fps[i], valid_fps[:i])
    dists.extend(1.0 - s for s in sims)
  clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True) if n else []
  groups = np.empty(len(smiles), dtype=int)
  next_id = 0
  for cluster in clusters:
    for local_i in cluster:
      groups[valid_idx[local_i]] = next_id
    next_id += 1
  for i, fp in enumerate(fps):
    if fp is None:
      groups[i] = next_id
      next_id += 1
  return groups


def _assign_by_groups(group_ids, seed, test_fraction):
  """Place whole groups into test (in shuffled order) until the target held-out size is reached."""
  n = len(group_ids)
  target_test = int(round(test_fraction * n))
  rng = np.random.default_rng(seed)
  unique = np.unique(group_ids)
  rng.shuffle(unique)
  test = []
  for g in unique:
    if len(test) >= target_test:
      break
    test.extend(np.where(group_ids == g)[0].tolist())
  test_set = set(test)
  train_idx = np.array([i for i in range(n) if i not in test_set])
  test_idx = np.array(sorted(test_set))
  return train_idx, test_idx


def _both_classes(labels, train_idx, test_idx):
  """True iff both classes 0 and 1 appear in train AND in test."""
  labels = np.asarray(labels)
  if len(train_idx) == 0 or len(test_idx) == 0:
    return False
  return set(labels[train_idx].tolist()) >= {0, 1} and set(labels[test_idx].tolist()) >= {0, 1}


def _jaccard(a, b):
  """Jaccard similarity of two index sets (0 when both empty)."""
  if not a and not b:
    return 0.0
  union = len(a | b)
  return len(a & b) / union if union else 0.0
