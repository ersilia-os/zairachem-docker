"""Tanimoto similarity of query molecules to a model's training set.

The similarity is the mean Tanimoto similarity to the ``k`` nearest training molecules on Morgan
fingerprints (radius 2, 2048 bits), the neighbourhood ersilia-os/eosquality uses for its training
distance (Sheridan et al. 2004). A query that is itself a training molecule (same standardised
SMILES) leaves its own entry out, so it gets its leave-one-out value. The search runs on an FPSim2
index.
"""

import os

import numpy as np

from zairachem.base.utils.logging import logger

#: Nearest training molecules averaged per query.
K_NEIGHBORS = 5
#: Morgan fingerprint used for the index and the queries.
FP_TYPE = "Morgan"
FP_PARAMS = {"radius": 2, "fpSize": 2048}
INDEX_FILENAME = "training_fps.h5"


def _build_index(train_smiles, path):
  """Write an FPSim2 index of ``train_smiles`` (ids are their positions) to ``path``."""
  from FPSim2.io import create_db_file

  create_db_file(
    [[smi, i] for i, smi in enumerate(train_smiles)],
    path,
    mol_format="smiles",
    fp_type=FP_TYPE,
    fp_params=FP_PARAMS,
  )


def training_similarity(query_smiles, train_smiles, workdir, k=K_NEIGHBORS):
  """Return each query's mean Tanimoto similarity to its ``k`` nearest training molecules.

  Parameters
  ----------
  query_smiles : list of str
      Standardised SMILES of the molecules to score.
  train_smiles : list of str
      Standardised SMILES of the model's training molecules.
  workdir : str
      Folder where the FPSim2 index of the training set is written.
  k : int, optional
      Neighbours averaged; capped at ``len(train_smiles) - 1`` for tiny training sets.

  Returns
  -------
  numpy.ndarray
      Mean similarity per query, in [0, 1] (1 = identical neighbourhood); NaN for a query RDKit
      cannot parse.
  """
  from FPSim2 import FPSim2Engine

  train_smiles = list(train_smiles)
  k = max(1, min(k, len(train_smiles) - 1))
  os.makedirs(workdir, exist_ok=True)
  index_path = os.path.join(workdir, INDEX_FILENAME)
  _build_index(train_smiles, index_path)
  engine = FPSim2Engine(index_path)
  out = np.full(len(query_smiles), np.nan)
  for i, smi in enumerate(query_smiles):
    try:
      hits = engine.top_k(smi, k + 1, threshold=0.0)
    except Exception as e:  # FPSim2 raises its own errors for SMILES RDKit cannot parse
      logger.debug(f"[similarity] no fingerprint for {smi!r}: {e}")
      continue
    sims = [float(h["coeff"]) for h in hits if train_smiles[int(h["mol_id"])] != smi][:k]
    if sims:
      out[i] = float(np.mean(sims))
  return out
