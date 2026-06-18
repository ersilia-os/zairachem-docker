"""Compute the low-dimensional projections shown in the report.

Projections are 2-D embeddings used only for visualization (they are not model features). There is
always a built-in **molecular weight vs LogP** projection (RDKit, no model needed), so projections
never fail. Additional Ersilia projection models can be requested via ``projection_ids``; each is
computed through its served API and split into (x, y) pairs.

Output (row-aligned to ``data/data.csv``), both consumed by the report:
  * ``data/projections.csv``      — ``input`` + one ``<x>``/``<y>`` column per projection pair.
  * ``data/projections.json``     — manifest: ``[{"name","title","x","y"}, …]``.
"""

import csv, json, os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from zairachem.describe.descriptors.utils import get_model_url
from zairachem.treat.imputers import DescriptorBase
from zairachem.base.utils.utils import fetch_schema_from_github, post
from zairachem.base.utils.logging import logger
from zairachem.base.vars import (
  DATA_SUBFOLDER,
  PARAMETERS_FILE,
  ERSILIA_DATA_FILENAME,
  PROJECTIONS_FILENAME,
  PROJECTIONS_MANIFEST_FILENAME,
  DEFAULT_ISAURA_BATCH_SIZE,
)


class Manifolds(DescriptorBase):
  def __init__(self, batch_size=None):
    DescriptorBase.__init__(self)
    self.path = self.get_output_dir()
    self.input_file = os.path.join(self.path, DATA_SUBFOLDER, ERSILIA_DATA_FILENAME)
    self.params = self._load_params()
    self.projection_ids = self.params.get("projection_ids") or []
    self.batch_size = batch_size or DEFAULT_ISAURA_BATCH_SIZE

  def _load_params(self):
    with open(os.path.join(self.path, DATA_SUBFOLDER, PARAMETERS_FILE), "r") as f:
      return json.load(f)

  def _load_data(self):
    with open(self.input_file, "r") as f:
      reader = csv.reader(f)
      next(reader)  # header
      return [row[0] for row in reader]

  def group_by_prefix(self, names):
    """Group column names by their prefix (before the first underscore)."""
    groups = {}
    for name in names:
      prefix, *_ = name.split("_", 1)
      groups.setdefault(prefix, []).append(name)
    return groups

  def _compute_mw_logp(self):
    """Built-in baseline projection: molecular weight (x) vs LogP (y). Always available."""
    xs, ys = [], []
    for smi in self.inputs:
      mol = Chem.MolFromSmiles(smi) if smi else None
      if mol is None:
        xs.append(np.nan)
        ys.append(np.nan)
      else:
        xs.append(round(float(Descriptors.MolWt(mol)), 4))
        ys.append(round(float(Descriptors.MolLogP(mol)), 4))
    return {
      "name": "mwlogp",
      "title": "Molecular weight vs LogP",
      "x": "mw_x",
      "y": "logp_y",
      "x_values": xs,
      "y_values": ys,
    }

  def _projection_model_error(self, model_id, exc):
    return RuntimeError(
      f"Projection model '{model_id}' failed to compute — its served Docker image looks broken "
      f"({exc}). Re-fetch the model (e.g. `ersilia fetch {model_id}`) or rebuild its image, then "
      f"re-run."
    )

  def _compute_ersilia_projections(self):
    """Each requested Ersilia projection model → one or more (x, y) projection pairs via its API."""
    projections = []
    for model_id in self.projection_ids:
      url = get_model_url(model_id)
      try:
        rows = post(self.inputs, url)  # list of per-row dicts keyed by schema column
      except RuntimeError as e:
        raise self._projection_model_error(model_id, e) from e
      cols = fetch_schema_from_github(model_id)[0]
      for prefix, members in self.group_by_prefix(cols).items():
        if len(members) < 2:
          continue
        xcol, ycol = members[0], members[1]
        projections.append({
          "name": prefix,
          "title": f"{prefix.upper()} projection",
          "x": xcol,
          "y": ycol,
          "x_values": [r.get(xcol) for r in rows],
          "y_values": [r.get(ycol) for r in rows],
        })
    return projections

  def _write_projections(self, projections):
    data_dir = os.path.join(self.path, DATA_SUBFOLDER)
    df = pd.DataFrame({"input": self.inputs})
    manifest = []
    for p in projections:
      df[p["x"]] = p["x_values"]
      df[p["y"]] = p["y_values"]
      manifest.append({"name": p["name"], "title": p["title"], "x": p["x"], "y": p["y"]})
    df.to_csv(os.path.join(data_dir, PROJECTIONS_FILENAME), index=False)
    with open(os.path.join(data_dir, PROJECTIONS_MANIFEST_FILENAME), "w") as f:
      json.dump(manifest, f, indent=2)

  def run(self):
    self.inputs = self._load_data()
    logger.info(f"[manifolds] Computing projections for {len(self.inputs)} molecules")
    projections = [self._compute_mw_logp()]
    if self.projection_ids:
      projections += self._compute_ersilia_projections()
    self._write_projections(projections)
    logger.info(f"[manifolds] Wrote {len(projections)} projection(s)")
