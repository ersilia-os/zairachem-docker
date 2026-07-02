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
from zairachem.describe.descriptors.raw import DescribeMonitor
from zairachem.base.utils.isaura_report import quiet_isaura_reads
from zairachem.treat.imputers import DescriptorBase
from zairachem.base.utils.utils import fetch_schema_from_github
from zairachem.base.utils.logging import logger
from zairachem.base.utils.progress import STEP_COLORS
from zairachem.base.utils.pipeline import PipelineStep
from zairachem.base.vars import (
  DATA_SUBFOLDER,
  METADATA_SUBFOLDER,
  PARAMETERS_FILE,
  ERSILIA_DATA_FILENAME,
  PROJECTIONS_FILENAME,
  PROJECTIONS_MANIFEST_FILENAME,
  DEFAULT_ISAURA_BATCH_SIZE,
)


class ProjectionMonitor(DescribeMonitor):
  """Live per-model table for the Treat step's Ersilia projection models (same columns/events as
  the Describe table, relabelled). Runs after the transformers table has closed — never nested."""

  item_label = "Projection"
  title = "Computing projections"
  running_verb = "projecting"


class Manifolds(DescriptorBase):
  def __init__(self, batch_size=None):
    DescriptorBase.__init__(self)
    self.path = self.get_output_dir()
    self.input_file = os.path.join(self.path, DATA_SUBFOLDER, ERSILIA_DATA_FILENAME)
    self.params = self._load_params()
    self.projection_ids = self.params.get("projection_ids") or []
    self.batch_size = batch_size or DEFAULT_ISAURA_BATCH_SIZE

  def _load_params(self):
    with open(os.path.join(self.path, METADATA_SUBFOLDER, PARAMETERS_FILE), "r") as f:
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

  def _run_projection_model(self, model_id, progress_cb=None):
    """Compute a projection model through the store-aware client, returning per-row dicts + columns.

    Reuses ``BinaryStreamClient`` so projections are cached in the isaura project store exactly like
    featurizers (hybrid read-from-project + contribute-back, with the heavy→JSON fallback). Returns
    ``(rows, cols)`` where ``rows`` is a list of per-row dicts keyed by the model's output columns.
    """
    from zairachem.describe.descriptors.api import BinaryStreamClient

    client = BinaryStreamClient(
      csv_path=self.input_file,
      model_id=model_id,
      project_name=os.path.basename(self.path),
      batch_size=self.batch_size,
    )
    client.url = get_model_url(model_id)
    client._provenance_kind = "projections"  # count toward the projector group, not featurizers
    if progress_cb is not None:
      client._progress_cb = progress_cb
      client._show_progress = False
    res = client.run(output_h5=None, isaura_batch_size=self.batch_size)
    cols = res.get("dims") or fetch_schema_from_github(model_id)[0]
    data = res.get("data")
    if data is None and res.get("h5_file"):
      # Defensive: if the client ever streams to h5 instead of returning in-memory rows, read the
      # coordinates back rather than silently writing an empty projection.
      from zairachem.base.utils.matrices import open_h5

      h5 = open_h5(res["h5_file"])
      if h5 is not None:
        cols = h5.features()
        data = h5.values()
    if data is None:
      raise RuntimeError(
        f"Projection model '{model_id}' returned no coordinates — refusing to write an empty "
        f"projection. Check the model container logs (docker logs)."
      )
    rows = [dict(zip(cols, row)) for row in data]
    return rows, cols

  def _compute_ersilia_projections(self):
    """Each requested Ersilia projection model → one or more (x, y) projection pairs."""
    projections = []
    monitor = ProjectionMonitor(
      self.projection_ids, color=STEP_COLORS.get("projections", "bright_cyan")
    )
    # The table is the single live region here, so neutralize isaura's read-time bars (the client
    # reads cached projections from the store on the hybrid path) to avoid colliding live displays.
    with monitor.live(), quiet_isaura_reads():
      for model_id in self.projection_ids:
        monitor.start(model_id)
        try:
          rows, cols = self._run_projection_model(model_id, progress_cb=monitor.apply_event)
        except SystemExit:
          monitor.finish(model_id, ok=False)
          raise
        except Exception as e:
          monitor.finish(model_id, ok=False)
          raise self._projection_model_error(model_id, e) from e
        n_pairs = 0
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
          n_pairs += 1
        monitor.update_fields(model_id, source=f"{n_pairs} projection(s)")
        monitor.finish(model_id, ok=True)
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
    step = PipelineStep("manifolds", self.path)
    if step.is_done():
      logger.info("[manifolds] Projections already computed — skipping.")
      return
    self.inputs = self._load_data()
    logger.info(f"[manifolds] Computing projections for {len(self.inputs)} molecules")
    projections = [self._compute_mw_logp()]
    if self.projection_ids:
      projections += self._compute_ersilia_projections()
    self._write_projections(projections)
    logger.info(f"[manifolds] Wrote {len(projections)} projection(s)")
    step.update()
