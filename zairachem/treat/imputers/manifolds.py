"""Compute the low-dimensional projections shown in the report.

Projections are 2-D embeddings used only for visualization (they are not model features). There is
always a built-in **molecular weight vs LogP** projection (RDKit, no model needed), so projections
never fail. Additional Ersilia projection models can be requested via ``projection_ids``; each is
computed through its served API and split into (x, y) pairs.

Output (row-aligned to ``data/data.csv``), both consumed by the report:
  * ``data/projections.csv``      — ``input`` + one ``<x>``/``<y>`` column per projection pair.
  * ``data/projections.json``     — manifest: ``[{"name","title","x","y"}, …]``.
"""

import csv
import json
import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from zairachem.base.utils.isaura_report import quiet_isaura_reads
from zairachem.base.utils.logging import logger
from zairachem.base.utils.pipeline import PipelineStep
from zairachem.base.utils.progress import STEP_COLORS
from zairachem.base.utils.utils import fetch_schema_from_github
from zairachem.base.vars import (
  DATA_SUBFOLDER,
  DEFAULT_ISAURA_BATCH_SIZE,
  DESCRIPTORS_SUBFOLDER,
  ERSILIA_DATA_FILENAME,
  METADATA_SUBFOLDER,
  PARAMETERS_FILE,
  PROJECTION_CACHE_FILENAME,
  PROJECTIONS_FILENAME,
  PROJECTIONS_MANIFEST_FILENAME,
)
from zairachem.describe.descriptors.raw import DescribeMonitor
from zairachem.describe.descriptors.utils import get_model_url
from zairachem.treat.imputers import DescriptorBase


class ProjectionMonitor(DescribeMonitor):
  """Live per-model table for the Treat step's Ersilia projection models (same columns/events as
  the Describe table, relabelled). Runs after the transformers table has closed — never nested."""

  item_label = "Projection"
  title = "Computing projections"
  running_verb = "projecting"


class Manifolds(DescriptorBase):
  """Compute the 2-D projections shown in the report (molecular weight vs logP, plus Ersilia projection models).

  Parameters
  ----------
  path : str
    Run folder.
  batch_size : int, optional
    Rows per chunk when reading from or writing to the isaura store (the projection itself is one
    request to the model server).
  """

  def __init__(self, path, batch_size=None):
    DescriptorBase.__init__(self, path)
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

  def _run_projection_model(self, model_id, progress_cb=None, sampler=None):
    """Compute a projection model through the store-aware client, returning per-row dicts + columns.

    Reuses ``BinaryStreamClient`` so projections are cached in the isaura project store exactly like
    featurizers (hybrid read-from-project + contribute-back), in a single request. Returns
    ``(rows, cols)`` where ``rows`` is a list of per-row dicts keyed by the model's output columns.
    """
    from zairachem.describe.descriptors.api import BinaryStreamClient

    cached = self._read_cache(model_id)
    if cached is not None:
      return cached
    sample = self._projection_sample()
    client = BinaryStreamClient(
      path=self.path,
      csv_path=self.input_file if sample is None else self._write_sample(sample),
      model_id=model_id,
      project_name=os.path.basename(self.path),
    )
    client.url = get_model_url(model_id)
    client._provenance_kind = "projections"  # count toward the projector group, not featurizers
    client.splittable = False  # the projection embeds the molecules of a request together
    # One request of at most MAX_PROJECTION_MOLECULES: binary buys nothing here, and an image that
    # cannot read it can answer with empty rows, so projectors always use JSON.
    client.fixed_output_type = "simple"
    # The profile keeps the image's memory and cores (its response format is fixed above).
    from zairachem.base.utils.model_version import _local_image_digest
    from zairachem.describe.descriptors import profiles

    client.profile = profiles.load(model_id, _local_image_digest(model_id))
    client.sampler = sampler  # memory and cores into the profile, when describe measures them
    if progress_cb is not None:
      client._progress_cb = progress_cb
      client._show_progress = False
    try:
      res = client.run(output_h5=None, isaura_batch_size=self.batch_size)
    finally:
      profiles.save(client.profile)
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
    if data is None or not np.isfinite(np.asarray(data, dtype=float)).any():
      raise RuntimeError(
        f"Projection model '{model_id}' returned no coordinates — refusing to write an empty "
        f"projection. Check the model container logs (docker logs)."
      )
    rows = [dict(zip(cols, row)) for row in data]
    if sample is not None:
      full = [{} for _ in self.inputs]  # unsampled rows get no coordinates (NaN in the outputs)
      for i, row in zip(sample, rows):
        full[i] = row
      rows = full
    self._write_cache(model_id, rows, cols)
    return rows, cols

  def _cache_path(self, model_id):
    return os.path.join(self.path, DESCRIPTORS_SUBFOLDER, model_id, PROJECTION_CACHE_FILENAME)

  def _write_cache(self, model_id, rows, cols):
    """Keep a projection model's coordinates in the run (computed during describe, read here)."""
    path = self._cache_path(model_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    clean = [{k: (None if v is None else float(v)) for k, v in r.items()} for r in rows]
    with open(path, "w") as f:
      json.dump({"cols": list(cols), "rows": clean}, f)

  def _read_cache(self, model_id):
    """``(rows, cols)`` from the run's cached projection, or None if absent or for other inputs."""
    path = self._cache_path(model_id)
    if not os.path.exists(path):
      return None
    with open(path) as f:
      data = json.load(f)
    rows = data.get("rows") or []
    if len(rows) != len(self.inputs):
      return None
    if not any(v is not None and np.isfinite(v) for r in rows for v in r.values()):
      return None  # an empty projection cached by a model that could not read the request
    return rows, data["cols"]

  def precompute(self, model_id, progress_cb=None, sampler=None):
    """Compute and cache one projection model's coordinates (describe runs this alongside the
    descriptor models, so the projections step only reads them). ``progress_cb`` reports into the
    caller's live table; ``sampler`` measures the server's memory and cores into its profile."""
    self.inputs = self._load_data()
    self._run_projection_model(model_id, progress_cb=progress_cb, sampler=sampler)

  def _projection_sample(self):
    """Row indices projected when there are more molecules than one request may hold, else None.

    A projector embeds the molecules of a request together, and the server would split a larger
    request into chunks embedded separately, so at most ``MAX_PROJECTION_MOLECULES`` molecules (a
    fixed, seeded sample) are projected.
    """
    from zairachem.base.vars import MAX_PROJECTION_MOLECULES

    n = len(self.inputs)
    if n <= MAX_PROJECTION_MOLECULES:
      return None
    logger.info(
      f"[manifolds] Projecting a sample of {MAX_PROJECTION_MOLECULES:,} of {n:,} molecules"
    )
    rng = np.random.default_rng(0)
    return np.sort(rng.choice(n, MAX_PROJECTION_MOLECULES, replace=False)).tolist()

  def _write_sample(self, sample):
    path = os.path.join(self.path, DATA_SUBFOLDER, "projection_sample.csv")
    with open(path, "w", newline="") as f:
      writer = csv.writer(f)
      writer.writerow(["smiles"])
      for i in sample:
        writer.writerow([self.inputs[i]])
    return path

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
        monitor.update_fields(model_id, source=f"{n_pairs} projection{'' if n_pairs == 1 else 's'}")
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
    """Compute and save the projections, skipping them if already done."""
    step = PipelineStep("manifolds", self.path)
    if step.is_done():
      logger.info("[manifolds] Projections already computed — skipping.")
      return
    self.inputs = self._load_data()
    logger.info(f"[manifolds] Computing projections for {len(self.inputs)} molecules")
    projections = [self._compute_mw_logp()]
    if self.projection_ids:
      # Start the projection servers (shared between runs and stopped by the run that started them, like the descriptors').
      from zairachem.describe.descriptors.describe import Describer

      missing = [m for m in self.projection_ids if self._read_cache(m) is None]
      if missing:  # describe usually computed them already
        Describer(path=self.path).setup_model_servers(missing)
      projections += self._compute_ersilia_projections()
    self._write_projections(projections)
    logger.info(f"[manifolds] Wrote {len(projections)} projections")
    step.update()
