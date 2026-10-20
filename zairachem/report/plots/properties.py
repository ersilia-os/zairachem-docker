import numpy as np

from zairachem.report import BasePlot
from zairachem.report.fetcher import ResultsFetcher
from zairachem.report.plots._common import (
  _draw_one_property,
  _rdkit_properties,
)


class _PropertyDistributionPlot(BasePlot):
  """Base for a single physico-chemical property histogram (actives vs inactives). Subclasses set
  ``stem``, ``label`` and ``prop_index`` (0=MW, 1=LogP, 2=TPSA in ``_rdkit_properties``)."""

  stem = "property"
  label = ""
  prop_index = 0

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 4))
    self.name = self.stem  # BasePlot.__init__ sets self.name="base"; use the subclass stem
    self.is_available = False
    if not self.has_clf_data():
      return
    rf = ResultsFetcher(path=path)
    smiles = rf.get_smiles()
    bt = rf.get_actives_inactives()
    if not smiles or bt is None:
      return
    keep = ~np.isnan(np.asarray(bt, dtype=float))
    smiles = [s for s, k in zip(smiles, keep) if k]
    bt = np.asarray(bt, dtype=float)[keep].astype(int)
    if len(bt) == 0:
      return
    values = _rdkit_properties(smiles)[self.prop_index]
    self.is_available = _draw_one_property(self.ax, values, bt, self.label)


class PropertyMwPlot(_PropertyDistributionPlot):
  """Molecular-weight distribution by class."""

  stem = "property-mw"
  label = "Molecular weight"
  prop_index = 0


class PropertyLogpPlot(_PropertyDistributionPlot):
  """LogP distribution by class."""

  stem = "property-logp"
  label = "LogP"
  prop_index = 1
