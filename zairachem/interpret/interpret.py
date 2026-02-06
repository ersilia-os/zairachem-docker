import os
import pandas as pd
import json
from tqdm import tqdm

from zairachem.base.vars import (
  OUTPUT_FILENAME,
  SMILES_COLUMN,
  INTERPRETABILITY_SUBFOLDER,
  GLOBAL_INTERPRET_SUBFOLDER,
  SUBSTRUCTURE_INTERPRET_SUBFOLDER,
  DATAMOL_MODEL_FILENAME,
  ACCFG_MODEL_FILENAME,
)

from xai4chem.supervised import Regressor
from xai4chem.representations import DatamolDescriptor, MorganFingerprint, AccFgFingerprint

from zairachem.base import ZairaBase
from zairachem.base.utils.pipeline import PipelineStep

class ExplainFitter(ZairaBase):
  def __init__(self, path):
    ZairaBase.__init__(self)
    if path is None:
      self.path = self.get_output_dir()
    else:
      self.path = path
    self.output_dir = os.path.abspath(self.path)
    assert os.path.exists(self.output_dir)

  def _get_model_output(self):
    df = pd.read_csv(os.path.join(self.path, OUTPUT_FILENAME))
    preds_col = [col for col in df.columns if ("clf" in col and "bin" not in col) or "reg" in col]
    assert len(preds_col) == 1, "More than one predictions column detected"
    return df, preds_col[0]

  def fit_accfg(self):
    accfg_path = os.path.join(self.output_dir, INTERPRETABILITY_SUBFOLDER, SUBSTRUCTURE_INTERPRET_SUBFOLDER)
    os.makedirs(accfg_path, exist_ok=True)

    df, preds_col = self._get_model_output()
    smiles = df[SMILES_COLUMN].tolist()
    featurizer = AccFgFingerprint()
    X = featurizer.fit(smiles)
    y_pred = df[preds_col].tolist()

    model = Regressor(accfg_path, fingerprints="accfg")
    model.fit(X, y_pred)
    model.evaluate(X, smiles, y_pred)
    model.explain(smiles)
    
    model_filename = os.path.join(accfg_path, ACCFG_MODEL_FILENAME)
    model.save_model(model_filename)

  def fit_datamol(self):
    datamol_path = os.path.join(self.output_dir, INTERPRETABILITY_SUBFOLDER, GLOBAL_INTERPRET_SUBFOLDER)
    os.makedirs(datamol_path, exist_ok=True)

    df, preds_col = self._get_model_output()
    smiles = df[SMILES_COLUMN].tolist()
    featurizer = DatamolDescriptor()
    X = featurizer.fit(smiles)
    y_pred = df[preds_col].tolist()

    model = Regressor(datamol_path, fingerprints="datamol")
    model.fit(X, y_pred)
    model.evaluate(X, smiles, y_pred)
    model.explain(smiles)
    
    model_filename = os.path.join(datamol_path, DATAMOL_MODEL_FILENAME)
    model.save_model(model_filename)

  def run(self):
    self.fit_datamol()
    self.fit_accfg()


class Explainer(ZairaBase):
  def __init__(self, path, interpret_substructures=False):
    ZairaBase.__init__(self)
    if path is None:
      self.path = self.get_output_dir()
    else:
      self.path = path
    with open(os.path.join(self.get_output_dir(), "session.json")) as f:
      self.model_dir = json.load(f)["model_dir"]

    self.output_dir = os.path.abspath(self.path)
    assert os.path.exists(self.output_dir)
    self.global_explainer_path = os.path.join(self.model_dir, INTERPRETABILITY_SUBFOLDER, GLOBAL_INTERPRET_SUBFOLDER, DATAMOL_MODEL_FILENAME)
    assert os.path.exists(self.global_explainer_path)
    self.substructure_explainer_path = os.path.join(self.model_dir, INTERPRETABILITY_SUBFOLDER, SUBSTRUCTURE_INTERPRET_SUBFOLDER, ACCFG_MODEL_FILENAME)
    assert os.path.exists(self.substructure_explainer_path)
    self.interpret_substructures = interpret_substructures

  def _get_model_output(self):
    df = pd.read_csv(os.path.join(self.path, OUTPUT_FILENAME))
    return df

  def explain_accfg(self):
    accfg_path = os.path.join(self.output_dir, INTERPRETABILITY_SUBFOLDER, SUBSTRUCTURE_INTERPRET_SUBFOLDER)
    df = self._get_model_output()
    smiles = df[SMILES_COLUMN].tolist()

    model = Regressor(accfg_path)
    model.load_model(self.substructure_explainer_path)
    for smi in tqdm(smiles):
      model.explain_mol_atoms(smi)

  def explain_datamol(self):
    datamol_path = os.path.join(self.output_dir, INTERPRETABILITY_SUBFOLDER, GLOBAL_INTERPRET_SUBFOLDER)
    df = self._get_model_output()
    smiles = df[SMILES_COLUMN].tolist()

    model = Regressor(datamol_path)
    model.load_model(self.global_explainer_path)  
    model.explain_preds(smiles, datamol_path)

  def run(self):
    self.explain_datamol()
    if self.interpret_substructures:
      self.explain_accfg()


class Interpreter(ZairaBase):
  def __init__(self, path, interpret_substructures=False):
    ZairaBase.__init__(self)
    if path is None:
      self.path = self.get_output_dir()
    else:
      self.path = path
    self.output_dir = os.path.abspath(self.path)
    session_path = os.path.join(self.output_dir, "session.json")
    with open(session_path) as f:
      self.mode = json.load(f)["mode"]
    self.interpret_substructures = interpret_substructures

  def run(self):
    step = PipelineStep("interpretability", self.output_dir)
    if not step.is_done():
      if self.mode == "fit":
        ExplainFitter(path=self.path).run()
        if self.interpret_substructures:
          Explainer(path=self.path, interpret_substructures=self.interpret_substructures).explain_accfg()
      elif self.mode == "predict":
        Explainer(path=self.path, interpret_substructures=self.interpret_substructures).run()
      step.update()
    else:
      self.logger.warning(
        "[yellow]Interpretability step for predictions is already done. Skipping this step![/]"
      )
