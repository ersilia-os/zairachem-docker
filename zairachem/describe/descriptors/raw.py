import json, os
import pandas as pd
from zairachem.base import ZairaBase
from zairachem.base.utils.matrices import Hdf5
from zairachem.describe.descriptors.api import BinaryStreamClient
from zairachem.describe.descriptors.utils import Hdf5Data, get_model_url
from zairachem.base.vars import (
  PARAMETERS_FILE,
  DATA_SUBFOLDER,
  DATA_FILENAME,
  DESCRIPTORS_SUBFOLDER,
  REFERENCE_DESCRIPTOR,
  RAW_DESC_FILENAME,
  ERSILIA_DATA_FILENAME,
  MODELS_WITH_PORT,
)


class RawLoader(ZairaBase):
  def __init__(self):
    ZairaBase.__init__(self)
    self.path = self.get_output_dir()

  def open(self, eos_id):
    path = os.path.join(self.path, DESCRIPTORS_SUBFOLDER, eos_id, RAW_DESC_FILENAME)
    return Hdf5(path)


class RawDescriptors(ZairaBase):
  def __init__(self):
    ZairaBase.__init__(self)
    self.path = self.get_output_dir()
    self.params = self._load_params()
    self.input_csv = os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME)
    self.input_csv_ersilia = os.path.join(self.path, DATA_SUBFOLDER, ERSILIA_DATA_FILENAME)
    self._process_ersilia_inputs()
    self.api = BinaryStreamClient(csv_path=self.input_csv_ersilia)
    if self.is_predict():
      self.trained_path = self.get_trained_dir()

  def _load_params(self):
    with open(os.path.join(self.path, DATA_SUBFOLDER, PARAMETERS_FILE), "r") as f:
      params = json.load(f)
    return params

  def _process_ersilia_inputs(self):
    df = pd.read_csv(self.input_csv)
    df = df["smiles"]
    df.to_csv(self.input_csv_ersilia, index=False)

  def eos_ids(self):
    eos_ids = list(set(self.params["ersilia_hub"]))
    if REFERENCE_DESCRIPTOR not in eos_ids:
      eos_ids += [REFERENCE_DESCRIPTOR]
    return eos_ids

  def done_eos_ids(self):
    with open(os.path.join(self.trained_path, DESCRIPTORS_SUBFOLDER, "done_eos.json"), "r") as f:
      done_eos_ids = json.load(f)
    return done_eos_ids

  def output_h5_filename(self, eos_id):
    path = os.path.join(self.path, DESCRIPTORS_SUBFOLDER, eos_id)
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, RAW_DESC_FILENAME)

  def _run_eos(self, eos_id):
    output_h5 = self.output_h5_filename(eos_id)
    res = self.api.run()
    try:
      Hdf5Data(res).save(output_h5)
    except Exception as e:
      self.logger.error(f"Exception in h5: {e}")

  def run(self):
    done_eos = []
    if self.is_predict():
      eos_ids = self.done_eos_ids()
    else:
      eos_ids = self.eos_ids()
    for i, eos_id in enumerate(eos_ids):
      self.api.url = get_model_url(
        eos_id
      )  
      self.logger.info(f"An api url {self.api.url} assigned for descriptor model [green]{eos_id}[/]")
      try:
        self._run_eos(eos_id)
        done_eos += [eos_id]
      except:
        continue
    with open(os.path.join(self.path, DESCRIPTORS_SUBFOLDER, "done_eos.json"), "w") as f:
      json.dump(done_eos, f, indent=4)
