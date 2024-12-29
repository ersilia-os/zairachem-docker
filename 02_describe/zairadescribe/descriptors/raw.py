import json
import os

from zairabase import ZairaBase
from zairabase.utils.matrices import Hdf5
from zairabase.vars import PARAMETERS_FILE, DATA_SUBFOLDER, DATA_FILENAME, DESCRIPTORS_SUBFOLDER, REFERENCE_DESCRIPTOR, RAW_DESC_FILENAME, SESSION_FILE

from ersilia import logger
from ersilia import ErsiliaModel

class RawLoader(ZairaBase):
    def __init__(self):
        ZairaBase.__init__(self)
        self.path = self.get_output_dir()

    def open(self, eos_id):
        path = os.path.join(self.path, DESCRIPTORS_SUBFOLDER, eos_id, RAW_DESC_FILENAME)
        return Hdf5(path)

class ModelArtifact(object):
    def __init__(self, model_id):
        self.model_id = model_id
        self.logger = logger
        try:
            self.load_model()
        except:
            self.model = None

    def load_model(self):
        self.model = ErsiliaModel(
            model=self.model_id,
            save_to_lake=False,
            service_class="pulled_docker",
            fetch_if_not_available=True,
        )

    def run(self,input_csv, output_h5):
        self.model.serve()
        self.model.run(input=input_csv, output=output_h5)
        self.model.close()
    
    def info(self):
        info = self.model.info()
        return info


class RawDescriptors(ZairaBase):
    def __init__(self):
        ZairaBase.__init__(self)
        self.path = self.get_output_dir()
        self.params = self._load_params()
        self.input_csv = os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME)
        if self.is_predict():
            self.trained_path = self.get_trained_dir()

    def _load_params(self):
        with open(os.path.join(self.path, DATA_SUBFOLDER, PARAMETERS_FILE), "r") as f:
            params = json.load(f)
        return params

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
        ma = ModelArtifact(eos_id)
        ma.run(self.input_csv, output_h5)
        print("success")
        Hdf5(output_h5).save_summary_as_csv()

    def run(self):
        done_eos = []
        if self.is_predict():
            print("HERE")
            eos_ids = self.done_eos_ids()
        else:
            eos_ids = self.eos_ids()
        for eos_id in eos_ids:
            try:
                self._run_eos(eos_id)
                done_eos += [eos_id]
            except:
                continue
        with open(
            os.path.join(self.path, DESCRIPTORS_SUBFOLDER, "done_eos.json"), "w"
        ) as f:
            json.dump(done_eos, f, indent=4)