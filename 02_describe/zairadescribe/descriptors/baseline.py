from ersilia import ErsiliaModel
import h5py
import os

from .treated import FullLineSimilarityImputer, Imputer
from .reference import REFERENCE_FOLDER_NAME

from .. import ZairaBase
from zairabase.vars import DESCRIPTORS_SUBFOLDER

class Embedder(ZairaBase):
    def __init__(self):
        ZairaBase.__init__(self)
        self.dim = 5000
        self.model = REFERENCE_FOLDER_NAME

    def _calculate(self, smiles_list, output_h5):
        if output_h5 is None:
            with ErsiliaModel(self.model) as mdl:
                X = mdl.api(api_name=None, input=smiles_list, output="numpy")
            return X
        else:
            with ErsiliaModel(self.model) as mdl:
                mdl.api(api_name=None, input=smiles_list, output=output_h5)

    def calculate(self, smiles_list, output_h5=None):
        X = self._calculate(smiles_list, output_h5)
        if X is None:
            with h5py.File(output_h5, "r") as f:
                X = f["Values"][:]
        imp = FullLineSimilarityImputer()
        imp_simple = Imputer()
        trained_path = self.get_trained_dir()
        path = self.get_output_dir()
        if not self.is_predict():
            imp.fit(X, smiles_list)
            X = imp.transform(X, smiles_list)
            imp.save(
                os.path.join(
                    path, DESCRIPTORS_SUBFOLDER, "{0}.joblib".format(imp._prefix)
                )
            )
            imp_simple.fit(X)
            imp_simple.save(
                os.path.join(path, DESCRIPTORS_SUBFOLDER, "imputer_simple.joblib")
            )
        else:
            fn = os.path.join(
                trained_path, DESCRIPTORS_SUBFOLDER, "{0}.joblib".format(imp._prefix)
            )
            fn_simple = os.path.join(
                trained_path, DESCRIPTORS_SUBFOLDER, "imputer_simple.joblib"
            )
            if os.path.exists(fn):
                imp = imp.load(fn)
                X = imp.transform(X, smiles_list)
            else:
                imp_simple = imp_simple.load(fn_simple)
                X = imp_simple.transform(X)
        if output_h5 is None:
            return X
        else:
            with h5py.File(output_h5, "r") as f:
                keys = f["Keys"][:]
                inputs = f["Inputs"][:]
                features = f["Features"][:]
            os.remove(output_h5)
            with h5py.File(output_h5, "w") as f:
                f.create_dataset("Keys", data=keys)
                f.create_dataset("Features", data=features)
                f.create_dataset("Inputs", data=inputs)
                f.create_dataset("Values", data=X)
