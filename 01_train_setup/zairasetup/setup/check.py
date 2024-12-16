import os
import json
import pandas as pd
import csv
from rdkit import DataStructs
from rdkit import Chem
from standardiser import standardise

from zairabase.vars import INPUT_SCHEMA_FILENAME, RAW_INPUT_FILENAME, MAPPING_FILENAME
from zairabase.vars import COMPOUND_IDENTIFIER_COLUMN, MAPPING_ORIGINAL_COLUMN , MAPPING_DEDUPE_COLUMN, VALUES_COLUMN, SMILES_COLUMN
from zairabase.vars import DATA_SUBFOLDER, DATA_FILENAME


class SetupChecker(object):
    def __init__(self, path):
        for f in os.listdir(path):
            if RAW_INPUT_FILENAME in f:
                self.input_file = os.path.join(path, f)
        self.input_schema = os.path.join(path, DATA_SUBFOLDER, INPUT_SCHEMA_FILENAME)
        self.data_file = os.path.join(path, DATA_SUBFOLDER, DATA_FILENAME)
        self.mapping_file = os.path.join(path, DATA_SUBFOLDER, MAPPING_FILENAME)

    def _get_input_schema(self):
        with open(self.input_schema, "r") as f:
            self.input_schema_dict = json.load(f)

    def remap(self):
        dm = pd.read_csv(self.mapping_file)
        dd = pd.read_csv(self.data_file)
        cid_mapping = list(dm[COMPOUND_IDENTIFIER_COLUMN])
        cid_data = list(dd[COMPOUND_IDENTIFIER_COLUMN])
        cid_data_idx = {}
        for i, cid in enumerate(cid_data):
            cid_data_idx[cid] = i
        new_idxs = []
        for cid in cid_mapping:
            if cid not in cid_data_idx:
                new_idxs += [""]
            else:
                new_idxs += [cid_data_idx[cid]]
        orig_idxs = list(dm[MAPPING_ORIGINAL_COLUMN])
        with open(self.mapping_file, "w") as f:  
            writer = csv.writer(f, delimiter=",")
            writer.writerow([MAPPING_ORIGINAL_COLUMN, MAPPING_DEDUPE_COLUMN, COMPOUND_IDENTIFIER_COLUMN])
            for o, u, c in zip(orig_idxs, new_idxs, cid_mapping):
                writer.writerow([o, u, c])


    def check_smiles(self):
        self._get_input_schema()
        input_smiles_column = self.input_schema_dict["smiles_column"]
        di = pd.read_csv(self.input_file)
        dd = pd.read_csv(self.data_file)
        ismi = list(di[input_smiles_column])
        dsmi = list(dd[SMILES_COLUMN])
        mapping = pd.read_csv(self.mapping_file)
        discrepancies = 0
        for oidx, uidx, cid in mapping.values:
            if str(oidx) == "nan" or str(uidx) == "nan":
                continue
            oidx = int(oidx)
            uidx = int(uidx)
            omol = Chem.MolFromSmiles(ismi[oidx])
            umol = Chem.MolFromSmiles(dsmi[uidx])
            ofp = Chem.RDKFingerprint(omol)
            ufp = Chem.RDKFingerprint(umol)
            sim = DataStructs.FingerprintSimilarity(ofp, ufp)
            if sim < 0.6:
                try:
                    omol = standardise.run(omol)
                except:
                    continue
                ofp = Chem.RDKFingerprint(omol)
                sim = DataStructs.FingerprintSimilarity(ofp, ufp)
                if sim < 0.6:
                    print("Low similarity", sim, cid, ismi[oidx], dsmi[uidx])
                    discrepancies += 1
        assert discrepancies < mapping.shape[0] * 0.25

    def check_activity(self):
        self._get_input_schema()
        input_values_column = self.input_schema_dict["values_column"]
        if input_values_column is None:
            return
        di = pd.read_csv(self.input_file)
        dd = pd.read_csv(self.data_file)
        ival = list(di[input_values_column])
        dval = list(dd[VALUES_COLUMN])
        mapping = pd.read_csv(self.mapping_file)
        for oidx, uidx, cid in mapping.values:
            if str(oidx) == "nan" or str(uidx) == "nan":
                continue
            oidx = int(oidx)
            uidx = int(uidx)
            difference = ival[oidx] - dval[uidx]
            if difference > 0.01:
                print("High activity difference", difference, cid, oidx, uidx)

    def run(self):
        self.remap()
        self.check_smiles()
        self.check_activity()
