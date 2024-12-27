import os
import pandas as pd
import json
import numpy as np
import collections
from rdkit import Chem

from .schema import InputSchema

from zairabase.vars import DATA_SUBFOLDER
from zairabase.vars import ERSILIA_HUB_DEFAULT_MODELS, DEFAULT_ESTIMATORS

from zairabase.vars import COMPOUNDS_FILENAME, VALUES_FILENAME, MAPPING_FILENAME, INPUT_SCHEMA_FILENAME
from zairabase.vars import (
    MAPPING_ORIGINAL_COLUMN,
    MAPPING_DEDUPE_COLUMN,
    COMPOUND_IDENTIFIER_COLUMN,
    USER_COMPOUND_IDENTIFIER_COLUMN,
    SMILES_COLUMN,
    DATE_COLUMN,
    QUALIFIER_COLUMN,
    VALUES_COLUMN,
    GROUP_COLUMN,
    PARAMETERS_FILE
)


class ParametersFile(object):
    def __init__(self, path=None, full_path=None, passed_params=None):
        if passed_params is not None:
            self.params = dict(
                (k, v) for k, v in passed_params.items() if v is not None
            )
        else:
            self.params = {}
        if path is None and full_path is None:
            self.filename = None
        else:
            if full_path is None:
                self.filename = os.path.join(path, PARAMETERS_FILE)
            else:
                self.filename = full_path
            self.params = self.load()

    def load(self):
        if self.filename is not None:
            with open(self.filename, "r") as f:
                data = json.load(f)
        else:
            data = {}
        for k, v in self.params.items():
            data[k] = v
        if "credibility_range" not in data:
            data["credibility_range"] = {"min": None, "max": None}
        if "threshold" not in data:
            if "thresholds" in data:
                pass
            else:
                data["thresholds"] = {
                    "expert_1": None
                }
        else:
            data["thresholds"] = {
                "expert_1": data["threshold"]
            }
            del data["threshold"]
        if "direction" not in data:
            data["direction"] = "high"
        if "ersilia_hub" not in data:
            data["ersilia_hub"] = ERSILIA_HUB_DEFAULT_MODELS
        if "estimators" not in data:
            data["estimators"] = DEFAULT_ESTIMATORS
        return data


class SingleFile(InputSchema):
    def __init__(self, input_file, params):
        InputSchema.__init__(self, input_file)
        self.params = params
        self.df = pd.read_csv(input_file)

    def _make_identifiers(self):
        all_smiles = list(
            set(
                list(self.df[self.df[self.smiles_column].notnull()][self.smiles_column])
            )
        )
        smiles2identifier = {}
        n = len(str(len(all_smiles)))
        for i, smi in enumerate(all_smiles):
            identifier = "CID{0}".format(str(i).zfill(n))
            smiles2identifier[smi] = identifier
        identifiers = []
        for smi in list(self.df[self.smiles_column]):
            if smi in smiles2identifier:
                identifiers += [smiles2identifier[smi]]
            else:
                identifiers += [None]
        return identifiers

    def _impute_qualifiers(self):
        qualifiers = []
        for qual in list(self.df[self.qualifier_column]):
            qual = str(qual).lower()
            if qual == "" or qual == "nan" or qual == "none":
                qualifiers += ["="]
            else:
                qualifiers += [qual]
        return qualifiers

    def normalize_dataframe(self):
        resolved_columns = self.resolve_columns()
        self.identifier_column = resolved_columns["identifier_column"]
        self.smiles_column = resolved_columns["smiles_column"]
        self.qualifier_column = resolved_columns["qualifier_column"]
        self.values_column = resolved_columns["values_column"]
        self.date_column = resolved_columns["date_column"]
        self.group_column = resolved_columns["group_column"]
        identifiers = self._make_identifiers()
        if self.identifier_column is None:
            df = pd.DataFrame({COMPOUND_IDENTIFIER_COLUMN: identifiers})
        else:
            df = pd.DataFrame({COMPOUND_IDENTIFIER_COLUMN: identifiers, USER_COMPOUND_IDENTIFIER_COLUMN: self.df[self.identifier_column]})
        df[SMILES_COLUMN] = self.df[self.smiles_column]

        if self.qualifier_column is None:
            qualifiers = ["="] * self.df.shape[0]
        else:
            qualifiers = self._impute_qualifiers()
        df[QUALIFIER_COLUMN] = qualifiers

        df[VALUES_COLUMN] = self.df[self.values_column]

        if self.date_column:
            df[DATE_COLUMN] = self.df[self.date_column]
        else:
            df[DATE_COLUMN] = None

        if self.group_column:
            df[GROUP_COLUMN] = self.df[self.group_column]
        else:
            df[GROUP_COLUMN] = None
        assert df.shape[0] == self.df.shape[0]
        return df

    def dedupe(self, df, path):
        mapping = collections.defaultdict(list)
        cid2smiles = {}
        for i, r in enumerate(df[[COMPOUND_IDENTIFIER_COLUMN, SMILES_COLUMN]].values):
            cid = r[0]
            smi = r[1]
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mapping[cid] += [i]
            cid2smiles[cid] = smi
        unique_cids = sorted(set(mapping.keys()))
        unique_cids_idx = dict((k, i) for i, k in enumerate(unique_cids))
        mapping = dict((x, k) for k, v in mapping.items() for x in v)
        R = []
        for i in range(df.shape[0]):
            if i in mapping:
                cid = mapping[i]
                R += [[i, unique_cids_idx[cid], cid]]
            else:
                R += [[i, None, None]]
        dfm = pd.DataFrame(
            R,
            columns=[
                MAPPING_ORIGINAL_COLUMN,
                MAPPING_DEDUPE_COLUMN,
                COMPOUND_IDENTIFIER_COLUMN,
            ],
        )
        if USER_COMPOUND_IDENTIFIER_COLUMN in df.columns:
            dfm[USER_COMPOUND_IDENTIFIER_COLUMN]=df[USER_COMPOUND_IDENTIFIER_COLUMN]
        dfm.to_csv(os.path.join(path, MAPPING_FILENAME), index=False)
        R = []
        for cid in unique_cids:
            R += [[cid, cid2smiles[cid]]]
        dfc = pd.DataFrame(R, columns=[COMPOUND_IDENTIFIER_COLUMN, SMILES_COLUMN])
        return dfc

    def compounds_table(self, df, path):
        dfc = self.dedupe(df, path)
        return dfc

    def values_table(self, df):
        dfv = pd.DataFrame({COMPOUND_IDENTIFIER_COLUMN: df[COMPOUND_IDENTIFIER_COLUMN]})
        dfv[QUALIFIER_COLUMN] = df[QUALIFIER_COLUMN]
        dfv[VALUES_COLUMN] = df[VALUES_COLUMN]
        dedupe = collections.defaultdict(list)
        for r in dfv[
            [
                COMPOUND_IDENTIFIER_COLUMN,
                QUALIFIER_COLUMN,
                VALUES_COLUMN,
            ]
        ].values:
            dedupe[r[0]] += [(r[1], r[2])]
        R = []
        for k, v in dedupe.items():
            v = np.median([x[1] for x in v])
            R += [[k, "=", v]]
        dfv = pd.DataFrame(
            R,
            columns=[
                COMPOUND_IDENTIFIER_COLUMN,
                QUALIFIER_COLUMN,
                VALUES_COLUMN,
            ],
        )
        return dfv

    def input_schema(self):
        sc = {
            "input_file": self.input_file,
            "identifier_column": self.identifier_column,
            "smiles_column": self.smiles_column,
            "qualifier_column": self.qualifier_column,
            "values_column": self.values_column,
            "date_column": self.date_column,
            "group_column": self.group_column,
        }
        return sc

    def process(self):
        path = os.path.join(self.get_output_dir(), DATA_SUBFOLDER)
        df = self.normalize_dataframe()
        dfc = self.compounds_table(df, path)
        dfc.to_csv(os.path.join(path, COMPOUNDS_FILENAME), index=False)
        dfv = self.values_table(df)
        dfv.to_csv(os.path.join(path, VALUES_FILENAME), index=False)
        schema = self.input_schema()
        with open(os.path.join(path, INPUT_SCHEMA_FILENAME), "w") as f:
            json.dump(schema, f, indent=4)


class SingleFileForPrediction(SingleFile):
    def __init__(self, input_file, params):
        SingleFile.__init__(self, input_file, params)
        self.trained_path = self.get_trained_dir()

    def get_trained_values_column(self):
        with open(
            os.path.join(self.trained_path, DATA_SUBFOLDER, INPUT_SCHEMA_FILENAME), "r"
        ) as f:
            return json.load(f)["values_column"]

    def normalize_dataframe(self):
        resolved_columns = self.resolve_columns()
        print(resolved_columns)
        self.identifier_column = resolved_columns["identifier_column"]
        self.smiles_column = resolved_columns["smiles_column"]
        identifiers = self._make_identifiers()
        if self.identifier_column is None:
            df = pd.DataFrame({COMPOUND_IDENTIFIER_COLUMN: identifiers})
        else:
            identifiers_skip = list(self.df[self.identifier_column])
            df = pd.DataFrame(
                {
                    COMPOUND_IDENTIFIER_COLUMN: identifiers,
                    COMPOUND_IDENTIFIER_COLUMN + "_skip": identifiers_skip,
                }
            )
        self.qualifier_column = resolved_columns["qualifier_column"]
        self.values_column = resolved_columns["values_column"]
        if self.values_column is not None and self.params is not None:
            trained_values_column = self.get_trained_values_column()
            if self.values_column != trained_values_column:
                self.logger.warning(
                    "Inconsistent values column, {0} vs {1}".format(
                        self.values_column, trained_values_column
                    )
                )
        df = pd.DataFrame({COMPOUND_IDENTIFIER_COLUMN: identifiers})
        df[SMILES_COLUMN] = self.df[self.smiles_column]
        assert df.shape[0] == self.df.shape[0]

        if self.values_column is not None and self.params is not None:
            if self.qualifier_column is None:
                qualifiers = ["="] * self.df.shape[0]
            else:
                qualifiers = self._impute_qualifiers()
            df[QUALIFIER_COLUMN] = qualifiers
            df[VALUES_COLUMN] = self.df[self.values_column]
            self.has_tasks = True
        else:
            self.has_tasks = False

        return df

    def input_schema(self):
        sc = {
            "input_file": self.input_file,
            "identifier_column": self.identifier_column,
            "smiles_column": self.smiles_column,
            "qualifier_column": self.qualifier_column,
            "values_column": self.values_column,
        }
        return sc

    def process(self):
        path = os.path.join(self.get_output_dir(), DATA_SUBFOLDER)
        df = self.normalize_dataframe()
        dfc = self.dedupe(df, path)
        dfc.to_csv(os.path.join(path, COMPOUNDS_FILENAME), index=False)
        if self.has_tasks:
            dfv = self.values_table(df)
            dfv.to_csv(os.path.join(path, VALUES_FILENAME), index=False)
        schema = self.input_schema()
        with open(os.path.join(path, INPUT_SCHEMA_FILENAME), "w") as f:
            json.dump(schema, f, indent=4)
