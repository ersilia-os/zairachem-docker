import os
import pandas as pd
import json
from zairabase.vars import DATA_FILENAME
from . import DATA0_FILENAME, STANDARD_COMPOUNDS_FILENAME, FOLDS_FILENAME,TASKS_FILENAME, SCHEMA_MERGE_FILENAME
from . import COMPOUND_IDENTIFIER_COLUMN, SMILES_COLUMN, STANDARD_SMILES_COLUMN

class DataMerger(object):
    def __init__(self, path):
        self.path = path

    def get_input_file(self):
        return pd.read_csv(os.path.join(self.path, DATA0_FILENAME))
    
    def get_standard_smiles(self):
        return pd.read_csv(os.path.join(self.path, STANDARD_COMPOUNDS_FILENAME))
    
    def get_folds(self):
        return pd.read_csv(os.path.join(self.path, FOLDS_FILENAME))
    
    def get_tasks(self):
        return pd.read_csv(os.path.join(self.path, TASKS_FILENAME))
    
    def run(self):
        df1 = self.get_standard_smiles()
        df2 = self.get_folds()
        df = pd.merge(df1, df2, on=[COMPOUND_IDENTIFIER_COLUMN, SMILES_COLUMN, STANDARD_SMILES_COLUMN])
        df = df.drop(columns=[SMILES_COLUMN])
        df = df.rename(columns={STANDARD_SMILES_COLUMN:SMILES_COLUMN})
        df_tsk = self.get_tasks()
        df = df.merge(df_tsk, on=COMPOUND_IDENTIFIER_COLUMN)
        schema = {
            "compounds": list(df[[COMPOUND_IDENTIFIER_COLUMN, SMILES_COLUMN]].columns),
            "folds": list(df[[c for c in list(df.columns) if "fld" in c]].columns),
            "tasks": [
                c for c in list(df_tsk.columns) if "reg_" in c or "clf_" in c
            ],
        }

        df.to_csv(os.path.join(self.path, DATA_FILENAME), index=False)
        with open(os.path.join(self.path, SCHEMA_MERGE_FILENAME), "w") as f:
            json.dump(schema, f, indent=4)


class DataMergerForPrediction(object):
    def __init__(self, path):
        self.path = path

    def run(self, has_tasks):
        if not has_tasks:
            df_cpd = pd.read_csv(os.path.join(self.path, COMPOUNDS_FILENAME))[
                [COMPOUND_IDENTIFIER_COLUMN, SMILES_COLUMN]
            ]
            schema = {"compounds": list(df_cpd.columns)}
            with open(os.path.join(self.path, SCHEMA_MERGE_FILENAME), "w") as f:
                json.dump(schema, f, indent=4)
            df = df_cpd
            df.to_csv(os.path.join(self.path, DATA_FILENAME), index=False)
        else:
            df_cpd = pd.read_csv(os.path.join(self.path, COMPOUNDS_FILENAME))[
                [COMPOUND_IDENTIFIER_COLUMN, SMILES_COLUMN]
            ]
            df_tsk = pd.read_csv(os.path.join(self.path, TASKS_FILENAME))
            df_tsk = df_tsk[
                [
                    c
                    for c in list(df_tsk.columns)
                    if "reg_" in c or "clf_" in c or c == COMPOUND_IDENTIFIER_COLUMN
                ]
            ]
            df = df_cpd.merge(df_tsk, on="compound_id")
            schema = {
                "compounds": list(df_cpd.columns),
                "tasks": [
                    c for c in list(df_tsk.columns) if c != COMPOUND_IDENTIFIER_COLUMN
                ],
            }
            df.to_csv(os.path.join(self.path, DATA_FILENAME), index=False)
            with open(os.path.join(self.path, SCHEMA_MERGE_FILENAME), "w") as f:
                json.dump(schema, f, indent=4)
