import os
import pandas as pd
from . import (
    STANDARD_COMPOUNDS_FILENAME,
    DATA0_FILENAME,
    DATA1_FILENAME,
    COMPOUND_IDENTIFIER_COLUMN,
    SMILES_COLUMN,
    STANDARD_SMILES_COLUMN,
    ASSAY_IDENTIFIER_COLUMN,
    QUALIFIER_COLUMN,
    VALUES_COLUMN
)

class UnifyData(object):
    def __init__(self, path):
        self.path = path

    def get_input_file(self):
        return pd.read_csv(os.path.join(self.path, DATA0_FILENAME))
    
    def get_processed_file(self):
        return pd.read_csv(os.path.join(self.path, STANDARD_COMPOUNDS_FILENAME))
    
    def run(self):
        df1 = self.get_input_file()
        df2 = self.get_processed_file()
        df1 = df1.drop(columns=[SMILES_COLUMN])
        df2 = df2.drop(columns=[SMILES_COLUMN]) #we rename the standard smiles to smiles for downstream easiness
        df2=df2.rename(columns={STANDARD_SMILES_COLUMN:SMILES_COLUMN})
        df = pd.merge(df1,df2, on=[COMPOUND_IDENTIFIER_COLUMN], how="inner")
        df = df[[COMPOUND_IDENTIFIER_COLUMN, SMILES_COLUMN, VALUES_COLUMN, ASSAY_IDENTIFIER_COLUMN, QUALIFIER_COLUMN]]
        df.to_csv(os.path.join(self.path, DATA1_FILENAME), index=False)