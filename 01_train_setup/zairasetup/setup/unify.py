import os
import pandas as pd
from . import (
    STANDARD_COMPOUNDS_FILENAME,
    DATA0_FILENAME,
    DATA1_FILENAME,
    FOLDS_FILENAME,
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
    
    def get_standard_smiles(self):
        return pd.read_csv(os.path.join(self.path, STANDARD_COMPOUNDS_FILENAME))
    
    def get_folds(self):
        return pd.read_csv(os.path.join(self.path, FOLDS_FILENAME))
    
    def run(self):
        df1 = self.get_input_file()
        df2 = self.get_standard_smiles()
        df3 = self.get_folds()
        df1 = df1.drop(columns=[SMILES_COLUMN])
        df2 = df2.drop(columns=[SMILES_COLUMN])
        df_ = pd.merge(df1,df2, on=[COMPOUND_IDENTIFIER_COLUMN], how="inner")
        df = pd.merge(df_, df3, on=[COMPOUND_IDENTIFIER_COLUMN, STANDARD_SMILES_COLUMN])
        df.to_csv(os.path.join(self.path, DATA1_FILENAME), index=False)