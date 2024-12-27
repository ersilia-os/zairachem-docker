import os
import pandas as pd

from zairabase.vars import SMILES_COLUMN, COMPOUNDS_FILENAME, COMPOUND_IDENTIFIER_COLUMN, STANDARD_SMILES_COLUMN
from zairasetup.tools.chembl_structure.standardizer import standardize_molblock_from_smiles


class ChemblStandardize(object):
    """Molecule standardiser from ChEMBL"""

    def __init__(self, outdir):
        self.outdir = outdir
        self.input_file = self.get_input_file()
        self.output_file = self.get_output_file()

    def get_output_file(self):
        return os.path.join(self.outdir, COMPOUNDS_FILENAME.split(".")[0] + "_std.csv")

    def get_input_file(self):
        return os.path.join(self.outdir, COMPOUNDS_FILENAME)
    
    def run(self):
        df = pd.read_csv(self.input_file)
        R = []
        for r in df.values:
            identifier = r[0]
            smi = r[1]
            st_smi = standardize_molblock_from_smiles(smi, get_smiles=True)
            if st_smi is not None:
                R += [[identifier, smi, st_smi]]
        df = pd.DataFrame(R, columns = [COMPOUND_IDENTIFIER_COLUMN, SMILES_COLUMN, STANDARD_SMILES_COLUMN])
        df.to_csv(self.output_file, index=False)