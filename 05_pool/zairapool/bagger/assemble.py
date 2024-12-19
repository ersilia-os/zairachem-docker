import os
import joblib
import numpy as np
import pandas as pd

from ..base import BaseOutcomeAssembler

from zairabase.vars import DATA_SUBFOLDER, POOL_SUBFOLDER, RESULTS_UNMAPPED_FILENAME, RESULTS_MAPPED_FILENAME

class BaggerAssembler(BaseOutcomeAssembler):
    def __init__(self, path=None):
        BaseOutcomeAssembler.__init__(self, path=path)

    def _back_to_raw(self, df):
        for c in list(df.columns):
            if "reg" in c:
                transformer = joblib.load(
                    os.path.join(
                        self.trained_path,
                        DATA_SUBFOLDER,
                        "{0}_transformer.joblib".format(c.split("_")[1]), #TODO AMEND WHEN REGRESSION, only one, no pwr, qnt etc
                    )
                )
                trn = np.array(df[c]).reshape(-1, 1)
                raw = transformer.inverse_transform(trn)[:, 0]
                df["reg_raw"] = raw
        return df

    def run(self, df):
        #df = self._back_to_raw(df)
        df_c = self._get_compounds()
        df_y = df
        avail_columns = set(df_y.columns)
        if "compound_id" in avail_columns:
            df = df_c.merge(df_y, how="left", on="compound_id")
        else:
            df = pd.concat([df_c, df_y], axis=1)
        df = df.reset_index(drop=True)
        df.to_csv(
            os.path.join(self.path, POOL_SUBFOLDER, RESULTS_UNMAPPED_FILENAME),
            index=False,
        )
        mappings = self._get_mappings()
        df = self._remap(df, mappings)
        df.to_csv(
            os.path.join(self.path, POOL_SUBFOLDER, RESULTS_MAPPED_FILENAME),
            index=False,
        )
