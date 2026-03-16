import glob, os, shutil
import pandas as pd

from zairachem.base.vars import (
  DESCRIPTORS_SUBFOLDER,
  POOL_SUBFOLDER,
  ESTIMATORS_SUBFOLDER,
  REPORT_SUBFOLDER,
  OUTPUT_FILENAME,
  OUTPUT_TABLE_FILENAME,
  PERFORMANCE_TABLE_FILENAME,
  OUTPUT_XLSX_FILENAME,
  DATA_SUBFOLDER,
  DATA_FILENAME,
  RAW_INPUT_FILENAME,
  RESULTS_MAPPED_FILENAME,
  RESULTS_UNMAPPED_FILENAME,
)
from zairachem.base import ZairaBase
from zairachem.base.utils.pipeline import PipelineStep

CLEAN_TARGET_ALL = "all"
CLEAN_TARGET_MODEL = "model"
CLEAN_TARGET_PREDICT = "predict"
VALID_CLEAN_TARGETS = [CLEAN_TARGET_ALL, CLEAN_TARGET_MODEL, CLEAN_TARGET_PREDICT]


class Cleaner(ZairaBase):
  def __init__(self, path, target=CLEAN_TARGET_ALL):
    ZairaBase.__init__(self)
    if path is None:
      self.path = self.get_output_dir()
    else:
      self.path = path
    self.output_dir = os.path.abspath(self.path)
    self.target = target
    self._is_predict = self.is_predict()
    if self._is_predict:
      self.trained_dir = self.get_trained_dir()
    else:
      self.trained_dir = self.path
    assert os.path.exists(self.output_dir)

  def _clean_descriptors_by_subfolder(self, path, subfolder):
    full_path = os.path.join(path, subfolder)
    if not os.path.exists(full_path):
      return
    for d in os.listdir(full_path):
      if d.startswith("fp2sim"):
        continue
      if os.path.isdir(os.path.join(full_path, d)):
        self._clean_descriptors_by_subfolder(full_path, d)
      else:
        if d.endswith(".h5"):
          os.remove(os.path.join(full_path, d))

  def _clean_descriptors(self, path):
    self._clean_descriptors_by_subfolder(path, DESCRIPTORS_SUBFOLDER)

  def run(self):
    if self.target == CLEAN_TARGET_ALL:
      self.logger.debug("Cleaning descriptors from output directory")
      self._clean_descriptors(self.output_dir)
      if self._is_predict and self.trained_dir != self.output_dir:
        self.logger.debug("Cleaning descriptors from model directory")
        self._clean_descriptors(self.trained_dir)
    elif self.target == CLEAN_TARGET_MODEL:
      if self._is_predict:
        self.logger.debug("Cleaning descriptors from model directory only")
        self._clean_descriptors(self.trained_dir)
      else:
        self.logger.debug("Cleaning descriptors from model directory (fit mode)")
        self._clean_descriptors(self.output_dir)
    elif self.target == CLEAN_TARGET_PREDICT:
      if self._is_predict:
        self.logger.debug("Cleaning descriptors from predict directory only")
        self._clean_descriptors(self.output_dir)
      else:
        self.logger.warning("Cannot clean predict directory during fit - skipping")


class Flusher(ZairaBase):
  def __init__(self, path, target=CLEAN_TARGET_ALL):
    ZairaBase.__init__(self)
    if path is None:
      self.path = self.get_output_dir()
    else:
      self.path = path
    self.output_dir = os.path.abspath(self.path)
    self.target = target
    self._is_predict = self.is_predict()
    if self._is_predict:
      self.trained_dir = self.get_trained_dir()
    else:
      self.trained_dir = self.path
    assert os.path.exists(self.output_dir)

  def _remover(self, path):
    if not os.path.exists(path):
      return
    rm_dirs = []
    rm_files = []
    for root, dirs, files in os.walk(path):
      for filename in files:
        if filename.endswith(".json") or filename.endswith(".csv"):
          continue
        rm_files += [os.path.join(root, filename)]
      for dirname in dirs:
        if dirname.startswith("flaml"):
          rm_dirs += [os.path.join(root, dirname)]
        if dirname.startswith("kerastuner"):
          rm_dirs += [os.path.join(root, dirname)]
    for f in rm_files:
      if os.path.exists(f):
        os.remove(f)
    for d in rm_dirs:
      if os.path.exists(d):
        shutil.rmtree(d)

  def _flush(self, path):
    self.logger.debug("Removing files from descriptors folder in {0}".format(path))
    self._remover(os.path.join(path, DESCRIPTORS_SUBFOLDER))
    self.logger.debug("Removing files from estimators folder in {0}".format(path))
    self._remover(os.path.join(path, ESTIMATORS_SUBFOLDER))

  def run(self):
    if self.target == CLEAN_TARGET_ALL:
      self._flush(self.output_dir)
      if self._is_predict and self.trained_dir != self.output_dir:
        self._flush(self.trained_dir)
    elif self.target == CLEAN_TARGET_MODEL:
      if self._is_predict:
        self.logger.debug("Flushing model directory only")
        self._flush(self.trained_dir)
      else:
        self.logger.debug("Flushing model directory (fit mode)")
        self._flush(self.output_dir)
    elif self.target == CLEAN_TARGET_PREDICT:
      if self._is_predict:
        self.logger.debug("Flushing predict directory only")
        self._flush(self.output_dir)
      else:
        self.logger.warning("Cannot flush predict directory during fit - skipping")


class Anonymizer(ZairaBase):
  def __init__(self, path, target=CLEAN_TARGET_ALL):
    ZairaBase.__init__(self)
    if path is None:
      self.path = self.get_output_dir()
    else:
      self.path = path
    self.output_dir = os.path.abspath(self.path)
    self.target = target
    self._is_predict = self.is_predict()
    if self._is_predict:
      self.trained_dir = self.get_trained_dir()
    else:
      self.trained_dir = self.path
    self._empty_string = "NA"

  def _remove_file_if_exists(self, file_name):
    if os.path.exists(file_name):
      os.remove(file_name)

  def _replace_sensitive_columns(self, file_name):
    if not os.path.exists(file_name):
      return
    columns = ["input-smiles", "smiles", "inchikey"]
    df = pd.read_csv(file_name)
    df_columns = set(list(df.columns))
    for col in columns:
      if col in df_columns:
        df[col] = self._empty_string
    df.to_csv(file_name, index=False)

  def _remove_sensitive_columns(self, file_name):
    if not os.path.exists(file_name):
      return
    desc_columns = [
      "nHBD",
      "nHBA",
      "cLogP",
      "nHeteroAtoms",
      "RingCount",
      "nRotatableBonds",
      "nAromaticBonds",
      "nAcidicGroup",
      "nBasicGroup",
      "AtomicPolarizability",
      "MolWt",
      "TPSA",
    ]
    prefixes = ["umap-", "pca-", "lolp-", "exact_", "sim_", "train_smiles_"]
    df = pd.read_csv(file_name)
    columns = list(df.columns)
    to_remove = []
    for col in columns:
      if col in desc_columns:
        to_remove += [col]
        continue
      for pref in prefixes:
        if col.startswith(pref):
          to_remove += [col]
    to_remove = list(set(to_remove))
    df.drop(columns=to_remove, inplace=True)
    df.to_csv(file_name, index=False)

  def _replace_first_last_descriptors(self, file_name):
    if not os.path.exists(file_name):
      return
    df = pd.read_csv(file_name)
    df.loc[0:1, df.columns[1:]] = self._empty_string
    df.to_csv(file_name, index=False)

  def _remove_raw_input(self, path):
    file_name = os.path.join(path, RAW_INPUT_FILENAME + ".csv")
    self._remove_file_if_exists(file_name)

  def _remove_output_table_xlsx(self, path):
    file_name = os.path.join(path, OUTPUT_XLSX_FILENAME)
    self._remove_file_if_exists(file_name)

  def _clear_all_sensitive_columns(self, path):
    self._replace_sensitive_columns(os.path.join(path, DATA_SUBFOLDER, DATA_FILENAME))
    self._replace_sensitive_columns(os.path.join(path, OUTPUT_FILENAME))
    self._remove_sensitive_columns(os.path.join(path, OUTPUT_TABLE_FILENAME))
    self._replace_sensitive_columns(os.path.join(path, OUTPUT_TABLE_FILENAME))

  def _clear_descriptors(self, path):
    Cleaner(
      path=path, target=CLEAN_TARGET_PREDICT if path == self.output_dir else CLEAN_TARGET_MODEL
    ).run()
    subfolder = os.path.join(path, DESCRIPTORS_SUBFOLDER)
    if not os.path.exists(subfolder):
      return
    for d in os.listdir(subfolder):
      if d.startswith("fp2sim"):
        self._remove_file_if_exists(os.path.join(subfolder, d))
    for file_path in glob.iglob(subfolder + "/**", recursive=True):
      file_name = file_path.split("/")[-1]
      if file_name == "raw_summary.csv":
        self._replace_first_last_descriptors(file_path)

  def _clear_estimators(self, path):
    subfolder = os.path.join(path, ESTIMATORS_SUBFOLDER).rstrip("/")
    if not os.path.exists(subfolder):
      return
    for file_path in glob.iglob(subfolder + "/**", recursive=True):
      file_name = file_path.split("/")[-1]
      if file_name == RESULTS_MAPPED_FILENAME or file_name == RESULTS_UNMAPPED_FILENAME:
        self._replace_sensitive_columns(file_path)
    manifolds_data = os.path.join(subfolder, "manifolds", "data.csv")
    self._remove_file_if_exists(manifolds_data)

  def _clear_pool(self, path):
    pool_data = os.path.join(path, POOL_SUBFOLDER, "data.csv")
    if os.path.exists(pool_data):
      self._remove_sensitive_columns(pool_data)
    self._replace_sensitive_columns(os.path.join(path, POOL_SUBFOLDER, RESULTS_MAPPED_FILENAME))
    self._replace_sensitive_columns(os.path.join(path, POOL_SUBFOLDER, RESULTS_UNMAPPED_FILENAME))

  def _clear_report(self, path):
    report_folder = os.path.join(path, REPORT_SUBFOLDER)
    if not os.path.exists(report_folder):
      return
    self._remove_file_if_exists(os.path.join(report_folder, "tanimoto-similarity-to-train.png"))
    self._remove_file_if_exists(os.path.join(report_folder, "projection-pca.png"))
    self._remove_file_if_exists(os.path.join(report_folder, "projection-umap.png"))
    self._remove_file_if_exists(os.path.join(report_folder, OUTPUT_TABLE_FILENAME))

  def _anonymize_path(self, path):
    self._remove_raw_input(path)
    self._remove_output_table_xlsx(path)
    self._clear_all_sensitive_columns(path)
    self._clear_descriptors(path)
    self._clear_estimators(path)
    self._clear_pool(path)
    self._clear_report(path)

  def run(self):
    if self.target == CLEAN_TARGET_ALL:
      self._anonymize_path(self.output_dir)
      if self._is_predict and self.trained_dir != self.output_dir:
        self._anonymize_path(self.trained_dir)
    elif self.target == CLEAN_TARGET_MODEL:
      if self._is_predict:
        self.logger.debug("Anonymizing model directory only")
        self._anonymize_path(self.trained_dir)
      else:
        self.logger.debug("Anonymizing model directory (fit mode)")
        self._anonymize_path(self.output_dir)
    elif self.target == CLEAN_TARGET_PREDICT:
      if self._is_predict:
        self.logger.debug("Anonymizing predict directory only")
        self._anonymize_path(self.output_dir)
      else:
        self.logger.warning("Cannot anonymize predict directory during fit - skipping")


class OutputToExcel(ZairaBase):
  def __init__(self, path, clean=False, flush=False):
    ZairaBase.__init__(self)
    if path is None:
      self.path = self.get_output_dir()
    else:
      self.path = path
    self.output_csv = os.path.join(self.path, OUTPUT_TABLE_FILENAME)
    self.performance_csv = os.path.join(self.path, PERFORMANCE_TABLE_FILENAME)
    self.output_xlsx = os.path.join(self.path, OUTPUT_XLSX_FILENAME)

  def run(self):
    if not os.path.exists(self.output_csv):
      self.logger.warning(f"Output table not found: {self.output_csv}")
      return
    df_o = pd.read_csv(self.output_csv)
    if not os.path.exists(self.performance_csv):
      df_p = None
    else:
      df_p = pd.read_csv(self.performance_csv)
    with pd.ExcelWriter(self.output_xlsx, mode="w", engine="openpyxl") as writer:
      df_o.to_excel(writer, sheet_name="Output", index=False)
      if df_p is not None:
        df_p.to_excel(writer, sheet_name="Performance", index=False)


class Finisher(ZairaBase):
  def __init__(
    self, path, clean=False, flush=False, anonymize=False, clean_target=CLEAN_TARGET_ALL
  ):
    ZairaBase.__init__(self)
    if path is None:
      self.path = self.get_output_dir()
    else:
      self.path = path
    self.clean = clean
    self.flush = flush
    self.anonymize = anonymize
    self.clean_target = clean_target if clean_target in VALID_CLEAN_TARGETS else CLEAN_TARGET_ALL

  def _clean_descriptors(self):
    Cleaner(path=self.path, target=self.clean_target).run()

  def _flush(self):
    Flusher(path=self.path, target=self.clean_target).run()

  def _anonymize(self):
    Anonymizer(path=self.path, target=self.clean_target).run()

  def _predictions_file(self):
    src = os.path.join(self.path, POOL_SUBFOLDER, RESULTS_MAPPED_FILENAME)
    dst = os.path.join(self.path, OUTPUT_FILENAME)
    if os.path.exists(src):
      shutil.copy(src, dst)

  def _output_table_file(self):
    src = os.path.join(self.path, REPORT_SUBFOLDER, OUTPUT_TABLE_FILENAME)
    dst = os.path.join(self.path, OUTPUT_TABLE_FILENAME)
    if os.path.exists(src):
      shutil.copy(src, dst)

  def _performance_table_file(self):
    filename = os.path.join(self.path, REPORT_SUBFOLDER, PERFORMANCE_TABLE_FILENAME)
    if not os.path.exists(filename):
      return
    shutil.copy(
      filename,
      os.path.join(self.path, PERFORMANCE_TABLE_FILENAME),
    )

  def _to_excel(self):
    OutputToExcel(path=self.path).run()

  def run_all(self):
    self.logger.debug("Finishing")
    self._predictions_file()
    self._output_table_file()
    self._performance_table_file()
    self._to_excel()
    if self.clean:
      self.logger.info(f"Cleaning with target: {self.clean_target}")
      self._clean_descriptors()
    if self.flush:
      self.logger.info(f"Flushing with target: {self.clean_target}")
      self._flush()
    if self.anonymize:
      self.logger.info(f"Anonymizing with target: {self.clean_target}")
      self._anonymize()

  def run(self):
    step = PipelineStep("finish", self.path)
    if not step.is_done():
      self.run_all()
      step.update()
    else:
      self.logger.warning(
        "[yellow]Finishing setup for requested inferece is already done. Skippign this step![/]"
      )
    self.logger.info("[green]All zairachem successfully completed![/]")
