import glob, os, shutil
import pandas as pd

from zairachem.base.vars import (
  DESCRIPTORS_SUBFOLDER,
  POOL_SUBFOLDER,
  ESTIMATORS_SUBFOLDER,
  MODEL_SUBFOLDER,
  FOLDS_SUBFOLDER,
  REPORT_SUBFOLDER,
  RESULTS_SUBFOLDER,
  OUTPUT_FILENAME,
  OUTPUT_TABLE_FILENAME,
  PERFORMANCE_TABLE_FILENAME,
  DATA_SUBFOLDER,
  DATA_FILENAME,
  RAW_INPUT_FILENAME,
  ERSILIA_DATA_FILENAME,
  SMILES_LIST_FILENAME,
  RESULTS_MAPPED_FILENAME,
  RESULTS_UNMAPPED_FILENAME,
)
from zairachem.base import ZairaBase
from zairachem.base.utils.pipeline import PipelineStep

# Anonymization scope constants. Cleaner and Anonymizer still use these internally; the public
# clean/flush flags and the --clean-target option were removed (Finisher is anonymize-only).
CLEAN_TARGET_ALL = "all"
CLEAN_TARGET_MODEL = "model"
CLEAN_TARGET_PREDICT = "predict"


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
      entry = os.path.join(full_path, d)
      if os.path.isdir(entry):
        if d.endswith("_chunks"):
          shutil.rmtree(entry)
        else:
          self._clean_descriptors_by_subfolder(full_path, d)
      else:
        if d.endswith(".h5"):
          os.remove(entry)

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
    file_name = os.path.join(path, DATA_SUBFOLDER, RAW_INPUT_FILENAME + ".csv")
    self._remove_file_if_exists(file_name)

  def _clear_all_sensitive_columns(self, path):
    self._replace_sensitive_columns(os.path.join(path, DATA_SUBFOLDER, DATA_FILENAME))
    self._replace_sensitive_columns(os.path.join(path, OUTPUT_FILENAME))
    results_table = os.path.join(path, RESULTS_SUBFOLDER, OUTPUT_TABLE_FILENAME)
    self._remove_sensitive_columns(results_table)
    self._replace_sensitive_columns(results_table)

  def _clear_descriptors(self, path):
    # Choose the target that actually cleans THIS path in the current mode. Only the
    # predict directory (in predict mode) uses PREDICT; every other case — including
    # fit, where output_dir IS the model dir — uses MODEL, so descriptors are removed
    # rather than skipped (PREDICT target is a no-op during a fit).
    if path == self.output_dir and self._is_predict:
      target = CLEAN_TARGET_PREDICT
    else:
      target = CLEAN_TARGET_MODEL
    Cleaner(path=path, target=target).run()
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
    for stem in ("tanimoto-similarity-to-train", "projection-pca", "projection-umap"):
      self._remove_file_if_exists(os.path.join(report_folder, "png", stem + ".png"))
      self._remove_file_if_exists(os.path.join(report_folder, "pdf", stem + ".pdf"))
    self._remove_file_if_exists(os.path.join(report_folder, OUTPUT_TABLE_FILENAME))

  def _anonymize_path(self, path):
    self._remove_raw_input(path)
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


class Finisher(ZairaBase):
  def __init__(self, path, anonymize=False, keep_intermediate_data=False):
    ZairaBase.__init__(self)
    if path is None:
      self.path = self.get_output_dir()
    else:
      self.path = path
    self.anonymize = anonymize
    self.keep_intermediate_data = keep_intermediate_data

  def _anonymize(self):
    Anonymizer(path=self.path).run()

  @staticmethod
  def _rm(path):
    if os.path.exists(path):
      os.remove(path)

  def _clean_intermediate_data(self):
    """Remove artifacts not needed once the run is finished — keeping only what predict reuses (the
    trained model) and the user-facing deliverables (``results/`` + the report).

    Always (fit and predict): the heavy descriptor matrices (raw/treated ``.h5`` + ``*_chunks`` via
    :class:`Cleaner`), the unused ersilia-format input + bare smiles list, and the per-estimator
    training diagnostics (``evaluation*.json``). A FIT dir keeps its model (estimators
    ``.onnx``/pooler/AD, transformers, pool ``results_unmapped.csv``, ``done_eos.json``,
    ``inputs/data.csv``) and the report. It also keeps the **small** report inputs — the raw input
    copy, the 2-D projection coords + manifest, and the dedup mapping — so the report (and the other
    step subcommands) can be re-run against a finished model without re-training.

    At PREDICT it additionally drops the throwaway ``descriptors/`` + ``model/`` trees: nothing there
    is reused (predict reads the *trained* model, never its own output), and the predictions already
    live in ``results/`` (with per-descriptor columns in the output table). Gated by
    ``keep_intermediate_data``.
    """
    Cleaner(path=self.path).run()
    data_dir = os.path.join(self.path, DATA_SUBFOLDER)
    # NB: raw_input.csv, projections.csv/.json and mapping.csv are deliberately NOT trimmed — the
    # report reads them, and keeping them (all tiny) lets a finished model be re-reported / re-run.
    for fn in (ERSILIA_DATA_FILENAME, SMILES_LIST_FILENAME):
      self._rm(os.path.join(data_dir, fn))
    # Held-out validation sub-runs (--evaluate) are scratch: their per-fold estimators/pool mirror the
    # main model's footprint ~10x. The validation results the report needs (validation_table.csv +
    # holdout_summary.json) already live under report/, so the whole folds/ tree is disposable.
    folds_dir = os.path.join(self.path, FOLDS_SUBFOLDER)
    if os.path.isdir(folds_dir):
      shutil.rmtree(folds_dir)
    # Per-estimator SimpleEvaluator diagnostics — never read by predict or the report (which uses
    # cv_report.json / oof.csv instead).
    for f in glob.glob(
      os.path.join(self.path, ESTIMATORS_SUBFOLDER, "**", "evaluation*.json"), recursive=True
    ):
      self._rm(f)
    if self.is_predict():
      # descriptors/ + model/ are throwaway at predict. Keeps results/ + report/ + metadata/ (and
      # inputs/data.csv + input_schema.json for provenance). NB: these are now sibling top-level dirs
      # (not a single pipeline/ tree), so each is removed explicitly.
      for subfolder in (DESCRIPTORS_SUBFOLDER, MODEL_SUBFOLDER):
        throwaway_dir = os.path.join(self.path, subfolder)
        if os.path.isdir(throwaway_dir):
          shutil.rmtree(throwaway_dir)

  def _predictions_file(self):
    src = os.path.join(self.path, POOL_SUBFOLDER, RESULTS_MAPPED_FILENAME)
    dst = os.path.join(self.path, OUTPUT_FILENAME)
    if os.path.exists(src):
      shutil.copy(src, dst)

  def _output_table_file(self):
    src = os.path.join(self.path, REPORT_SUBFOLDER, OUTPUT_TABLE_FILENAME)
    dst = os.path.join(self.path, RESULTS_SUBFOLDER, OUTPUT_TABLE_FILENAME)
    if os.path.exists(src):
      shutil.copy(src, dst)

  def _performance_table_file(self):
    filename = os.path.join(self.path, REPORT_SUBFOLDER, PERFORMANCE_TABLE_FILENAME)
    if not os.path.exists(filename):
      return
    shutil.copy(
      filename,
      os.path.join(self.path, RESULTS_SUBFOLDER, PERFORMANCE_TABLE_FILENAME),
    )

  def run_all(self):
    self.logger.debug("Finishing")
    self._predictions_file()
    self._output_table_file()
    self._performance_table_file()
    if self.anonymize:
      self.logger.info("Anonymizing outputs")
      self._anonymize()
    if self.keep_intermediate_data:
      self.logger.info("Keeping intermediate data (--keep-intermediate-data)")
    else:
      self.logger.info("Cleaning intermediate data (pass --keep-intermediate-data to keep it)")
      self._clean_intermediate_data()

  def run(self):
    step = PipelineStep("finish", self.path)
    if not step.is_done():
      self.run_all()
      step.update()
    else:
      self.logger.info("Finishing already done — skipping.")
    self.logger.info("[green]All zairachem successfully completed![/]")
