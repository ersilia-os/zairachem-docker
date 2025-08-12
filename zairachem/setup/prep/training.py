import json, os, shutil

from zairachem.base import ZairaBase, create_session_symlink
from zairachem.setup.prep import (
  ParametersFile,
  SingleFile,
  ChemblStandardize,
  FoldEnsemble,
  SingleTasks,
  DataMerger,
  SetupCleaner,
  SetupChecker,
  PipelineStep,
  SessionFile,
)
from zairachem.base.vars import (
  PARAMETERS_FILE,
  RAW_INPUT_FILENAME,
  DATA_SUBFOLDER,
  DESCRIPTORS_SUBFOLDER,
  ESTIMATORS_SUBFOLDER,
  POOL_SUBFOLDER,
  LITE_SUBFOLDER,
  INTERPRETABILITY_SUBFOLDER,
  APPLICABILITY_SUBFOLDER,
  REPORT_SUBFOLDER,
  DISTILL_SUBFOLDER,
  OUTPUT_FILENAME,
)


from rdkit import RDLogger

RDLogger.logger().setLevel(RDLogger.CRITICAL)


class TrainSetup(object):
  def __init__(
    self,
    input_file,
    output_dir,
    time_budget,
    task,
    threshold,
    direction,
    parameters,
  ):
    if output_dir is None:
      output_dir = input_file.split(".")[0]
    if task is None and threshold is not None:
      task = "classification"
    else:
      if task is None:
        task = "regression"
    assert task in ["regression", "classification"]
    passed_params = {
      "time_budget": time_budget,
      "threshold": threshold,
      "direction": direction,
      "task": task,
    }
    self.params = self._load_params(parameters, passed_params)
    self.input_file = os.path.abspath(input_file)
    self.output_dir = os.path.abspath(output_dir)
    self.time_budget = time_budget  # TODO

  def _copy_input_file(self):
    extension = self.input_file.split(".")[-1]
    shutil.copy(
      self.input_file,
      os.path.join(self.output_dir, RAW_INPUT_FILENAME + "." + extension),
    )

  def _load_params(self, params, passed_params):
    return ParametersFile(full_path=params, passed_params=passed_params).load()

  def _save_params(self):
    with open(os.path.join(self.output_dir, DATA_SUBFOLDER, PARAMETERS_FILE), "w") as f:
      json.dump(self.params, f, indent=4)

  def _make_output_dir(self):
    if os.path.exists(self.output_dir):
      shutil.rmtree(self.output_dir)
    os.makedirs(self.output_dir)

  def _open_session(self):
    sf = SessionFile(self.output_dir)
    sf.open_session(mode="fit", output_dir=self.output_dir, model_dir=self.output_dir)
    create_session_symlink(self.output_dir)

  def _make_subfolder(self, name):
    os.makedirs(os.path.join(self.output_dir, name))

  def _make_subfolders(self):
    self._make_subfolder(DATA_SUBFOLDER)
    self._make_subfolder(DESCRIPTORS_SUBFOLDER)
    self._make_subfolder(ESTIMATORS_SUBFOLDER)
    self._make_subfolder(POOL_SUBFOLDER)
    self._make_subfolder(LITE_SUBFOLDER)
    self._make_subfolder(INTERPRETABILITY_SUBFOLDER)
    self._make_subfolder(APPLICABILITY_SUBFOLDER)
    self._make_subfolder(REPORT_SUBFOLDER)
    self._make_subfolder(DISTILL_SUBFOLDER)

  def _initialize(self):
    step = PipelineStep("initialize", self.output_dir)

    if not step.is_done():
      self._make_output_dir()
      self._open_session()
      self._make_subfolders()
      self._save_params()
      self._copy_input_file()
      step.update()

  def _normalize_input(self):
    step = PipelineStep("normalize_input", self.output_dir)
    if not step.is_done():
      f = SingleFile(self.input_file, self.params)
      f.process()
      step.update()

  def _standardise(self):
    step = PipelineStep("standardise_smiles", self.output_dir)
    if not step.is_done():
      std = ChemblStandardize(os.path.join(self.output_dir, DATA_SUBFOLDER))
      std.run()
      step.update()

  def _create_folds(self):
    step = PipelineStep("create_folds", self.output_dir)
    if not step.is_done():
      std = FoldEnsemble(os.path.join(self.output_dir, DATA_SUBFOLDER))
      std.run()
      step.update()

  def _tasks(self):
    step = PipelineStep("tasks", self.output_dir)
    if not step.is_done():
      SingleTasks(os.path.join(self.output_dir, DATA_SUBFOLDER)).run()
      step.update()

  def _merge(self):
    step = PipelineStep("merge", self.output_dir)
    if not step.is_done():
      DataMerger(os.path.join(self.output_dir, DATA_SUBFOLDER)).run()
      step.update()

  def _check(self):
    step = PipelineStep("setup_check", self.output_dir)
    if not step.is_done():
      SetupChecker(self.output_dir).run()
      step.update()

  def _clean(self):
    step = PipelineStep("clean", self.output_dir)
    if not step.is_done():
      SetupCleaner(os.path.join(self.output_dir, DATA_SUBFOLDER)).run()
      step.update()

  def update_elapsed_time(self):
    ZairaBase().update_elapsed_time()

  def is_done(self):
    if os.path.exists(os.path.join(self.output_dir, OUTPUT_FILENAME)):
      return True
    else:
      return False

  def setup(self):
    self._initialize()
    self._normalize_input()
    self._standardise()
    self._create_folds()
    self._tasks()
    self._merge()
    self._check()
    self._clean()
    self.update_elapsed_time()
