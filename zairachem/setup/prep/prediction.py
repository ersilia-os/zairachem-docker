import json
import os, shutil
from zairachem.setup.prep import (
  ParametersFile,
  SingleFileForPrediction,
  ChemblStandardize,
  SingleTasksForPrediction,
  DataMergerForPrediction,
  SetupCleaner,
  SetupChecker,
)

from zairachem.base import ZairaBase, create_session_symlink
from zairachem.base.utils.logging import logger
from zairachem.base.utils.console import summary_panel
from zairachem.base.utils.preflight import require_docker_and_base, report_model_images
from zairachem.base.utils.console import echo
from zairachem.base.utils.isaura_report import (
  report_isaura_coverage,
  check_isaura_version_consistency,
  create_and_migrate_project,
  project_exists,
  sanitize_project_name,
)
from zairachem.base.vars import (
  PARAMETERS_FILE,
  RAW_INPUT_FILENAME,
  DATA_SUBFOLDER,
  DATA_FILENAME,
  SMILES_COLUMN,
  DESCRIPTORS_SUBFOLDER,
  ESTIMATORS_SUBFOLDER,
  POOL_SUBFOLDER,
  REPORT_SUBFOLDER,
  OUTPUT_FILENAME,
  DEFAULT_ISAURA_BUCKET,
)

from zairachem.base.utils.pipeline import PipelineStep, SessionFile


class PredictSetup(object):
  def __init__(
    self,
    input_file,
    model_dir,
    output_dir,
    override_dir,
    time_budget=120,
    store_read=False,
    nn=False,
    store_write=False,
  ):
    self.input_file = os.path.abspath(input_file)
    self.override_dir = override_dir
    self.nn = nn
    if output_dir is None:
      self.output_dir = os.path.abspath(self.input_file.split(".")[0])
    else:
      self.output_dir = os.path.abspath(output_dir)
    # Read AND write both target the run's own project (named like the output folder); the central
    # lake (isaura-public) is only migrated in at the start. The folder name is sanitized to a
    # valid S3/MinIO bucket name (lowercase, no underscores).
    project = sanitize_project_name(os.path.basename(os.path.normpath(self.output_dir)))
    self.read_store = project if store_read else None
    self.contribute_store = project if store_write else None

    if os.path.exists(self.output_dir) and not self.override_dir:  # TODO add if wanted
      logger.warning(
        f"Specified output directory existed at {self.output_dir}. Please remove it manually or use [red]--override[/] flag to remove it."
      )
    if os.path.exists(self.output_dir) and self.override_dir:
      logger.warning(
        f"Specified output directory existed at {self.output_dir}. Removing the directory!."
      )
      if self.override_dir:
        shutil.rmtree(self.output_dir)
    assert model_dir is not None, "Model directory not specified"
    self.model_dir = os.path.abspath(model_dir)
    assert self.model_is_ready(), "Model is not ready"
    self.time_budget = time_budget  # TODO

  def model_is_ready(self):
    if not os.path.exists(self.model_dir):
      return False
    if not os.path.exists(os.path.join(self.model_dir, OUTPUT_FILENAME)):
      return False
    if not os.path.exists(os.path.join(self.model_dir, ESTIMATORS_SUBFOLDER)):
      return False
    return True

  def _copy_input_file(self):
    extension = self.input_file.split(".")[-1]
    shutil.copy(
      self.input_file,
      os.path.join(self.output_dir, RAW_INPUT_FILENAME + "." + extension),
    )

  def _make_output_dir(self):
    pass
    # if os.path.exists(self.output_dir):
    #   shutil.rmtree(self.output_dir)
    # os.makedirs(self.output_dir)

  def _open_session(self):
    sf = SessionFile(self.output_dir)
    sf.open_session(mode="predict", output_dir=self.output_dir, model_dir=self.model_dir)
    create_session_symlink(self.output_dir)

  def _make_subfolder(self, name):
    os.makedirs(os.path.join(self.output_dir, name))

  def _make_subfolders(self):
    self._make_subfolder(DATA_SUBFOLDER)
    self._make_subfolder(DESCRIPTORS_SUBFOLDER)
    self._make_subfolder(ESTIMATORS_SUBFOLDER)
    self._make_subfolder(POOL_SUBFOLDER)
    self._make_subfolder(REPORT_SUBFOLDER)
    shutil.copyfile(
      os.path.join(self.model_dir, DATA_SUBFOLDER, PARAMETERS_FILE),
      os.path.join(self.output_dir, DATA_SUBFOLDER, PARAMETERS_FILE),
    )
    self._update_params()

  def _update_params(self):
    """Update params for prediction, overriding training params for isaura settings.

    The isaura precalculation settings (read_store, contribute_store, enable_nns)
    should NOT be inherited from training. They must be explicitly specified
    during prediction, otherwise they default to disabled (None/False).
    """
    params_path = os.path.join(self.output_dir, DATA_SUBFOLDER, PARAMETERS_FILE)
    with open(params_path, "r") as f:
      params = json.load(f)
    # Always override isaura settings - don't inherit from training
    params["read_store"] = self.read_store
    params["enable_nns"] = self.nn
    params["contribute_store"] = self.contribute_store
    with open(params_path, "w") as f:
      json.dump(params, f, indent=4)

  def _initialize(self):
    step = PipelineStep("initialize", self.output_dir)
    if not step.is_done():
      self._make_output_dir()
      self._make_subfolders()
      self._open_session()
      self._copy_input_file()
      step.update()

  def _normalize_input(self):
    step = PipelineStep("normalize_input", self.output_dir)
    if not step.is_done():
      params = ParametersFile(
        path=os.path.join(self.model_dir, DATA_SUBFOLDER, PARAMETERS_FILE)
      ).load()
      f = SingleFileForPrediction(self.input_file, params)
      f.process()
      self.has_tasks = f.has_tasks
      step.update()

  def _standardize(self):
    step = PipelineStep("standardize", self.output_dir)
    if not step.is_done():
      std = ChemblStandardize(os.path.join(self.output_dir, DATA_SUBFOLDER))
      std.run()
      step.update()

  def _tasks(self):
    step = PipelineStep("tasks", self.output_dir)
    if not step.is_done():
      logger.info("[setup] Starting tasks step")
      SingleTasksForPrediction(os.path.join(self.output_dir, DATA_SUBFOLDER)).run()
      step.update()
      logger.info("[setup] Tasks step complete")

  def _merge(self):
    step = PipelineStep("merge", self.output_dir)
    if not step.is_done():
      logger.info("[setup] Starting merge step")
      DataMergerForPrediction(os.path.join(self.output_dir, DATA_SUBFOLDER)).run(self.has_tasks)
      step.update()
      logger.info("[setup] Merge step complete")

  def _clean(self):
    step = PipelineStep("clean", self.output_dir)
    if not step.is_done():
      logger.info("[setup] Starting clean step")
      SetupCleaner(os.path.join(self.output_dir, DATA_SUBFOLDER)).run()
      step.update()
      logger.info("[setup] Clean step complete")

  def _check(self):
    step = PipelineStep("setup_check", self.output_dir)
    if not step.is_done():
      logger.info("[setup] Starting check step")
      SetupChecker(self.output_dir).run()
      step.update()
      logger.info("[setup] Check step complete")

  def _print_summary(self):
    import pandas as pd

    def collapse(p):
      home = os.path.expanduser("~")
      return "~" + p[len(home) :] if p.startswith(home) else p

    data_dir = os.path.join(self.output_dir, DATA_SUBFOLDER)
    params = {}
    params_path = os.path.join(data_dir, PARAMETERS_FILE)
    if os.path.exists(params_path):
      with open(params_path) as f:
        params = json.load(f)

    task = params.get("task")
    task_label = {"classification": "Classification (binary)", "regression": "Regression"}.get(
      task, task or "?"
    )

    project = sanitize_project_name(os.path.basename(os.path.normpath(self.output_dir)))
    read = "[green]on[/]" if params.get("read_store") else "[red]off[/]"
    write = "[green]on[/]" if params.get("contribute_store") else "[red]off[/]"
    isaura = (
      f"project [bold]{project}[/] (read {read} · write {write}) · "
      f"lake [bold]{DEFAULT_ISAURA_BUCKET}[/]"
    )
    if params.get("enable_nns"):
      isaura += " · nearest-neighbors"

    rows = [
      ("Input", os.path.basename(self.input_file)),
      ("Output", collapse(self.output_dir)),
      ("Model", collapse(self.model_dir)),
      ("Task", task_label),
    ]
    data_path = os.path.join(data_dir, DATA_FILENAME)
    if os.path.exists(data_path):
      rows.append(("Compounds", f"{len(pd.read_csv(data_path)):,}"))
    rows.append(("Featurizers", params.get("featurizer_ids", [])))
    projection_ids = params.get("projection_ids", [])
    rows.append(("Projection", projection_ids or "[yellow]skipped[/]"))
    rows.append(("Isaura store", isaura))

    summary_panel("ZairaChem · Predict", rows)

  def update_elapsed_time(self):
    ZairaBase().update_elapsed_time()

  def is_done(self):
    if os.path.exists(os.path.join(self.output_dir, OUTPUT_FILENAME)):
      return True
    else:
      return False

  def _model_id_groups(self):
    with open(os.path.join(self.model_dir, DATA_SUBFOLDER, PARAMETERS_FILE)) as f:
      p = json.load(f)
    return p.get("featurizer_ids", []), p.get("projection_ids", [])

  def _input_smiles(self):
    import pandas as pd

    df = pd.read_csv(os.path.join(self.output_dir, DATA_SUBFOLDER, DATA_FILENAME))
    return df[SMILES_COLUMN].astype(str).tolist()

  def setup(self):
    require_docker_and_base()
    self._initialize()
    self._normalize_input()
    self._standardize()
    if self.has_tasks:
      self._tasks()
    self._merge()
    self._check()
    self._clean()
    self._print_summary()
    featurizers, projections = self._model_id_groups()
    report_model_images(featurizers, projections)
    # Coverage/version checks always inspect the central lake (migration source), not the project.
    check_isaura_version_consistency(DEFAULT_ISAURA_BUCKET, featurizers, projections)
    smiles = self._input_smiles()
    report_isaura_coverage(DEFAULT_ISAURA_BUCKET, featurizers, projections, smiles)
    # Project lifecycle (provisional for predict; the well-defined case is fit).
    if self.contribute_store:
      create_and_migrate_project(
        self.contribute_store, featurizers, projections, smiles, output_dir=self.output_dir
      )
    elif self.read_store and project_exists(self.read_store) is False:
      echo(
        f"Isaura project '{self.read_store}' does not exist — nothing to read; "
        "computing from scratch.",
        kind="warning",
      )
      self.read_store = None
      self._update_params()
    self.update_elapsed_time()


class ONNXPredictSetup(PredictSetup):
  def __init__(self, input_file, output_dir, model_dir, time_budget):
    super().__init__(input_file, output_dir, model_dir, time_budget)

  def _make_subfolders(self):
    self._make_subfolder(DATA_SUBFOLDER)
    self._make_subfolder(REPORT_SUBFOLDER)

  def _normalize_input(self):
    step = PipelineStep("normalize_input", self.output_dir)
    if not step.is_done():
      f = SingleFileForPrediction(self.input_file, None)
      f.process()
      self.has_tasks = f.has_tasks
      step.update()

  def setup(self):
    self._initialize()
    self._normalize_input()
    self._standardize()
    if self.has_tasks:
      self._tasks()
    self._merge()
    self._clean()
    self.update_elapsed_time()
