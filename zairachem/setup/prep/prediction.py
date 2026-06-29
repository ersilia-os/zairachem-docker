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

from zairachem.base import create_session_symlink, params_path
from zairachem.base.utils.logging import logger
from zairachem.base.utils.console import summary_panel
from zairachem.base.utils.preflight import require_docker_and_base, report_model_images
from zairachem.base.utils.console import echo
from zairachem.base.utils.utils import write_smiles_list
from zairachem.base.utils.isaura_report import (
  report_store_availability,
  check_isaura_version_consistency,
  create_and_migrate_project,
  project_exists,
)
from zairachem.base.vars import (
  DATA_SUBFOLDER,
  DATA_FILENAME,
  METADATA_SUBFOLDER,
  RESULTS_SUBFOLDER,
  TRANSFORMERS_SUBFOLDER,
  DESCRIPTORS_SUBFOLDER,
  ESTIMATORS_SUBFOLDER,
  POOL_SUBFOLDER,
  REPORT_SUBFOLDER,
  OUTPUT_FILENAME,
  DEFAULT_ISAURA_BUCKET,
)

from zairachem.base.utils.pipeline import PipelineStep, SessionFile
from zairachem.setup.prep.base import BaseSetup, format_store_summary


class PredictSetup(BaseSetup):
  def __init__(
    self,
    input_file,
    model_dir,
    output_dir,
    override_dir,
    time_budget=120,
    store=None,
  ):
    self.input_file = os.path.abspath(input_file)
    self.override_dir = override_dir
    if output_dir is None:
      self.output_dir = os.path.abspath(self.input_file.split(".")[0])
    else:
      self.output_dir = os.path.abspath(output_dir)
    # `store` is the already-resolved isaura project name (or None), produced by the CLI (a bare
    # --store reuses the store the model was trained with). read_store/contribute_store keep the
    # internal describe/treat contract (both = the project, or both None).
    self.store = store
    self.read_store = store
    self.contribute_store = store

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
    if model_dir is None:
      self._report_not_ready(None, None)
    self.model_dir = os.path.abspath(model_dir)
    self._check_model_ready()
    self.time_budget = time_budget  # TODO

  def _required_artifacts(self):
    """(label, absolute path) pairs that a complete, trained ZairaChem model must contain."""
    return [
      ("Run configuration", params_path(self.model_dir)),
      ("Descriptor manifest", os.path.join(self.model_dir, DESCRIPTORS_SUBFOLDER, "done_eos.json")),
      ("Trained estimators", os.path.join(self.model_dir, ESTIMATORS_SUBFOLDER)),
      ("Model output table", os.path.join(self.model_dir, OUTPUT_FILENAME)),
    ]

  def model_is_ready(self):
    """True if the model folder contains every artifact prediction needs."""
    if not os.path.isdir(self.model_dir):
      return False
    return all(os.path.exists(p) for _, p in self._required_artifacts())

  def _check_model_ready(self):
    """Validate the model folder up front; show a formatted message and stop if it's incomplete."""
    if not os.path.isdir(self.model_dir):
      self._report_not_ready(self.model_dir, None)
      return
    checks = [(label, p, os.path.exists(p)) for label, p in self._required_artifacts()]
    if not all(ok for _, _, ok in checks):
      self._report_not_ready(self.model_dir, checks)

  def _report_not_ready(self, model_dir, checks):
    from rich import box
    from rich.panel import Panel

    from zairachem.base.utils.console import console

    if model_dir is None:
      body = (
        "No model directory was provided.\n\nPass a trained model with [bold]-m / --model-dir[/]."
      )
    elif checks is None:
      body = (
        f"Model directory not found:\n  [red]{model_dir}[/]\n\n"
        "Train a model first with [bold]zairachem fit[/]."
      )
    else:
      lines = "\n".join(
        f"  {'[green]✓[/]' if ok else '[red]✗[/]'} {label}  [dim]({os.path.relpath(p, model_dir)})[/]"
        for label, p, ok in checks
      )
      body = (
        "This folder is not a complete ZairaChem model — required files are missing:\n\n"
        f"{lines}\n\n"
        f"[dim]Train it first, e.g.[/]  [bold]zairachem fit -i <data.csv> -m {model_dir} -c[/]"
      )
    console.print(
      Panel(
        body,
        title="[bold red]✖  Model not ready for prediction[/]",
        title_align="left",
        border_style="red",
        box=box.ROUNDED,
        padding=(1, 2),
        expand=False,
      )
    )
    raise SystemExit(1)

  def _make_output_dir(self):
    pass
    # if os.path.exists(self.output_dir):
    #   shutil.rmtree(self.output_dir)
    # os.makedirs(self.output_dir)

  def _open_session(self):
    sf = SessionFile(self.output_dir)
    sf.open_session(mode="predict", output_dir=self.output_dir, model_dir=self.model_dir)
    create_session_symlink(self.output_dir)

  def _make_subfolders(self):
    self._make_subfolder(DATA_SUBFOLDER)
    self._make_subfolder(METADATA_SUBFOLDER)
    self._make_subfolder(RESULTS_SUBFOLDER)
    self._make_subfolder(TRANSFORMERS_SUBFOLDER)
    self._make_subfolder(DESCRIPTORS_SUBFOLDER)
    self._make_subfolder(ESTIMATORS_SUBFOLDER)
    self._make_subfolder(POOL_SUBFOLDER)
    self._make_subfolder(REPORT_SUBFOLDER)
    shutil.copyfile(params_path(self.model_dir), params_path(self.output_dir))
    self._update_params()

  def _update_params(self):
    """Set the isaura store for this prediction, overriding the value inherited from training.

    The store (read_store/contribute_store/store) is resolved at the CLI — a bare ``--store`` reuses
    the store the model was trained with; an explicit name overrides it; omitted means no store.
    """
    pp = params_path(self.output_dir)
    with open(pp, "r") as f:
      params = json.load(f)
    params["store"] = self.store
    params["read_store"] = self.read_store
    params["contribute_store"] = self.contribute_store
    with open(pp, "w") as f:
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
      params = ParametersFile(path=params_path(self.model_dir)).load()
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
    pp = params_path(self.output_dir)
    if os.path.exists(pp):
      with open(pp) as f:
        params = json.load(f)

    task = params.get("task")
    task_label = {"classification": "Classification (binary)", "regression": "Regression"}.get(
      task, task or "?"
    )

    isaura = format_store_summary(params.get("store"))

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
    if projection_ids:
      proj = "MW vs LogP  " + "  ".join(f"[green]{m}[/]" for m in projection_ids)
    else:
      proj = "[green]MW vs LogP[/] [dim](built-in)[/]"
    rows.append(("Projection", proj))
    rows.append(("Isaura store", isaura))

    summary_panel("ZairaChem · Predict", rows)

  def _model_id_groups(self):
    with open(params_path(self.model_dir)) as f:
      p = json.load(f)
    return p.get("featurizer_ids", []), p.get("projection_ids", [])

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
    # Persist a bare one-column `smiles` list of the run's compounds for ad-hoc manual use.
    write_smiles_list(os.path.join(self.output_dir, DATA_SUBFOLDER), smiles)
    # Show where the input compounds already live, as a resolution waterfall (project → lake →
    # remote → to-compute): each store is checked only on the compounds the earlier ones didn't have.
    # Skipped entirely with no store: nothing is read from any store, so the table would be noise.
    project = self.contribute_store  # == read_store; both set together by --store
    if project:
      report_store_availability(featurizers, projections, smiles, project=project)
    if project and project != DEFAULT_ISAURA_BUCKET:
      # Store on (a named project): seed it from the lake (idempotent; tops up on a re-run), then read.
      create_and_migrate_project(
        project, featurizers, projections, smiles, output_dir=self.output_dir
      )
      # Safety net: if the project still doesn't exist (isaura/engine down, migration failed),
      # disable the store for this run so Describe computes cleanly from scratch.
      if project_exists(project) is False:
        echo("Isaura store unavailable — computing from scratch.", kind="warning")
        self.store = None
        self.read_store = None
        self.contribute_store = None
        self._update_params()
    # store == isaura-public: read/write the shared lake directly — no self-migration needed.
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
