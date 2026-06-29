import json, os, shutil

from zairachem.base import create_session_symlink, params_path
from zairachem.base.utils.console import summary_panel
from zairachem.base.utils.preflight import (
  require_docker_and_base,
  report_model_images,
  report_reference_transformers,
  validate_model_roles,
)
from zairachem.base.utils.console import echo
from zairachem.base.utils.utils import write_smiles_list
from zairachem.base.utils.isaura_report import (
  report_store_availability,
  check_isaura_version_consistency,
  create_and_migrate_project,
  project_exists,
)
from zairachem.setup.prep import (
  ModelIdsFile,
  SingleFile,
  ChemblStandardize,
  SingleTasks,
  DataMerger,
  SetupCleaner,
  SetupChecker,
  PipelineStep,
  SessionFile,
)
from zairachem.setup.prep.base import BaseSetup, format_store_summary
from zairachem.base.vars import (
  DEFAULT_FEATURIZERS,
  DEFAULT_PROJECTIONS,
  DEFAULT_REFERENCE_LIBRARY,
  DATA_SUBFOLDER,
  DATA_FILENAME,
  METADATA_SUBFOLDER,
  RESULTS_SUBFOLDER,
  TRANSFORMERS_SUBFOLDER,
  INPUT_SCHEMA_FILENAME,
  DEFAULT_ISAURA_BUCKET,
  DESCRIPTORS_SUBFOLDER,
  ESTIMATORS_SUBFOLDER,
  POOL_SUBFOLDER,
  REPORT_SUBFOLDER,
)


class TrainSetup(BaseSetup):
  def __init__(self, input_file, output_dir, task, model_ids, store, projection_ids=None):
    if output_dir is None:
      output_dir = input_file.split(".")[0]
    if model_ids is not None:
      self.model_ids = ModelIdsFile(model_ids).load()
    else:
      # No --featurizer-ids given: fall back to the default descriptors.
      self.model_ids = {}
    self.featurizer_ids = self.model_ids.get("featurizer_ids", DEFAULT_FEATURIZERS)
    # Projection models (--projection-ids) drive the report's 2-D embedding; they are NOT model
    # features. Explicit --projection-ids wins; else a projection_ids key in the --featurizer-ids
    # JSON file; else the default. The report always also shows the built-in MW-vs-LogP projection.
    if projection_ids:
      self.projection_ids = ModelIdsFile.parse_ids(projection_ids, flag="--projection-ids")
    else:
      self.projection_ids = self.model_ids.get("projection_ids", DEFAULT_PROJECTIONS)
    self.input_file = os.path.abspath(input_file)
    self.output_dir = os.path.abspath(output_dir)
    self.task = task
    assert self.task in ["classification", "regression"]
    # `store` is the already-resolved isaura project name (or None for "no store"), produced by the
    # CLI: bare --store -> the model-dir name; --store NAME -> NAME; --store isaura-public -> the
    # shared lake (read/write directly, no migration). read_store/contribute_store keep the internal
    # describe/treat contract (both = the project, or both None); `store` is persisted so downstream
    # steps and predict reuse the resolved name without re-deriving it.
    self.project = store
    project = store
    self.params = {
      "task": self.task,
      "featurizer_ids": self.featurizer_ids,
      "projection_ids": self.projection_ids,
      "store": project,
      "read_store": project,
      "contribute_store": project,
      # Reference library whose pre-fitted transformers the treat step applies. Persisted so
      # predict runs reuse the exact same library (the transformer copies are also saved per model).
      "reference_library": DEFAULT_REFERENCE_LIBRARY,
    }

  def _save_params(self):
    with open(params_path(self.output_dir), "w") as f:
      json.dump(self.params, f, indent=4)

  def _make_output_dir(self):
    if os.path.exists(self.output_dir):
      shutil.rmtree(self.output_dir)
    os.makedirs(self.output_dir)

  def _open_session(self):
    sf = SessionFile(self.output_dir)
    sf.open_session(mode="fit", output_dir=self.output_dir, model_dir=self.output_dir)
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

  def _initialize(self):
    # Always rebuild from scratch. `run_fit` already returns early for a *complete* model
    # (output.csv present → "already trained"), so any dir reaching here is absent or a partial
    # leftover. Rebuilding guarantees a clean state, fresh params, and — crucially — a freshly
    # re-pointed global session symlink (a stale/dangling one otherwise crashes the run).
    step = PipelineStep("initialize", self.output_dir)
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

  def _isaura_summary(self):
    return format_store_summary(self.params.get("store"))

  def _print_summary(self):
    import pandas as pd

    data_dir = os.path.join(self.output_dir, DATA_SUBFOLDER)
    df = pd.read_csv(os.path.join(data_dir, DATA_FILENAME))
    n_compounds = len(df)

    # Original activity column name, as written to the input schema during normalization.
    values_column = "?"
    schema_path = os.path.join(data_dir, INPUT_SCHEMA_FILENAME)
    if os.path.exists(schema_path):
      with open(schema_path) as f:
        values_column = json.load(f).get("values_column", "?")

    home = os.path.expanduser("~")
    output = self.output_dir
    if output.startswith(home):
      output = "~" + output[len(home) :]

    rows = [
      ("Input", f"[white]{os.path.basename(self.input_file)}[/]"),
      ("Output", f"[dim]{output}[/]"),
    ]
    if self.task == "classification":
      rows.append(("Task", "[magenta]Classification[/] [dim](binary)[/]"))
      if "bin" in df.columns:
        n_act = int(df["bin"].sum())
        rows.append((
          "Compounds",
          f"[bold]{n_compounds:,}[/]   [green]{n_act:,} active[/] · [yellow]{n_compounds - n_act:,} inactive[/]",
        ))
      else:
        rows.append(("Compounds", f"[bold]{n_compounds:,}[/]"))
    else:
      rows.append(("Task", "[magenta]Regression[/]"))
      rows.append(("Compounds", f"[bold]{n_compounds:,}[/]"))
    rows.append(("Activity column", f"[cyan]{values_column}[/]"))
    rows.append(("Featurizers", "  ".join(f"[green]{m}[/]" for m in self.featurizer_ids)))
    if self.projection_ids:
      proj = "MW vs LogP  " + "  ".join(f"[green]{m}[/]" for m in self.projection_ids)
    else:
      proj = "[green]MW vs LogP[/] [dim](built-in)[/]"
    rows.append(("Projection", proj))
    rows.append(("Isaura store", self._isaura_summary()))

    summary_panel("ZairaChem · Setup", rows)

  def setup(self):
    require_docker_and_base()
    self._initialize()
    self._normalize_input()
    self._standardise()
    self._tasks()
    self._merge()
    self._check()
    self._clean()
    self._print_summary()
    validate_model_roles(self.featurizer_ids, self.projection_ids)
    report_model_images(self.featurizer_ids, self.projection_ids)
    # Confirm every featurizer has a transformer in the reference library before any descriptors are
    # computed; the treat step will apply these and cannot proceed without them.
    report_reference_transformers(
      self.featurizer_ids, self.params.get("reference_library", DEFAULT_REFERENCE_LIBRARY)
    )
    # Coverage/version checks always inspect the central lake (the migration source), not the
    # run's own read/write project.
    check_isaura_version_consistency(
      DEFAULT_ISAURA_BUCKET, self.featurizer_ids, self.projection_ids
    )
    smiles = self._input_smiles()
    # Persist a bare one-column `smiles` list of the run's compounds for ad-hoc manual use.
    write_smiles_list(os.path.join(self.output_dir, DATA_SUBFOLDER), smiles)
    # Show where the input compounds already live, as a resolution waterfall (project → lake →
    # remote → to-compute): each store is checked only on the compounds the earlier ones didn't have.
    # Skipped entirely with no store: nothing is read from any store, so everything is computed fresh
    # and the availability table would be noise.
    project = self.params.get("contribute_store")  # == read_store; both set together by --store
    if project:
      report_store_availability(self.featurizer_ids, self.projection_ids, smiles, project=project)
    if project and project != DEFAULT_ISAURA_BUCKET:
      # Store on (a named project): seed it from the lake — idempotent, so a re-run simply tops up
      # an existing project. Describe then reads it and caches new results.
      create_and_migrate_project(
        project, self.featurizer_ids, self.projection_ids, smiles, output_dir=self.output_dir
      )
      # Safety net: if the project still doesn't exist (isaura/engine down, migration failed),
      # disable the store for this run so Describe computes cleanly from scratch.
      if project_exists(project) is False:
        echo("Isaura store unavailable — computing from scratch.", kind="warning")
        self.params["read_store"] = None
        self.params["contribute_store"] = None
        self.params["store"] = None
        self._save_params()
    # store == isaura-public: read/write the shared lake directly — no self-migration needed.
    self.update_elapsed_time()
