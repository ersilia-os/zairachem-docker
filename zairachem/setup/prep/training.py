import json, os, shutil

from zairachem.base import ZairaBase, create_session_symlink
from zairachem.base.utils.console import summary_panel
from zairachem.base.utils.preflight import require_docker_and_base, report_model_images
from zairachem.base.utils.console import echo
from zairachem.base.utils.isaura_report import (
  report_isaura_coverage,
  check_isaura_version_consistency,
  check_isaura_project_available,
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
from zairachem.base.vars import (
  DEFAULT_FEATURIZERS,
  DEFAULT_PROJECTIONS,
  PARAMETERS_FILE,
  RAW_INPUT_FILENAME,
  DATA_SUBFOLDER,
  DATA_FILENAME,
  INPUT_SCHEMA_FILENAME,
  SMILES_COLUMN,
  DEFAULT_ISAURA_BUCKET,
  DESCRIPTORS_SUBFOLDER,
  ESTIMATORS_SUBFOLDER,
  POOL_SUBFOLDER,
  REPORT_SUBFOLDER,
  OUTPUT_FILENAME,
)

from rdkit import RDLogger

RDLogger.logger().setLevel(RDLogger.CRITICAL)


class TrainSetup(object):
  def __init__(self, input_file, output_dir, task, model_ids, store_read, nn, store_write):
    if output_dir is None:
      output_dir = input_file.split(".")[0]
    if model_ids is not None:
      self.model_ids = ModelIdsFile(model_ids).load()
    else:
      # No --eos-ids file provided: fall back to the default descriptors/projection.
      self.model_ids = {}
    self.featurizer_ids = self.model_ids.get("featurizer_ids", DEFAULT_FEATURIZERS)
    self.projection_ids = self.model_ids.get("projection_ids", DEFAULT_PROJECTIONS)
    self.input_file = os.path.abspath(input_file)
    self.output_dir = os.path.abspath(output_dir)
    self.task = task
    assert self.task in ["classification", "regression"]
    # Read AND write both target the run's own project (named like the model folder). The central
    # lake (isaura-public) is only read at the start to migrate data into the project; it is not the
    # run's read/write store. The --store r|w|rw flags toggle read/write on the project.
    project = os.path.basename(os.path.normpath(self.output_dir))
    self.params = {
      "task": self.task,
      "featurizer_ids": self.featurizer_ids,
      "projection_ids": self.projection_ids,
      "read_store": project if store_read else None,
      "enable_nns": nn,
      "contribute_store": project if store_write else None,
    }

  def _copy_input_file(self):
    extension = self.input_file.split(".")[-1]
    shutil.copy(
      self.input_file,
      os.path.join(self.output_dir, RAW_INPUT_FILENAME + "." + extension),
    )

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
    self._make_subfolder(REPORT_SUBFOLDER)

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
    # The run reads/writes its own project; the lake is migrated in at the start.
    project = os.path.basename(os.path.normpath(self.output_dir))
    read = "[green]on[/]" if self.params.get("read_store") else "[red]off[/]"
    write = "[green]on[/]" if self.params.get("contribute_store") else "[red]off[/]"
    summary = (
      f"project [bold]{project}[/] (read {read} · write {write}) · "
      f"lake [bold]{DEFAULT_ISAURA_BUCKET}[/]"
    )
    if self.params.get("enable_nns"):
      summary += " · nearest-neighbors"
    return summary

  def _input_smiles(self):
    import pandas as pd

    df = pd.read_csv(os.path.join(self.output_dir, DATA_SUBFOLDER, DATA_FILENAME))
    return df[SMILES_COLUMN].astype(str).tolist()

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
    rows.append(("Projection", "  ".join(f"[green]{m}[/]" for m in self.projection_ids)))
    rows.append(("Isaura store", self._isaura_summary()))

    summary_panel("ZairaChem · Setup", rows)

  def update_elapsed_time(self):
    ZairaBase().update_elapsed_time()

  def is_done(self):
    if os.path.exists(os.path.join(self.output_dir, OUTPUT_FILENAME)):
      return True
    else:
      return False

  def setup(self):
    require_docker_and_base()
    # Only guard the project name when we'll actually write (create) it.
    if self.params.get("contribute_store"):
      check_isaura_project_available(self.output_dir)
    self._initialize()
    self._normalize_input()
    self._standardise()
    self._tasks()
    self._merge()
    self._check()
    self._clean()
    self._print_summary()
    report_model_images(self.featurizer_ids, self.projection_ids)
    # Coverage/version checks always inspect the central lake (the migration source), not the
    # run's own read/write project.
    check_isaura_version_consistency(
      DEFAULT_ISAURA_BUCKET, self.featurizer_ids, self.projection_ids
    )
    smiles = self._input_smiles()
    report_isaura_coverage(DEFAULT_ISAURA_BUCKET, self.featurizer_ids, self.projection_ids, smiles)
    if self.params.get("contribute_store"):
      # Write enabled: create the project (named like the model folder) and seed it from the lake.
      create_and_migrate_project(
        self.params["contribute_store"],
        self.featurizer_ids,
        self.projection_ids,
        smiles,
        output_dir=self.output_dir,
      )
    elif self.params.get("read_store") and project_exists(self.params["read_store"]) is False:
      # Read-only against a project that doesn't exist: warn and fall back to computing from scratch.
      echo(
        f"Isaura project '{self.params['read_store']}' does not exist — nothing to read; "
        "computing from scratch.",
        kind="warning",
      )
      self.params["read_store"] = None
      self._save_params()
    self.update_elapsed_time()
