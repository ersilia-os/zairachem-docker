import json
import os
import shutil

from zairachem.base import params_path
from zairachem.base.utils.console import echo, summary_panel
from zairachem.base.utils.isaura_report import (
  check_isaura_version_consistency,
  create_and_migrate_project,
  project_exists,
  report_store_availability,
)
from zairachem.base.utils.logging import logger
from zairachem.base.utils.preflight import (
  prepare_rank_references,
  prepare_reference_transformers,
  report_model_images,
  require_docker_and_base,
  validate_model_roles,
)
from zairachem.base.utils.utils import write_smiles_list
from zairachem.base.vars import (
  DATA_FILENAME,
  DATA_SUBFOLDER,
  DEFAULT_FEATURIZERS,
  DEFAULT_ISAURA_BUCKET,
  DEFAULT_MAX_DESCRIPTORS,
  DEFAULT_PROJECTIONS,
  DEFAULT_REFERENCE_LIBRARY,
  DESCRIPTORS_SUBFOLDER,
  ESTIMATORS_SUBFOLDER,
  INPUT_SCHEMA_FILENAME,
  MAX_FEATURIZERS,
  MAX_PROJECTIONS,
  METADATA_SUBFOLDER,
  POOL_SUBFOLDER,
  REPORT_SUBFOLDER,
  RESULTS_SUBFOLDER,
  SMILES_COLUMN,
  SPLITS_FILENAME,
  TARGET_COLUMN,
  TRANSFORMERS_SUBFOLDER,
)
from zairachem.setup.prep import (
  ChemblStandardize,
  DataMerger,
  ModelIdsFile,
  PipelineStep,
  SessionFile,
  SetupChecker,
  SetupCleaner,
  SingleFile,
  SingleTasks,
)
from zairachem.setup.prep.base import BaseSetup, format_store_summary


def _check_model_count(ids, cap, kind, flag):
  """Reject a run that requests more Hub models of one kind than the cap allows.

  Parameters
  ----------
  ids : list of str
    The resolved Ersilia Model Hub IDs.
  cap : int
    Maximum number of IDs permitted for this kind.
  kind : str
    Human-readable model kind (e.g. ``"descriptor"``), used in the error message.
  flag : str
    The CLI flag the user would use to change the list, named in the error message.

  Raises
  ------
  ValueError
    If ``len(ids)`` exceeds ``cap``.
  """
  if len(ids) > cap:
    raise ValueError(
      f"Too many {kind} models: {len(ids)} requested but at most {cap} are allowed. "
      f"Trim {flag} to {cap} or fewer IDs."
    )


class TrainSetup(BaseSetup):
  """Setup step that prepares a fit run: data folder, ``parameters.json``, splits, featurizers."""

  def __init__(
    self,
    input_file,
    output_dir,
    task,
    model_ids,
    store,
    projection_ids=None,
    evaluate=False,
    evaluate_repeats=3,
    evaluate_schemas=None,
    max_descriptors=DEFAULT_MAX_DESCRIPTORS,
  ):
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
    # JSON file; else the default. The report always also shows the built-in MW-vs-LogP projection,
    # so `--projection-ids none` (or a bare -p) disables the Ersilia projections and leaves only it.
    if projection_ids is not None:
      if str(projection_ids).strip().lower() in ("none", "null", "off", ""):
        self.projection_ids = []
      else:
        self.projection_ids = ModelIdsFile.parse_ids(projection_ids, flag="--projection-ids")
    else:
      self.projection_ids = self.model_ids.get("projection_ids", DEFAULT_PROJECTIONS)
    # Enforce the per-run model-count ceilings regardless of source (CLI flag, JSON file, default).
    _check_model_count(self.featurizer_ids, MAX_FEATURIZERS, "descriptor", "--featurizer-ids")
    _check_model_count(self.projection_ids, MAX_PROJECTIONS, "projection", "--projection-ids")
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
      # Held-out validation (--evaluate): when True, setup writes metadata/splits.json and the
      # holdout step runs the folds after pooling. ``evaluate_repeats`` is the per-schema fold
      # count (random/scaffold/butina); total folds = 1 + 3 * repeats.
      "evaluate": bool(evaluate),
      "evaluate_repeats": max(0, int(evaluate_repeats)),
      # Which fold families to run (subset of holdout.splits.FOLD_SCHEMAS); None = all four.
      "evaluate_schemas": list(evaluate_schemas) if evaluate_schemas else None,
      # Pre-screen descriptors and fully train only the top-K; None = train all.
      "max_descriptors": int(max_descriptors) if max_descriptors else None,
    }
    # Set True by run_fit when continuing an unfinished run in an existing dir (no --override).
    self.resume = False

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
    step = PipelineStep("initialize", self.output_dir)
    if self.resume and step.is_done():
      # Resuming an unfinished run: keep every existing artifact AND the session "steps" list (so the
      # guarded downstream steps skip what's done and continue). Do NOT wipe and do NOT call
      # _open_session (it rewrites session.json without "steps", erasing the resume markers). Only
      # ensure the subfolders exist and restart the elapsed-time clock so the crash→resume gap isn't
      # billed. params.json + the input copy are already on disk.
      self._make_subfolders()
      self.reset_time()
      return
    # Fresh: rebuild from scratch. Guarantees a clean state, fresh params, and a freshly re-pointed
    # session symlink (a stale/dangling one otherwise crashes the run).
    self._make_output_dir()
    self._open_session()
    self._make_subfolders()
    self._save_params()
    self._copy_input_file()
    step.update()

  def _keep_featurizers(self, kept):
    """Restrict the run to ``kept`` featurizers, recording the dropped ones in ``params.json``.

    Downstream steps read the descriptor list from ``params.json``, so this is the one place a
    descriptor is removed from a run. ``skipped_featurizer_ids`` keeps resume comparisons honest.
    """
    dropped = [m for m in self.featurizer_ids if m not in kept]
    if not dropped:
      return
    self.featurizer_ids = list(kept)
    self.params["featurizer_ids"] = list(kept)
    skipped = self.params.get("skipped_featurizer_ids") or []
    self.params["skipped_featurizer_ids"] = skipped + [m for m in dropped if m not in skipped]
    self._save_params()

  def _load_disk_params(self):
    p = params_path(self.output_dir)
    if not os.path.exists(p):
      return None
    with open(p) as f:
      return json.load(f)

  def resume_config_conflict(self, explicit_config):
    """Reconcile the run config when resuming. The on-disk ``params.json`` is authoritative (the
    partial artifacts were computed under it). If the caller *explicitly* re-passed a config-affecting
    flag that conflicts with the trained config, return an error message so the run aborts (use
    --override for a fresh run with new settings); otherwise adopt the on-disk config and return None.

    ``explicit_config`` is the set of click option names the user passed on the command line (e.g.
    ``{"featurizer_ids"}``); flags not in it silently inherit the on-disk value.
    """
    disk = self._load_disk_params()
    if disk is None:
      return None
    checks = {
      "classification": ("task", self.task, disk.get("task")),
      # Descriptors skipped at setup (no transformer) still count as requested.
      "featurizer_ids": (
        "featurizers",
        sorted(self.featurizer_ids),
        sorted((disk.get("featurizer_ids") or []) + (disk.get("skipped_featurizer_ids") or [])),
      ),
      "projection_ids": ("projections", self.projection_ids, disk.get("projection_ids")),
      "store": ("store", self.params.get("store"), disk.get("store")),
      "max_descriptors": (
        "max descriptors",
        self.params.get("max_descriptors"),
        disk.get("max_descriptors"),
      ),
    }
    conflicts = [
      f"{label} (requested {cli!r}, trained with {dsk!r})"
      for flag, (label, cli, dsk) in checks.items()
      if flag in explicit_config and cli != dsk
    ]
    if conflicts:
      return (
        f"Cannot resume '{self.output_dir}' with a different configuration: {'; '.join(conflicts)}. "
        f"Pass --override to discard it and train fresh with the new settings."
      )
    # Adopt the original config so the setup summary + store logic match the trained model.
    self.params = disk
    self.task = disk.get("task", self.task)
    self.featurizer_ids = disk.get("featurizer_ids", self.featurizer_ids)
    self.projection_ids = disk.get("projection_ids", self.projection_ids)
    return None

  def _normalize_input(self):
    step = PipelineStep("normalize_input", self.output_dir)
    if not step.is_done():
      f = SingleFile(self.input_file, self.params, self.output_dir)
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

  def _evaluate_splits(self):
    """Compute the held-out fold definitions and persist them to ``metadata/splits.json``.

    Only runs when ``--evaluate`` is set. Cheap and deterministic: it reads the finalized
    ``data.csv`` (SMILES + the task's target column) and writes the 10-fold menu built by
    :func:`zairachem.holdout.splits.build_fold_definitions`. The fold execution itself happens later
    (after pooling), reusing the shared descriptors.
    """
    if not self.params.get("evaluate"):
      return
    step = PipelineStep("evaluate_splits", self.output_dir)
    if step.is_done():
      return
    import pandas as pd

    from zairachem.holdout.splits import build_fold_definitions

    data_dir = os.path.join(self.output_dir, DATA_SUBFOLDER)
    df = pd.read_csv(os.path.join(data_dir, DATA_FILENAME))
    folds = build_fold_definitions(
      list(df[SMILES_COLUMN]),
      list(df[TARGET_COLUMN[self.task]]),
      repeats=self.params.get("evaluate_repeats", 3),
      schemas=self.params.get("evaluate_schemas"),
    )
    with open(os.path.join(self.output_dir, METADATA_SUBFOLDER, SPLITS_FILENAME), "w") as f:
      json.dump(folds, f, indent=4)
    logger.info(f"[evaluate] Wrote {len(folds)} held-out fold definitions to {SPLITS_FILENAME}")
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
    md = self.params.get("max_descriptors")
    if self.task == "classification" and md:
      n = len(self.featurizer_ids)
      if md < n:
        rows.append((
          "Ensemble cap",
          f"up to [bold]{md}[/] of {n} [dim](greedy pre-screen by pool contribution)[/]",
        ))
      else:
        rows.append(("Ensemble cap", f"[dim]all {n} (no pre-screening)[/]"))
    if self.projection_ids:
      proj = "MW vs LogP  " + "  ".join(f"[green]{m}[/]" for m in self.projection_ids)
    else:
      proj = "[green]MW vs LogP[/] [dim](built-in)[/]"
    rows.append(("Projection", proj))
    rows.append(("Isaura store", self._isaura_summary()))
    splits = self._load_splits()
    if splits:
      rows.append(("Held-out validation", self._splits_one_liner(splits)))

    summary_panel("ZairaChem · Setup", rows)
    if splits:
      self._print_splits_table(splits, df)

  def _load_splits(self):
    """Load metadata/splits.json (the fold definitions), or None if --evaluate was not set."""
    path = os.path.join(self.output_dir, METADATA_SUBFOLDER, SPLITS_FILENAME)
    if not os.path.exists(path):
      return None
    with open(path) as f:
      return json.load(f)

  @staticmethod
  def _splits_one_liner(splits):
    counts = {}
    for spec in splits.values():
      counts[spec["strategy"]] = counts.get(spec["strategy"], 0) + 1
    parts = [f"{n}× {strat}" if n > 1 else strat for strat, n in counts.items()]
    return f"[bold]{len(splits)}[/] folds — " + " · ".join(parts)

  # Soft per-family tint for the fold names (cool hues, kept clear of the green→red % ramp).
  _SPLIT_HUES = {
    "random": "#79c0ff",
    "scaffold": "#d2a8ff",
    "scaffold_det": "#d2a8ff",
    "butina": "#56d4dd",
  }

  def _print_splits_table(self, splits, df):
    """Render the per-fold split table (Fold | Train | Test | the test set's target) at setup.

    The last column is the test set's active rate (classification) or its mean target value
    (regression), shaded green→amber→red by how far it deviates from the dataset's overall figure —
    calm green = well balanced, warm = imbalanced, as the scaffold split often is. Fold names are
    softly tinted by split family so the groups read at a glance.
    """
    from zairachem.base.utils.console import console, heat_hex, themed_table

    is_clf = self.task == "classification"
    y = df[TARGET_COLUMN[self.task]]
    overall = float(y.mean())
    # Normalizer for the colour ramp: the largest deviation a rate can show, or one standard
    # deviation of the target, which is the scale a mean is read against.
    worst = (max(overall, 1.0 - overall) if is_clf else float(y.std())) or 1.0
    table = themed_table("Held-out validation splits")
    for col, just in (
      ("Fold", "left"),
      ("Train", "right"),
      ("Test", "right"),
      ("Test % active" if is_clf else "Test mean", "right"),
    ):
      table.add_column(col, justify=just, no_wrap=True)
    for name, spec in splits.items():
      test_idx = spec["test_idx"]
      if test_idx:
        value = float(y.iloc[test_idx].mean())
        color = heat_hex(abs(value - overall) / worst)
        cell = f"[{color}]{value * 100:.0f}%[/]" if is_clf else f"[{color}]{value:.2f}[/]"
      else:
        cell = "[dim]—[/]"  # an empty test set has nothing to summarize
      hue = self._SPLIT_HUES.get(spec["strategy"], "white")
      table.add_row(
        f"[{hue}]{name}[/]",
        f"[dim]{len(spec['train_idx']):,}[/]",
        f"[dim]{len(test_idx):,}[/]",
        cell,
      )
    console.print(table)

  def setup(self):
    """Run the fit setup: prepare the data, featurizer transformers and isaura store.

    Normalises, standardises, merges, checks and cleans the input, writes held-out splits when
    evaluating, drops featurizers without a reference transformer, checks model images and
    rank references, and seeds the isaura project from the shared lake when a store is set.
    """
    require_docker_and_base()
    self._initialize()
    self._normalize_input()
    self._standardise()
    self._tasks()
    self._merge()
    self._check()
    self._clean()
    self._evaluate_splits()
    # Download every featurizer's reference-library transformer before anything else looks at the
    # descriptors: one without a transformer is dropped from the run here, so the summary, the image
    # check and every later step only see the descriptors that can actually be treated.
    self._keep_featurizers(
      prepare_reference_transformers(
        self.featurizer_ids,
        self.params.get("reference_library", DEFAULT_REFERENCE_LIBRARY),
        self.output_dir,
      )
    )
    self._print_summary()
    validate_model_roles(self.featurizer_ids, self.projection_ids)
    report_model_images(self.featurizer_ids, self.projection_ids)
    # Classification only. lazy-qsar's regression ensemble has no reference library -- its single
    # output is the value, with no rank to position -- so the estimate step never scores these
    # matrices on a regression fit (see estimate/estimators/pipe.py). Downloading them anyway costs
    # up to ~1.6 GB and several minutes for nothing.
    if self.task == "classification":
      prepare_rank_references(self.featurizer_ids)
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
