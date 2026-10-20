import os

import rich_click as click
from click.core import ParameterSource

from zairachem.base.utils.logging import logger
from zairachem.base.vars import DEFAULT_MAX_DESCRIPTORS
from zairachem.cli.create_cli import cli
from zairachem.cli.options import (
  _DESCRIBE_WORKERS_HELP,
  _EVALUATE_SENTINEL,
  _STORE_HELP,
  _STORE_SENTINEL,
  _resolve_evaluate,
  _resolve_store,
  common_options,
)
from zairachem.cli.steps import process_group


@cli.command(name="fit", help="Train a QSAR model from a labelled CSV.")
# Required I/O (input, model from common_options) → task / featurizers → run config (store, override,
# batch, workers) → output controls (skip-report, keep-intermediate-data, anonymize). `--anonymize`
# is declared here (not via common_options) so it sits with the other output controls.
@common_options(require_input=True, include_task=True, include_eos=True, require_model=True)
@click.option(
  "--store", "-s", is_flag=False, flag_value=_STORE_SENTINEL, default=None, help=_STORE_HELP
)
@click.option(
  "--override",
  "override_dir",
  is_flag=True,
  default=False,
  help="Wipe and rebuild the model directory if it already exists. Without this flag, an unfinished "
  "run in that directory is resumed; a completed model aborts the run.",
)
@click.option(
  "--batch-size",
  "-b",
  default=None,
  type=int,
  help="Rows per chunk when processing large datasets (default: 10000).",
)
@click.option("--workers", "describe_workers", default=None, type=int, help=_DESCRIBE_WORKERS_HELP)
@click.option(
  "--skip-report",
  "no_report",
  is_flag=True,
  default=False,
  help="Skip the plots and the HTML report (the slow, bulky part). The prediction and performance "
  "tables in results/ are still written.",
)
@click.option(
  "--keep-intermediate-data",
  is_flag=True,
  default=False,
  help="Keep intermediate data (descriptor matrices, 2-D projections, raw input copies) instead of "
  "cleaning it at the end. The trained model and results/report are always kept either way.",
)
@click.option(
  "--anonymize",
  is_flag=True,
  help="Blank out molecule structures (SMILES / InChIKey) in all outputs.",
)
@click.option(
  "--evaluate",
  is_flag=False,
  flag_value=_EVALUATE_SENTINEL,
  default=None,
  help="Run held-out validation: define 80:20 splits at setup and, after the final model is trained, "
  "re-fit + score the pooled model on each held-out fold (adds a report section; classification only). "
  "Bare --evaluate runs all schemas (random, scaffold, scaffold_det, butina); or pass a comma-separated "
  "subset, e.g. --evaluate scaffold,random.",
)
@click.option(
  "--repeats",
  "evaluate_repeats",
  default=3,
  type=int,
  help="Held-out repeats per schema (random/scaffold/Butina) when --evaluate is set (default: 3). "
  "Total folds = 1 + 3 × repeats (so --repeats 0 leaves only the deterministic scaffold_det). Lower "
  "it to cut compute proportionally — each fold still refits the exact production pipeline.",
)
@click.option(
  "--max-descriptors",
  "max_descriptors",
  default=DEFAULT_MAX_DESCRIPTORS,
  type=int,
  help=f"Cap the ensemble at the top-K descriptors (default: {DEFAULT_MAX_DESCRIPTORS}). They are "
  "pre-screened by mean held-out "
  "AUROC across the chemistry-aware splits (shallow proxy model, floor 0.55) and only the top-K are "
  "fully trained — cutting training cost and yielding a leaner model. Set it ≥ the number of "
  "descriptors you provide (e.g. 10) to train them all. Classification only; the hard cap is 10.",
)
def fit(
  input_file,
  classification,
  model_dir,
  featurizer_ids,
  projection_ids,
  anonymize,
  store,
  override_dir,
  batch_size,
  keep_intermediate_data,
  no_report,
  describe_workers,
  evaluate,
  evaluate_repeats,
  max_descriptors,
):
  """Train a QSAR model from a labelled CSV, then run every pipeline step."""
  from zairachem.setup.run_fit import run as run_fit

  logger.configure()
  logger.info("[#ff69b4]Running the setup pipeline to preprocess the input data[/]")
  if batch_size:
    logger.info(f"[#ff69b4]Using batch size: {batch_size}[/]")
  if classification:
    task = "classification"
  else:
    task = "regression"
  from zairachem.base.utils.progress import tracker

  # --evaluate is an optional-value flag: off / bare (all schemas) / a comma-separated subset.
  evaluate_on, evaluate_schemas = _resolve_evaluate(evaluate)
  store = _resolve_store(store, os.path.basename(os.path.normpath(model_dir)))
  # Config-affecting flags the user actually typed (vs click defaults). On resume, these are checked
  # against the trained model's on-disk config so a conflicting re-run aborts instead of silently
  # mixing settings; un-typed flags inherit the on-disk values.
  ctx = click.get_current_context()
  explicit_config = {
    name
    for name in (
      "classification",
      "featurizer_ids",
      "projection_ids",
      "store",
      "evaluate",
      "evaluate_repeats",
      "max_descriptors",
    )
    if ctx.get_parameter_source(name) == ParameterSource.COMMANDLINE
  }
  # The optional screen (--max-descriptors) and holdout (--evaluate) steps are only in the pipeline
  # when their flags are set; drop them from the tracker order otherwise so step numbering and the run
  # header reflect what actually runs.
  from zairachem.base.utils.pipeline_tracker import PIPELINE_STEPS

  steps = [
    k
    for k, _, _ in PIPELINE_STEPS
    if (k != "holdout" or evaluate_on) and (k != "screen" or (max_descriptors and classification))
  ]
  tracker.begin(
    "ZairaChem · QSAR training",
    subtitle=f"{os.path.basename(input_file)} → {os.path.basename(os.path.normpath(model_dir))} · {task}",
    steps=steps,
  )
  proceed = run_fit(
    input_file,
    task,
    store,
    output_dir=model_dir,
    model_ids_file=featurizer_ids,
    projection_ids=projection_ids,
    override=override_dir,
    explicit_config=explicit_config,
    evaluate=evaluate_on,
    evaluate_repeats=evaluate_repeats,
    evaluate_schemas=evaluate_schemas,
    max_descriptors=max_descriptors,
  )
  if proceed:
    process_group(
      os.path.abspath(model_dir),
      anonymize,
      batch_size=batch_size,
      keep_intermediate_data=keep_intermediate_data,
      no_report=no_report,
      describe_workers=describe_workers,
    )
  else:
    from zairachem.base.utils.progress import final_summary_panel

    final_summary_panel(os.path.abspath(model_dir))
