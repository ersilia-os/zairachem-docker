import os

import rich_click as click

from zairachem.base.utils.logging import logger
from zairachem.cli.create_cli import cli
from zairachem.cli.options import (
  _DESCRIBE_WORKERS_HELP,
  _STORE_HELP_PREDICT,
  _STORE_SENTINEL,
  _resolve_store,
  _trained_store,
  common_options,
)
from zairachem.cli.steps import process_group


@cli.command(name="predict", help="Predict activities for new molecules with a trained model.")
# Options render in this order: required I/O (input, model from common_options; then output) →
# run config (store, override, batch, workers) → output controls (skip-report, keep-intermediate
# -data, anonymize). `--anonymize` is declared here (not via common_options) so it sits with the
# other output controls instead of between the required --model-dir and --output-dir.
@common_options(require_input=True, require_model=True)
@click.option(
  "--output-dir", "-o", required=True, help="Directory to write the predictions and report into."
)
@click.option(
  "--store", "-s", is_flag=False, flag_value=_STORE_SENTINEL, default=None, help=_STORE_HELP_PREDICT
)
@click.option(
  "--override",
  "override_dir",
  is_flag=True,
  default=False,
  help="Overwrite the output directory if it already exists (otherwise the run aborts).",
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
  help="Skip the plots and the HTML report. The prediction and performance tables in results/ are "
  "still written.",
)
@click.option(
  "--keep-intermediate-data",
  is_flag=True,
  default=False,
  help="Keep the prediction run's intermediate data (descriptor matrices, projections, input copies, "
  "and the model/ tree). By default a finished predict run keeps only the results and report.",
)
@click.option(
  "--projection-ids",
  "-p",
  "projection_ids",
  required=False,
  is_flag=False,
  flag_value="none",
  default=None,
  help="Projection model ID(s) for the report's 2-D embedding (default: the model's own). Pass "
  "'none' — or a bare -p — to skip them and show only the built-in MW-vs-LogP map. The projection "
  "model is usually the slowest part of a prediction and feeds only a report figure.",
)
@click.option(
  "--anonymize",
  is_flag=True,
  help="Blank out molecule structures (SMILES / InChIKey) in all outputs.",
)
def predict(
  input_file,
  model_dir,
  anonymize,
  output_dir,
  store,
  projection_ids,
  override_dir,
  batch_size,
  describe_workers,
  keep_intermediate_data,
  no_report,
):
  """Predict activities for new molecules with a trained model."""
  from zairachem.setup.run_predict import run as run_predict

  logger.configure()
  logger.info("[#ff69b4]Running the setup pipeline to preprocess the input data for prediction[/]")
  if batch_size:
    logger.info(f"[#ff69b4]Using batch size: {batch_size}[/]")
  # Default to the store the model was trained with (reuse its cache) — whether --store is omitted OR
  # given bare. An explicit --store NAME overrides; --store isaura-public uses the shared lake.
  trained = _trained_store(model_dir)
  if store is None or store == _STORE_SENTINEL:
    store = trained
  else:
    store = _resolve_store(store, trained)
  # run_predict validates the model folder and prints the run header only once it's confirmed ready.
  proceed = run_predict(
    input_file,
    model_dir,
    output_dir,
    override_dir,
    store=store,
    projection_ids=projection_ids,
  )
  if proceed:
    process_group(
      os.path.abspath(output_dir),
      anonymize,
      batch_size=batch_size,
      keep_intermediate_data=keep_intermediate_data,
      no_report=no_report,
      describe_workers=describe_workers,
    )
  else:
    from zairachem.base.utils.progress import final_summary_panel

    final_summary_panel(os.path.abspath(output_dir))
