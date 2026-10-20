import os

import rich_click as click

from zairachem.base.utils.logging import logger
from zairachem.cli.create_cli import cli
from zairachem.cli.options import _STORE_HELP, _STORE_SENTINEL, _resolve_store, common_options


@cli.command(name="setup", help="Standardize and prepare the input molecules.")
@common_options(require_input=True, include_task=True, include_eos=True)
@click.option(
  "--store", "-s", is_flag=False, flag_value=_STORE_SENTINEL, default=None, help=_STORE_HELP
)
def setup_cmd(
  input_file,
  classification,
  model_dir,
  featurizer_ids,
  projection_ids,
  store,
):
  """Standardize and prepare the input molecules (the first pipeline step)."""
  from zairachem.setup.run_fit import run as run_fit

  logger.configure()
  logger.info("[#ff69b4]Running the setup pipeline to preprocess the input data[/]")
  if classification:
    task = "classification"
  else:
    task = "regression"
  _model = model_dir or os.path.splitext(os.path.basename(input_file))[0]
  store = _resolve_store(store, os.path.basename(os.path.normpath(_model)))
  run_fit(
    input_file,
    task,
    store,
    output_dir=model_dir,
    model_ids_file=featurizer_ids,
    projection_ids=projection_ids,
  )
