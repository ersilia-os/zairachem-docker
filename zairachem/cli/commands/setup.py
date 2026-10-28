import os

import rich_click as click

from zairachem.base.utils.logging import logger
from zairachem.cli.create_cli import cli
from zairachem.cli.options import (
  _STORE_HELP,
  _STORE_SENTINEL,
  _explicit_config,
  _resolve_store,
  _resolve_task,
  common_options,
)


@cli.command(name="setup", help="Standardize and prepare the input molecules.")
@common_options(require_input=True, include_task=True, include_eos=True)
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
def setup_cmd(
  input_file,
  classification,
  model_dir,
  featurizer_ids,
  projection_ids,
  store,
  override_dir,
):
  """Standardize and prepare the input molecules (the first pipeline step)."""
  from zairachem.setup.run_fit import run as run_fit

  logger.configure()
  logger.info("[#ff69b4]Running the setup pipeline to preprocess the input data[/]")
  task = _resolve_task(classification, input_file)
  _model = model_dir or os.path.splitext(os.path.basename(input_file))[0]
  store = _resolve_store(store, os.path.basename(os.path.normpath(_model)))
  # Hand over the typed flags, as `fit` does: without them a resumed run cannot tell that a new -f
  # or -c differs from the unfinished run's, and silently carries on with the old settings.
  ok = run_fit(
    input_file,
    task,
    store,
    output_dir=model_dir,
    model_ids_file=featurizer_ids,
    projection_ids=projection_ids,
    override=override_dir,
    explicit_config=_explicit_config(click.get_current_context()),
  )
  if not ok:
    raise SystemExit(1)
