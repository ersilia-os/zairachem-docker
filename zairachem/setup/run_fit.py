import os

from zairachem.setup.prep.training import TrainSetup
from zairachem.setup.prep import PipelineStep
from zairachem.base import create_session_symlink
from zairachem.base.utils.console import echo
from zairachem.base.utils.progress import tracker, summarize_setup


def run(
  input_file,
  task,
  store=None,
  output_dir=None,
  model_ids_file=None,
  projection_ids=None,
  override=False,
  explicit_config=None,
  evaluate=False,
  evaluate_repeats=3,
  evaluate_schemas=None,
):
  ts = TrainSetup(
    input_file,
    output_dir,
    model_ids=model_ids_file,
    task=task,
    store=store,
    projection_ids=projection_ids,
    evaluate=evaluate,
    evaluate_repeats=evaluate_repeats,
    evaluate_schemas=evaluate_schemas,
  )

  # Decide fresh / resume / refuse for an existing model directory (a fresh run on an absent dir, or
  # any --override run, skips straight to setup which wipes + rebuilds).
  resume = False
  if os.path.exists(ts.output_dir) and not override:
    create_session_symlink(ts.output_dir)
    if ts.is_done():
      # A complete model: re-training needs an explicit --override.
      echo(
        f"'{ts.output_dir}' is already a trained model. Pass --override to retrain it, or choose a "
        f"different -m.",
        kind="error",
      )
      return False
    if PipelineStep("initialize", ts.output_dir).is_done():
      # A partial run that got past initialize: resume it (keep artifacts + the session steps list).
      resume = True
    # else: a dir that never finished initialize is an inconsistent leftover with nothing worth
    # keeping — fall through to setup(), whose fresh path wipes + rebuilds it cleanly.

  ts.resume = resume
  if resume:
    conflict = ts.resume_config_conflict(explicit_config or set())
    if conflict:
      echo(conflict, kind="error")
      return False
    echo(f"Resuming the unfinished run in '{ts.output_dir}'.", kind="info")
  tracker.start("setup")
  ts.setup()
  tracker.complete("setup", summarize_setup(ts.output_dir))
  return True
