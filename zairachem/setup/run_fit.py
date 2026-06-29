import os

from zairachem.setup.prep.training import TrainSetup
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
):
  ts = TrainSetup(
    input_file,
    output_dir,
    model_ids=model_ids_file,
    task=task,
    store=store,
    projection_ids=projection_ids,
  )

  # Setup wipes and rebuilds the model directory, so refuse to clobber an existing one unless
  # --override is given. (A complete model gets a clearer message; a partial leftover just "exists".)
  if os.path.exists(ts.output_dir) and not override:
    create_session_symlink(ts.output_dir)
    what = "is already a trained model" if ts.is_done() else "already exists"
    echo(
      f"'{ts.output_dir}' {what}. Pass --override to overwrite it, or choose a different -m.",
      kind="error",
    )
    return False
  tracker.start("setup")
  ts.setup()
  tracker.complete("setup", summarize_setup(ts.output_dir))
  return True
