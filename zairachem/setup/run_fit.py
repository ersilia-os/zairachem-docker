from zairachem.setup.prep.training import TrainSetup
from zairachem.base import create_session_symlink
from zairachem.base.utils.console import echo
from zairachem.base.utils.progress import tracker, summarize_setup


def run(
  input_file,
  task,
  store_read=False,
  nn=False,
  store_write=False,
  output_dir=None,
  model_ids_file=None,
):
  ts = TrainSetup(
    input_file,
    output_dir,
    model_ids=model_ids_file,
    task=task,
    store_read=store_read,
    nn=nn,
    store_write=store_write,
  )

  if ts.is_done():
    # Already a complete model. Point the global session at THIS model (so any subsequent display
    # reflects it, not a stale run) and stop — the caller skips the rest of the pipeline.
    create_session_symlink(ts.output_dir)
    echo(
      f"Model already trained at {ts.output_dir}. Use --flush or a different -m to retrain.",
      kind="warning",
    )
    return False
  tracker.start("setup")
  ts.setup()
  tracker.complete("setup", summarize_setup(ts.output_dir))
  return True
