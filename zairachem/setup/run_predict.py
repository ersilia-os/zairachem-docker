from zairachem.setup.prep.prediction import PredictSetup
from zairachem.base import create_session_symlink
from zairachem.base.utils.console import echo
from zairachem.base.utils.progress import tracker, summarize_setup


def run(
  input_file,
  model_dir,
  output_dir=None,
  override_dir=False,
  store_read=False,
  nn=False,
  store_write=False,
):
  ps = PredictSetup(
    input_file,
    model_dir,
    output_dir,
    override_dir,
    time_budget=120,
    store_read=store_read,
    nn=nn,
    store_write=store_write,
  )
  if ps.is_done():
    create_session_symlink(ps.output_dir)
    echo(
      f"Predictions already exist at {ps.output_dir}. Use --override-dir or a different -o to redo.",
      kind="warning",
    )
    return False
  tracker.start("setup")
  ps.setup()
  tracker.complete("setup", summarize_setup(ps.output_dir))
  return True
