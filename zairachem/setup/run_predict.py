from zairachem.setup.prep.prediction import PredictSetup
from zairachem.base.utils.logging import logger


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
    logger.warning(
      "[yellow]Prediction setup for requested inferece is already done. Skippign this step![/]"
    )
    return
  ps.setup()
