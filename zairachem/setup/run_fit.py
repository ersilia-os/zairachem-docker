from zairachem.setup.prep.training import TrainSetup
from zairachem.base.utils.logging import logger


def run(input_file, task, output_dir=None, model_ids_file=None):
  ts = TrainSetup(
    input_file,
    output_dir,
    model_ids=model_ids_file,
    task=task,
  )

  if ts.is_done():
    logger.warning("[yellow]Setup for requested inference is already done. Skipping this step![/]")
    return
  ts.setup()
