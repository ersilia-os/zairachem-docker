from zairachem.setup.prep.training import TrainSetup


def run(input_file, output_dir=None, threshold=None, direction=None, parameters=None):
  ts = TrainSetup(
    input_file,
    output_dir,
    time_budget=120,
    task="classification",
    threshold=threshold,
    direction=direction,
    parameters=parameters,
  )
  if ts.is_done():
    return
  ts.setup()
