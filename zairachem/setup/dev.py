"""
from zairasetup.setup.prediction import PredictSetup
cf = PredictSetup("../../tests/osm_bin_ten.csv", "../../tests/prediction", "../../tests/demo")
cf.setup()
"""

from setup.prep.training import TrainSetup

cf = TrainSetup(
  "../input.csv",
  "../tests/fit",
  time_budget=120,
  task="classification",
  threshold=None,
  direction=None,
  parameters=None,
)
cf.setup()
