

from zairasetup.setup.prediction import PredictSetup
cf = PredictSetup("../../tests/osm_bin.csv", "../../tests/prediction2", "../../tests/demo2")
cf.setup()
"""

from zairasetup.setup.training import TrainSetup
cf = TrainSetup("../../tests/osm_bin.csv", "../../tests/demo3", time_budget=120, task="classification", threshold=None, direction=None, parameters=None)
cf.setup()
"""