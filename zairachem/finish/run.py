from zairachem.finish.finish import Finisher


def run(path=None, anonymize=False):
  r = Finisher(path, anonymize=anonymize)
  r.run()
