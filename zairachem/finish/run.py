from zairachem.finish.finish import Finisher


def run(path=None, clean=False, flush=False, anonymize=False):
  r = Finisher(path, clean, flush, anonymize)
  r.run()
