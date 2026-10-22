"""How to read a descriptor's inner cross-validation report, whichever task produced it.

lazy-qsar writes one ``cv_report.json`` per descriptor, recording its skill as AUROC for a
classifier and as Spearman's rho for a regressor (:data:`zairachem.base.vars.CV_SKILL`). Every
consumer -- the fetcher that feeds the plots, and the two HTML builders -- needs the same two
answers: which key holds the skill, and how to order descriptors by it.
"""

from zairachem.base.vars import CV_SKILL

#: Skill value that means "no skill at all", keyed by the out-of-fold skill key.
CHANCE = {oof: chance for oof, _train, _label, chance in CV_SKILL.values()}


def skill_keys(stats):
  """Return ``(oof_key, train_key, label)`` for a list of ``cv_report.json`` dicts.

  A single report carrying ``oof_rho`` settles it: only a regression run writes that key.
  """
  reg_oof = CV_SKILL["regression"][0]
  task = "regression" if any(s.get(reg_oof) is not None for s in stats) else "classification"
  oof, train, label, _chance = CV_SKILL[task]
  return oof, train, label


def sort_best_first(stats):
  """Sort ``stats`` in place, best descriptor first, on whichever skill the task reports.

  A descriptor whose skill is missing sorts last rather than first, so a failed measurement never
  claims the top identity colour.
  """
  oof_key, _train_key, _label = skill_keys(stats)
  stats.sort(
    key=lambda r: r.get(oof_key) if r.get(oof_key) is not None else -float("inf"), reverse=True
  )
  return stats
