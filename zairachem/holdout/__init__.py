"""Held-out validation: chemistry-aware train/test splits and their evaluation.

Splits are *defined* at setup (:mod:`zairachem.holdout.splits`) and written to
``metadata/splits.json``; when ``--evaluate`` is set the folds are *executed* after the pool step,
each re-fitting the estimators + pooler on its train slice and scoring the held-out slice. The final
production model (trained on all rows) is unaffected.
"""
