# Contributing

Open an issue before starting anything larger than a small fix, so we can agree on the approach.

1. Branch from `main` (`fix/...`, `feat/...`, `docs/...`) and open a pull request against it.
2. Set up the environment with `./install.sh`; it installs the exact versions pinned in `pyproject.toml`.
3. Before pushing, run `ruff check` and `ruff format`, and `pytest` (see *Tests* below).
4. Keep changes focused. Say in the pull request what you ran, and what you could not run (a full `fit` needs Docker and the Ersilia model images).

Code conventions (NumPy docstrings, exact dependency pins, logging, data handling) are in [`CLAUDE.md`](CLAUDE.md).
When you change dependencies, update the exact pins in `pyproject.toml`.

## Tests

`pytest` is deliberately small: it guards behaviour, not implementation. Add a test only when you
can name the bug it catches — reintroduce that bug first and check nothing else already fails.

It does not pin the numbers a run produces. That is `tests/test_golden.py`, which re-runs the
modelling steps on a kept fit and compares every column against a recorded reference. It is
opt-in (the fixture is ~25 MB and not in git; the module docstring says how to record one). Run it
against both the classification and the regression fixture for any change to the modelling path,
and say in the pull request that you did.
