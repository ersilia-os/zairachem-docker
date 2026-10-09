# Contributing

Open an issue before starting anything larger than a small fix, so we can agree on the approach.

1. Branch from `main` (`fix/...`, `feat/...`, `docs/...`) and open a pull request against it.
2. Set up the environment with `./install.sh`; it installs the versions pinned in `constraints.txt`.
3. Before pushing, run `ruff check` and `ruff format`, and `pytest`.
4. Keep changes focused. Say in the pull request what you ran, and what you could not run (a full `fit` needs Docker and the Ersilia model images).

Code conventions (NumPy docstrings, exact dependency pins, logging, data handling) are in [`CLAUDE.md`](CLAUDE.md).
When you change dependencies, update `pyproject.toml` and regenerate `constraints.txt` with `python scripts/make_constraints.py`.
