# ZairaChem — Developer Guide

ZairaChem is an automated QSAR modelling pipeline built on the Ersilia Model Hub. It ships as a flat Python package (`zairachem/`) with a rich-click CLI (`zairachem/cli.py`, entry point `zairachem = zairachem.cli:main`) and requires Docker at runtime.

## Working with the user

- **Ask, don't assume.** For any non-trivial decision (approach, naming, new dependency, ambiguous case) use `AskUserQuestion` BEFORE editing.
- **Plans are mandatory.** Anything beyond a one-line fix or read-only investigation goes through plan mode. Do not skip planning to "save time".
- **Surface uncertainty.** When several options are reasonable, name them and ask.

## Layout and packaging

- Packaging is **Poetry** (`[tool.poetry]` in `pyproject.toml`, `poetry.toml`), not PEP 621.
- Favour submodules (`base/`, `describe/`, `estimate/`, `pool/`, `report/`, ...) over large flat files; keep public APIs small.
- Pin every dependency to an exact `==`-style version. The only deliberate exceptions are the security floors for transitive dependencies (`pillow`, `urllib3`, `idna`), documented in `pyproject.toml`.
- Keep `pyproject.toml` in sync with what `zairachem/` imports.

## Code style

- Run `ruff check` and `ruff format` before every commit. `ruff.toml` is deliberate for this repo: line length 100, 2-space indent.
- Write succinct NumPy-style docstrings for public classes, functions and methods.
- Log through `loguru`; do not add new `logging.getLogger(...)` calls.

## Data

- `data/` is gitignored and eosvc-managed (rank-reference matrices): `eosvc download/upload --path data`. Never commit datasets, model artefacts or large binaries.
- Do not run several `predict` commands concurrently; they share a session symlink.

## README and releases

- Keep the README brief; long-form content belongs in `docs/`. Do not use the package name alone as the H1.
- Versions are semantic (`vMAJOR.MINOR.PATCH`); the git tag, GitHub release and `version` in `pyproject.toml` must match.
