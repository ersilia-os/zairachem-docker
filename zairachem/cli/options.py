import os

import rich_click as click
from click.core import ParameterSource

# `--store` is an optional-value option: omitted -> None (no store); `--store` alone -> this
# sentinel (resolve to a default name); `--store X` -> the literal X.
_STORE_SENTINEL = "\x00default"


def _resolve_store(value, default):
  """Resolve the --store value to a concrete isaura project name, or None for 'no store'.

  ``value`` is None (omitted), the sentinel (bare ``--store`` → use ``default``), or an explicit
  name. The result is sanitized to a valid bucket name; ``isaura-public`` passes through unchanged.
  """
  if value is None:
    return None
  from zairachem.base.utils.isaura_report import sanitize_project_name

  name = default if value == _STORE_SENTINEL else value
  return sanitize_project_name(name) if name else None


def _resolve_task(flag, input_file):
  """Resolve ``--classification/--regression`` to ``"classification"`` or ``"regression"``.

  ``flag`` is True (``-c``), False (``-r``) or None (not typed: detect it from the target column,
  read whole). A continuous target with ``-c``, or two values that are not 0/1, stops here with what
  to do instead, rather than being rounded to 0/1 later. A run with no readable target falls back
  to classification, and setup reports what is wrong with the input.
  """
  if flag is False:
    return "regression"
  import pandas as pd

  from zairachem.setup.prep.schema import InputSchema, not_binary_message, target_kind

  try:
    column = InputSchema(input_file).resolve_columns()["values_column"]
    values = pd.read_csv(input_file, usecols=[column])[column] if column else None
  except (AssertionError, OSError, ValueError):
    return "classification"
  if values is None:
    return "classification"
  kind, distinct = target_kind(pd.to_numeric(values, errors="coerce"))
  if kind == "binary":
    task, why = "classification", f"'{column}' has 0/1 labels"
  elif flag is None and kind == "continuous":
    task, why = "regression", f"'{column}' is continuous, {len(distinct):,} distinct values"
  else:
    raise click.UsageError(not_binary_message(column, kind, distinct))
  if flag is None:
    from zairachem.base.utils.console import echo

    echo(f"Task: {task} (detected: {why}).", kind="info")
  return task


# `--evaluate` is an optional-value option: omitted -> off; bare `--evaluate` -> this sentinel (all
# schemas); `--evaluate scaffold,random` -> that subset.
_EVALUATE_SENTINEL = "\x00all"


def _resolve_evaluate(value):
  """Resolve the --evaluate value into ``(enabled, schemas)``.

  ``None`` → ``(False, None)`` (off); the sentinel (bare ``--evaluate``) → ``(True, None)`` (all
  schemas); a comma/space-separated list → ``(True, [names])`` validated against the selectable fold
  schemas. Raises ``click.BadParameter`` on an unknown schema name.
  """
  if value is None:
    return False, None
  if value == _EVALUATE_SENTINEL:
    return True, None
  import re

  from zairachem.holdout.splits import FOLD_SCHEMAS

  names = [t for t in re.split(r"[,\s]+", str(value).strip()) if t]
  bad = [t for t in names if t not in FOLD_SCHEMAS]
  if not names or bad:
    raise click.BadParameter(
      f"'{value}': unknown split schema(s): {', '.join(bad) or '(empty)'}. Choose from "
      f"{', '.join(FOLD_SCHEMAS)} (comma-separated), or pass --evaluate alone for all.",
      param_hint="--evaluate",
    )
  return True, names


def _explicit_config(ctx):
  """Return the names of the config-affecting parameters typed on the command line."""
  # Config-affecting flags the user actually typed (vs click defaults). On resume, these are checked
  # against the trained model's on-disk config so a conflicting re-run aborts instead of silently
  # mixing settings; un-typed flags inherit the on-disk values. A command that lacks one of these
  # options simply never reports it (``get_parameter_source`` is None for an unknown name).
  return {
    name
    for name in (
      "classification",
      "featurizer_ids",
      "projection_ids",
      "store",
      "evaluate",
      "evaluate_repeats",
      "max_descriptors",
    )
    if ctx.get_parameter_source(name) == ParameterSource.COMMANDLINE
  }


def _trained_store(model_dir):
  """The store name persisted in a trained model's parameters.json (for predict reuse), or None."""
  if not model_dir:
    return None
  import json

  from zairachem.base.vars import METADATA_SUBFOLDER, PARAMETERS_FILE

  try:
    with open(os.path.join(model_dir, METADATA_SUBFOLDER, PARAMETERS_FILE)) as f:
      p = json.load(f)
    return p.get("store") or p.get("contribute_store") or p.get("read_store")
  except Exception:
    return None


_STORE_HELP = (
  "Cache descriptor precalculations in an isaura project to speed up re-runs. Omit for no store; "
  "bare --store uses the model-directory name; --store NAME uses NAME; --store isaura-public "
  "reads/writes the shared public lake directly."
)

_STORE_HELP_PREDICT = (
  "isaura store to read/write. Defaults to the store the model was trained with (omit or bare "
  "--store); --store NAME overrides; --store isaura-public uses the shared public lake."
)

_DESCRIBE_WORKERS_HELP = (
  "Most descriptor models to featurize at once (default: the Docker VM's CPUs). Models are also "
  "admitted only while their measured memory and CPU fit in the VM."
)


def common_options(
  require_input: bool = True,
  include_task: bool = False,
  include_eos: bool = False,
  require_model: bool = False,
):
  """Build the click decorator that adds the options shared by the pipeline commands.

  Parameters
  ----------
  require_input : bool
    Make ``--input-file`` required.
  include_task : bool
    Add the ``--classification/--regression`` option.
  include_eos : bool
    Add the ``--featurizer-ids`` and ``--projection-ids`` options.
  require_model : bool
    Make ``--model-dir`` required.

  Returns
  -------
  callable
    Decorator to apply to a click command.
  """

  def _decorator(func):
    options = _common_option_decorators(require_input, include_task, include_eos, require_model)
    for option in reversed(options):
      func = option(func)
    return func

  return _decorator


def _common_option_decorators(require_input, include_task, include_eos, require_model):
  """Return the click option decorators of :func:`common_options`, in display order."""
  options = [
    click.option(
      "--input-file",
      "-i",
      required=require_input,
      type=click.Path(exists=True, dir_okay=False),
      help="Input CSV of molecules (must contain a SMILES column).",
    ),
    click.option(
      "--model-dir",
      "-m",
      required=require_model,
      default=None,
      help="Model directory — created here when fitting; the trained model to use when predicting.",
    ),
  ]
  # Build in display order: required I/O (input, model) first, then task / featurizer-ids /
  # projection-ids. (Commands that want an --anonymize flag declare it themselves, grouped with
  # their own output-control flags.)
  if include_task:
    options.append(
      click.option(
        "--classification/--regression",
        "-c/-r",
        default=None,
        help="Model type. Detected from the target when omitted: 0/1 labels are classification, a "
        "continuous target is regression.",
      )
    )
  if include_eos:
    options.append(
      click.option(
        "--featurizer-ids",
        "-f",
        "featurizer_ids",
        required=False,
        default=None,
        help="Descriptor model IDs, comma-separated (e.g. eos8aa5,eos3l5f), or a JSON file with featurizer_ids/projection_ids keys. Default: a curated descriptor set.",
      )
    )
    options.append(
      click.option(
        "--projection-ids",
        "-p",
        "projection_ids",
        required=False,
        is_flag=False,
        flag_value="none",
        default=None,
        help="Projection model ID(s) for the report's 2-D embedding (default: eos1klk). Pass "
        "'none' — or a bare -p — to skip Ersilia projections and show only the built-in MW-vs-LogP "
        "map. Overrides any projection_ids in a --featurizer-ids JSON file.",
      )
    )
  return options


_MODEL_DIR_HELP = (
  "Model directory to run this step against. Re-running a step re-executes it (e.g. re-render the "
  "report after editing a plot) without redoing the rest."
)
