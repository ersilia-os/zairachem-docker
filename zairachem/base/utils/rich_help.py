"""Align the panels on the top-level ``zairachem --help`` screen.

rich-click sizes the *Options* panel's columns to their content (so its help text sits at a
fixed column) but sizes the *command* panels by a width ratio (so their help text drifts as the
terminal widens). The two therefore only line up at one specific terminal width.

This module renders the top-level group's Options panel as the same two-column ``[name | help]``
layout as the command panels and pins the first column of *both* to a shared width — the widest
option label or command name — so every panel's help text starts at the same column at any
terminal width. Only the top-level group is affected (via ``StatusGroupMixin``); every
subcommand's ``--help`` keeps rich-click's default multi-column option rendering.

Implemented against the pinned rich-click 1.8.9. ``get_rich_options`` builds its table by relying
on ``add_row`` to auto-create columns, and ``get_rich_commands`` adds two ratio columns
explicitly; we swap in ``rich.table.Table`` subclasses for the duration of each call to reshape
those tables without reimplementing rich-click's option-grouping logic.
"""

import contextlib

import rich_click as click
import rich_click.rich_help_rendering as _rendering
from rich.table import Table as _Table
from rich.text import Text

# Set by StatusGroupMixin.format_options for the duration of a single help render, then cleared.
# The table subclasses below read it at construction time.
_shared_width = None


class _MergedOptionsTable(_Table):
  """Options table collapsed to ``[name | help]``.

  rich-click passes each option as several leading cells (required marker, long form, short form,
  metavar) plus the help text. We merge the leading cells into a single first column pinned to the
  shared width, so the panel matches the command panels' two-column shape.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.add_column(no_wrap=True, width=_shared_width)
    self.add_column(no_wrap=False, ratio=1)

  def add_row(self, *cells, **kwargs):
    *lead, help_cell = cells
    name = Text()
    for cell in lead:
      if cell is None:
        continue
      text = cell if isinstance(cell, Text) else Text(str(cell))
      if not text.plain.strip():
        continue
      if name.plain:
        name.append(" ")
      name.append_text(text)
    return super().add_row(name, help_cell, **kwargs)


class _FixedFirstColumnTable(_Table):
  """Commands table whose first column is pinned to the shared width instead of the config ratio,
  so command names line up with the merged option labels."""

  def add_column(self, *args, **kwargs):
    if not self.columns:
      kwargs.pop("ratio", None)
      kwargs["width"] = _shared_width
    else:
      kwargs.pop("ratio", None)
      kwargs.setdefault("ratio", 1)
    return super().add_column(*args, **kwargs)


@contextlib.contextmanager
def _swap_table(table_cls):
  """Temporarily point rich-click's rendering module at ``table_cls`` for ``Table(...)`` calls."""
  original = _rendering.Table
  _rendering.Table = table_cls
  try:
    yield
  finally:
    _rendering.Table = original


def _option_label_len(param, ctx):
  """Plain width of the merged option label, matching ``_MergedOptionsTable.add_row``.

  Mirrors how rich-click builds the leading cells: a ``*`` required marker, the comma-joined long
  forms, the comma-joined short forms, and the metavar (for value-taking options), joined by single
  spaces.
  """
  longs, shorts = [], []
  for idx, opt in enumerate(param.opts):
    text = opt
    if idx < len(param.secondary_opts):
      text += "/" + param.secondary_opts[idx]
    (longs if opt.startswith("--") else shorts).append(text)

  parts = []
  if param.required:
    parts.append("*")
  if longs:
    parts.append(",".join(longs))
  if shorts:
    parts.append(",".join(shorts))
  if not getattr(param, "is_flag", False):
    try:
      metavar = param.make_metavar(ctx)
    except TypeError:  # older click: no ctx argument
      metavar = param.make_metavar()
    if metavar and metavar != "BOOLEAN":
      parts.append(metavar)
  return len(" ".join(parts))


def _compute_shared_width(group, ctx):
  """Widest option label or command name in the group's help — the shared first-column width."""
  widths = [len(name) for name in group.list_commands(ctx)]
  for param in group.get_params(ctx):
    if getattr(param, "hidden", False) or isinstance(param, click.Argument):
      continue
    widths.append(_option_label_len(param, ctx))
  return max(widths) if widths else None


class StatusGroupMixin:
  """Mixin for the top-level ``click.RichGroup`` that renders its Options and command panels with a
  shared first-column width so their help text lines up at any terminal width."""

  def format_options(self, ctx, formatter):
    from rich_click.rich_help_rendering import get_rich_commands, get_rich_options

    global _shared_width
    _shared_width = _compute_shared_width(self, ctx)
    try:
      with _swap_table(_MergedOptionsTable):
        get_rich_options(self, ctx, formatter)
      with _swap_table(_FixedFirstColumnTable):
        get_rich_commands(self, ctx, formatter)
    finally:
      _shared_width = None
