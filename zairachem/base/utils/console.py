"""Shared rich-based terminal output for zairachem.

These helpers print curated, user-facing output to the terminal *regardless* of the logging
verbosity (logging is a separate concern handled by ``zairachem.base.utils.logging``). The
visual vocabulary mirrors the Ersilia CLI: ``✓`` success (green), ``⚠`` warning (yellow),
``✖`` error (red), ``▪`` neutral (dim), and cyan for in-progress status.
"""

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

#: Process-wide console for pretty output. Shared so every step prints consistently.
console = Console()

_ICONS = {
  "success": ("✓", "green"),
  "warning": ("⚠", "yellow"),
  "error": ("✖", "red"),
  "run": ("▪", "cyan"),
  "info": ("▪", "dim"),
}


def echo(text, kind="info"):
  """Print a one-line status message with an Ersilia-style icon and color.

  Parameters
  ----------
  text : str
      Message to display.
  kind : str
      One of ``"success"``, ``"warning"``, ``"error"``, ``"run"`` or ``"info"``.
  """
  icon, style = _ICONS.get(kind, _ICONS["info"])
  console.print(f"  [{style}]{icon}[/] {text}")


def rule(title, *, style="cyan"):
  """Print a horizontal section divider with a left-aligned title.

  Parameters
  ----------
  title : str
      Section heading, shown left-aligned on the rule.
  style : str
      Rich style for the rule and title.
  """
  console.rule(f"[bold {style}]{title}[/]", align="left", style=style)


def summary_panel(title, rows, *, border_style="cyan"):
  """Render a bordered panel wrapping a two-column label/value table.

  Parameters
  ----------
  title : str
      Panel title, shown left-aligned in the border.
  rows : list of tuple
      ``(label, value)`` pairs. A value that is a list/tuple is rendered space-joined and
      wrapped (used for lists of Ersilia model IDs).
  border_style : str
      Rich style for the panel border.
  """
  table = Table(show_header=False, box=None, pad_edge=False, expand=False, padding=(0, 1))
  table.add_column(justify="right", style="bold cyan", no_wrap=True)
  table.add_column(justify="left", overflow="fold")
  for label, value in rows:
    if isinstance(value, (list, tuple)):
      value = "  ".join(str(v) for v in value)
    table.add_row(label, str(value))
  console.print(
    Panel(
      table,
      title=f"[bold cyan]{title}[/]",
      title_align="left",
      border_style=border_style,
      box=box.ROUNDED,
      padding=(1, 2),
      expand=False,
    )
  )
