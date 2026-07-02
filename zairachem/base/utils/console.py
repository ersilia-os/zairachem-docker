"""Shared rich-based terminal output for zairachem.

These helpers print curated, user-facing output to the terminal *regardless* of the logging
verbosity (logging is a separate concern handled by ``zairachem.base.utils.logging``).

Design goals: sleek and dense. A run shows only a few bordered *panels* (run header, setup facts,
final summary); everything else is rendered as themed **section rules**, **borderless tables**, and
**aligned key/value lines**, so the output reads as one clean, information-rich stream rather than a
stack of boxes. Each pipeline step has an accent colour (``progress.STEP_COLORS``); the active colour
is set via :func:`set_active_color` when a step starts and the helpers below default to it.
"""

from rich import box
from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table

#: Process-wide console for pretty output. Shared so every step prints consistently.
#: ``highlight=False``: all colour comes from our explicit markup/themes, not rich's automatic
#: number/string highlighter (which speckles digits cyan and fights the deliberate step palette).
console = Console(highlight=False)

#: Accent colour of the step currently running (set by the tracker); falls back to cyan.
_active_color = "cyan"


def set_active_color(color):
  """Set the accent colour used by the themed helpers (called when a step starts)."""
  global _active_color
  _active_color = color or "cyan"


def active_color():
  """The accent colour of the currently running step (cyan when no step is active)."""
  return _active_color


_ICONS = {
  "success": ("✓", "green"),
  "warning": ("⚠", "yellow"),
  "error": ("✖", "red"),
  "run": ("▪", "cyan"),
  "info": ("·", "dim"),
}


def echo(text, kind="info"):
  """Print a one-line status message with an Ersilia-style icon and color."""
  icon, style = _ICONS.get(kind, _ICONS["info"])
  console.print(f"  [{style}]{icon}[/] {text}")


def rule(title, *, style=None, right=None):
  """Print a sleek left-aligned section divider, themed to the active step.

  Parameters
  ----------
  title : str
      Section heading (already-marked-up text is fine).
  style : str, optional
      Rule/title colour; defaults to the active step's accent colour.
  right : str, optional
      Optional dim text shown after the title (e.g. ``"step 3/7"``).
  """
  style = style or active_color()
  label = f"[bold {style}]{title}[/]"
  if right:
    label += f"   [dim]{right}[/]"
  console.rule(label, align="left", style=style)


def detail(rows, *, color=None, indent=3):
  """Print borderless, aligned ``label → value`` lines (themed) — step detail with counts.

  ``rows`` is a list of ``(label, value)``; a list/tuple value is space-joined. Labels are dim and
  right-aligned; values use the active/passed colour's normal weight. No border — reads as content.
  """
  rows = [r for r in rows if r is not None]
  if not rows:
    return
  table = Table(show_header=False, box=None, pad_edge=False, padding=(0, 2))
  table.add_column(justify="right", style="dim", no_wrap=True)
  table.add_column(justify="left", overflow="fold")
  for label, value in rows:
    if isinstance(value, (list, tuple)):
      value = "  ".join(str(v) for v in value)
    table.add_row(label, str(value))
  console.print(Padding(table, (1, 0, 0, indent)))


def themed_table(title, *, color=None, caption=None):
  """Create a sleek borderless (header-underline only) table, themed to the active step."""
  color = color or active_color()
  return Table(
    title=f"[bold {color}]{title}[/]" if title else None,
    title_justify="left",
    caption=caption,
    caption_justify="left",
    box=box.SIMPLE_HEAD,
    border_style=color,
    header_style=f"bold {color}",
    pad_edge=False,
    expand=False,
  )


def heat_hex(t):
  """Map ``t`` in ``[0, 1]`` to a hex colour on a green→amber→red ramp.

  Uses the report's GitHub-ish palette (green ``#3fb950`` → amber ``#d29922`` → red ``#f85149``) so
  terminal-shaded magnitudes/deviations read the same as the HTML report. Handy for tinting a numeric
  column in a :func:`themed_table` (0 = calm/good, 1 = attention).
  """
  t = 0.0 if t < 0 else 1.0 if t > 1 else float(t)
  green, amber, red = (0x3F, 0xB9, 0x50), (0xD2, 0x99, 0x22), (0xF8, 0x51, 0x49)
  (a, b), u = ((green, amber), t / 0.5) if t < 0.5 else ((amber, red), (t - 0.5) / 0.5)
  r, g, bl = (round(a[i] + (b[i] - a[i]) * u) for i in range(3))
  return f"#{r:02x}{g:02x}{bl:02x}"


def summary_panel(title, rows, *, border_style=None, icon=None):
  """Render a bordered panel wrapping a two-column label/value table (used sparingly).

  Defaults its border/label colour to the active step's accent colour.
  """
  color = border_style or active_color()
  table = Table(show_header=False, box=None, pad_edge=False, expand=False, padding=(0, 1))
  table.add_column(justify="right", style=f"bold {color}", no_wrap=True)
  table.add_column(justify="left", overflow="fold")
  for label, value in rows:
    if isinstance(value, (list, tuple)):
      value = "  ".join(str(v) for v in value)
    table.add_row(label, str(value))
  heading = f"{icon}  {title}" if icon else title
  console.print(
    Panel(
      table,
      title=f"[bold {color}]{heading}[/]",
      title_align="left",
      border_style=color,
      box=box.ROUNDED,
      padding=(1, 2),
      expand=False,
    )
  )
