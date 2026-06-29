"""Pipeline progress display, per-step theming, and result summaries for the ZairaChem CLI.

This module was split for maintainability into three focused modules; it now re-exports their public
names so existing ``from zairachem.base.utils.progress import ...`` imports keep working unchanged:

* :mod:`zairachem.base.utils.live` — the live widgets (``LiveTableMonitor``, ``LiveProgressBar``,
  ``SetupProgress``) and the spinner / CPU-RAM caption / ``_bar`` primitives.
* :mod:`zairachem.base.utils.summaries` — per-step ``✓`` summaries, the detail blocks, and the
  closing run-summary panel.
* :mod:`zairachem.base.utils.pipeline_tracker` — the ``PipelineTracker`` and the shared ``tracker``.

Prefer importing from those modules directly in new code.
"""

# ruff: noqa: F401  (this module exists to re-export)
from zairachem.base.utils.live import (
  LiveProgressBar,
  LiveTableMonitor,
  SetupProgress,
  _bar,
  _resource_caption,
  _spinner_frame,
)
from zairachem.base.utils.pipeline_tracker import (
  PIPELINE_STEPS,
  STEP_COLORS,
  PipelineTracker,
  tracker,
)
from zairachem.base.utils.summaries import (
  SUMMARIES,
  _detail_rows,
  _is_predict,
  _LIVE_TABLE_STEPS,
  _PREDICT_STEP_DESC,
  _resolve_output_dir,
  final_summary_panel,
  summarize_describe,
  summarize_estimate,
  summarize_finish,
  summarize_pool,
  summarize_projections,
  summarize_report,
  summarize_setup,
  summarize_treat,
)
