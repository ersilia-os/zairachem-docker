"""Build a self-contained HTML report that displays the report figures, grouped by topic.

The page is deliberately plain and publication-oriented: a white background, the system font stack,
a sticky section nav, and responsive grids of figure cards. Redundant figures (e.g. the violin /
strip score distributions, or the UMAP / t-SNE / PCA projections) are collapsed into a single
*carousel* card that the reader pages through with arrows and dots.

Figures are referenced from the sibling ``png/`` folder (relative links), not base64-inlined — this
keeps ``report.html`` small, at the cost of the page not being standalone (it must travel with its
``png/`` folder). Written to ``<output_dir>/report/report.html`` next to ``png/``.
"""

from zairachem.report.html.render import write_html_report

__all__ = ["write_html_report"]
