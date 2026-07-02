"""Optional descriptor pre-screening (``--max-descriptors``).

Cheaply proxy-scores each computed descriptor on the training rows and keeps only the top-K, so the
expensive per-descriptor autoML fits (and the shipped model) are limited to the promising descriptors.
Leak-free: production screens on all training rows; each held-out fold re-screens on its own train slice.
"""
