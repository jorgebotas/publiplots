"""pp.annotate: in-plot value labels.

Public surface:
    - annotate: the strategy-dispatching entry point.
    - BarRecord: the per-bar record passed to `kind="bar_custom"` callables.

All other modules are internal and subject to change; do not import from
them directly.
"""
from publiplots.annotate._cache import BarRecord
from publiplots.annotate._dispatcher import annotate

__all__ = ["annotate", "BarRecord"]
