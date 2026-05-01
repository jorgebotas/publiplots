"""pp.annotate: in-plot value labels.

Public surface is the `annotate` function. All other modules are internal
and subject to change; do not import from them directly.
"""
from publiplots.annotate._dispatcher import annotate

__all__ = ["annotate"]
