"""publiplots Composer — multi-panel paper-figure builder.

Public API for PR 1 (single-row, axes-only):
- :class:`Canvas` — programmatic mm-precise canvas
- :class:`PanelAxes` — axes panel constructor
- :class:`Panel` — result type returned by ``canvas[label]``

See ``docs/superpowers/specs/2026-05-14-composer-design.md`` for the
full design.
"""

from publiplots.composer.canvas import Canvas
from publiplots.composer.exceptions import (
    ComposerError,
    ComposerOverflowError,
)
from publiplots.composer.panels import Panel, PanelAxes

__all__ = [
    "Canvas",
    "ComposerError",
    "ComposerOverflowError",
    "Panel",
    "PanelAxes",
]
