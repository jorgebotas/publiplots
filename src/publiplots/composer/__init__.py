"""publiplots Composer — multi-panel paper-figure builder.

Public API:
- :class:`Canvas` — programmatic mm-precise canvas (12 journal
  presets + ``'custom'`` escape hatch)
- :class:`PanelAxes` — axes panel constructor (pinned mm or ``'flex'``
  width; auto-letter / verbatim / suppressed labels)
- :class:`Panel` — result type returned by ``canvas[label]`` or
  ``canvas[i]``

See ``docs/superpowers/specs/2026-05-14-composer-design.md`` for the
full design.
"""

from publiplots.composer.canvas import Canvas
from publiplots.composer.exceptions import (
    ComposerAlignmentError,
    ComposerError,
    ComposerOverflowError,
    ComposerVectorError,
)
from publiplots.composer.panels import (
    Panel,
    PanelAxes,
    PanelGrid,
    PanelImage,
    PanelText,
)

__all__ = [
    "Canvas",
    "ComposerAlignmentError",
    "ComposerError",
    "ComposerOverflowError",
    "ComposerVectorError",
    "Panel",
    "PanelAxes",
    "PanelGrid",
    "PanelImage",
    "PanelText",
]
