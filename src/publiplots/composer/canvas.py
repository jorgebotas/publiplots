"""Canvas — the Composer's top-level orchestrator.

PR 1 supports single-row layouts via :meth:`Canvas.add_row` and raster
:meth:`Canvas.savefig`. The figure is created lazily when ``add_row``
runs; ``Canvas(...)`` itself just records configuration.

Internally, the canvas builds a 1-row × N-column :class:`FigureLayout`
where each column's mm width matches one panel's declared width. The
existing :class:`SubplotsAutoLayout` reactor handles xlabel / ylabel /
title decoration reservation (per spike Finding 4 — without it, axis
decorations clip at the canvas mediabox edge).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from publiplots.composer.exceptions import (
    ComposerError,
    ComposerOverflowError,
)
from publiplots.composer.panels import Panel, PanelAxes
from publiplots.composer.presets import resolve_preset
from publiplots.themes.rcparams import resolve_param


class Canvas:
    """Programmatic mm-precise canvas for multi-panel paper figures.

    Parameters
    ----------
    preset : str
        Preset name. PR 1 only supports ``'custom'``; journal presets
        (Cell, Nature, Nature Methods, Science) land in PR 2.
    width : float, optional
        Canvas width in millimeters. Required for ``preset='custom'``.

    Examples
    --------
    Single-row, two axes panels:

    >>> import publiplots as pp
    >>> canvas = pp.Canvas("custom", width=174.0)
    >>> canvas.add_row(
    ...     pp.PanelAxes(label="A", size=(70, 40)),
    ...     pp.PanelAxes(label="B", size=(70, 40)),
    ... )
    >>> pp.scatterplot(data=df, x="x", y="y", ax=canvas["A"].ax)
    >>> canvas.savefig("fig.png")

    Notes
    -----
    Multi-row layouts and ``add_column`` land in PR 3. Vector PDF/SVG
    save dispatches land in PR 5/PR 6. ``canvas.savefig('fig.pdf')``
    raises :class:`NotImplementedError` in PR 1.
    """

    def __init__(self, preset: str, *, width: Optional[float] = None) -> None:
        self._preset_name = preset
        spec = resolve_preset(preset, width=width)
        self._width_mm: float = spec["width_mm"]
        self._max_height_mm: Optional[float] = spec["max_height_mm"]

        # Lazy-initialized on first add_row():
        self._figure = None  # matplotlib.figure.Figure | None
        self._panels: Dict[str, Panel] = {}
        self._row_added: bool = False

    # ------------------------------------------------------------------
    # Read-only attributes
    # ------------------------------------------------------------------
    @property
    def width_mm(self) -> float:
        """Canvas width in millimeters."""
        return self._width_mm

    @property
    def figure(self):
        """The underlying matplotlib :class:`Figure`, or ``None`` until
        :meth:`add_row` runs."""
        return self._figure

    @property
    def figure_size_mm(self) -> Optional[Tuple[float, float]]:
        """``(width_mm, height_mm)`` after :meth:`add_row`, else ``None``."""
        if self._figure is None:
            return None
        # Convert mpl figure size (inches) back to mm.
        w_in, h_in = self._figure.get_size_inches()
        return (w_in * 25.4, h_in * 25.4)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def __getitem__(self, label: str) -> Panel:
        if label not in self._panels:
            raise KeyError(
                f"no panel with label {label!r}; known labels: {sorted(self._panels)}"
            )
        return self._panels[label]
