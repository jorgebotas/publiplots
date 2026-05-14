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

    # ------------------------------------------------------------------
    # add_row — single-row layout for PR 1
    # ------------------------------------------------------------------
    def add_row(self, *panels: PanelAxes) -> None:
        """Add a row of axes panels to the canvas.

        PR 1 supports calling ``add_row`` exactly once per canvas
        (single-row layouts only). PR 3 will lift this restriction.

        Parameters
        ----------
        *panels : PanelAxes
            One or more :class:`PanelAxes` instances. Their declared
            widths plus inter-panel ``hpad`` (3 mm by default from
            rcParams) plus outer + ylabel + right reservations must fit
            in the canvas width.

        Raises
        ------
        NotImplementedError
            If ``add_row`` was already called on this canvas.
        ValueError
            If no panels are passed, or if duplicate labels appear, or
            if a non-:class:`PanelAxes` argument is passed.
        ComposerOverflowError
            If the row's total width exceeds the canvas budget.
        """
        # --- guard against multi-row in PR 1 ------------------------
        if self._row_added:
            raise NotImplementedError(
                "Canvas.add_row called twice; multi-row support lands in PR 3"
            )

        # --- validate inputs ----------------------------------------
        if len(panels) == 0:
            raise ValueError("Canvas.add_row requires at least one panel")
        for p in panels:
            if not isinstance(p, PanelAxes):
                raise TypeError(
                    f"Canvas.add_row only accepts PanelAxes in PR 1, "
                    f"got {type(p).__name__} (PanelGrid/PanelText land in PR 3, "
                    f"PanelImage in PR 5)"
                )
        labels = [p.label for p in panels]
        seen: set = set()
        for lbl in labels:
            if lbl in seen:
                raise ValueError(f"duplicate panel label: {lbl!r}")
            seen.add(lbl)

        # --- compute geometry ---------------------------------------
        # Use the same rcParams defaults that pp.subplots uses so the
        # initial figure size is consistent with single-grid figures.
        outer_pad = float(resolve_param("subplots.outer_pad", None))
        hpad = float(resolve_param("subplots.wspace", None))  # inter-panel gap
        title_space = float(resolve_param("subplots.title_space", None))
        xlabel_space = float(resolve_param("subplots.xlabel_space", None))
        ylabel_space = float(resolve_param("subplots.ylabel_space", None))
        right = float(resolve_param("subplots.right", None))

        col_widths = tuple(p.size[0] for p in panels)
        # All panels in one row share the row height; PR 1 requires equal
        # heights (PR 2 adds 'flex'/'match' grammar). Use the max for now;
        # if heights differ, that's a future-PR issue, not a PR 1 error.
        row_height = max(p.size[1] for p in panels)

        ncols = len(panels)
        # Canvas width budget: panels + (n-1) hpads + ylabel reservations
        # for each column + right reservation for each column + outer pads.
        # ylabel_space and right are PER-COLUMN per FigureLayout's contract.
        decorations_width = (
            2 * outer_pad
            + ncols * ylabel_space
            + ncols * right
            + max(ncols - 1, 0) * hpad
        )
        panels_width = sum(col_widths)
        requested_width = panels_width + decorations_width
        if requested_width > self._width_mm + 1e-6:  # 1µm tolerance for float noise
            raise ComposerOverflowError(
                f"row width {requested_width:.2f}mm exceeds canvas width "
                f"{self._width_mm:.2f}mm; reduce panel widths or use a wider canvas",
                requested_mm=requested_width,
                available_mm=self._width_mm,
            )

        # NOTE: PR 1 does NOT auto-absorb width slack. If the user passes
        # panels that sum to less than the canvas width, the produced
        # figure is narrower than `self._width_mm`. Callers can size
        # panels to fit exactly (use the overflow-error math in reverse:
        # max-panel-width-per-row = (canvas_width - decorations_width)).
        # PR 2 adds 'flex' panel sizing which absorbs slack automatically.

        # --- build FigureLayout (1 row × N cols) --------------------
        from publiplots.layout.figure_layout import FigureLayout
        from publiplots.layout.auto_layout import SubplotsAutoLayout
        import matplotlib.pyplot as plt

        layout = FigureLayout(
            nrows=1,
            ncols=ncols,
            axes_size=(col_widths[0], row_height),  # fallback; col_widths overrides
            col_widths=col_widths,
            row_heights=(row_height,),
            title_space=(title_space,),
            xlabel_space=(xlabel_space,),
            ylabel_space=(ylabel_space,) * ncols,
            right=(right,) * ncols,
            hspace=0.0,           # only 1 row, irrelevant
            wspace=hpad,
            outer_pad=outer_pad,
            legend_column=0.0,
            suptitle_space=0.0,
        )
        W_mm, H_mm = layout.figure_size()

        # --- create the matplotlib figure at the computed mm size ---
        _MM2INCH = 1.0 / 25.4
        fig = plt.figure(
            figsize=(W_mm * _MM2INCH, H_mm * _MM2INCH),
            layout=None,
        )
        self._figure = fig

        # --- create axes per panel ----------------------------------
        for col_idx, panel_input in enumerate(panels):
            x0_frac, y0_frac, w_frac, h_frac = layout.axes_position(0, col_idx)
            ax = fig.add_axes((x0_frac, y0_frac, w_frac, h_frac))
            # bbox_mm is (x_mm, y_mm, w_mm, h_mm) bottom-left origin.
            bbox_mm = (
                x0_frac * W_mm,
                y0_frac * H_mm,
                w_frac * W_mm,
                h_frac * H_mm,
            )
            self._panels[panel_input.label] = Panel(
                label=panel_input.label,
                kind="axes",
                ax=ax,
                size_mm=(panel_input.size[0], panel_input.size[1]),
                bbox_mm=bbox_mm,
            )

        # --- attach SubplotsAutoLayout reactor (spike Finding 4) ----
        # All four auto-measurable sides start at rcParams defaults and
        # auto-grow on first draw to fit decorations. This prevents
        # axis labels from clipping at the canvas mediabox edge — the
        # exact bug the spike's bare-mpl fixture exhibited.
        fig._publiplots_auto_layout = SubplotsAutoLayout(
            fig, layout,
            locked=set(),               # let all four sides auto-measure
            locked_positions={},
        )

        self._row_added = True
