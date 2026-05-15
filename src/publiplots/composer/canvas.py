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

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from publiplots.composer.exceptions import ComposerOverflowError
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

    def __init__(
        self,
        preset: str,
        *,
        width: Optional[float] = None,
        abc: Union[str, bool, Sequence[str], None] = None,
    ) -> None:
        from publiplots.composer.abc_labels import DEFAULT_LABEL_STYLE

        self._preset_name = preset
        spec = resolve_preset(preset, width=width)
        self._width_mm: float = spec["width_mm"]
        self._max_height_mm: Optional[float] = spec["max_height_mm"]

        # abc resolution: if user passed None, fall back to preset default.
        self._abc = abc if abc is not None else spec["abc_default"]

        # Initialize label_style from the default + preset's label_size_pt.
        self._label_style: Dict[str, Any] = dict(DEFAULT_LABEL_STYLE)
        self._label_style["size"] = spec["label_size_pt"]

        # Lazy-initialized on first add_row():
        self._figure = None  # matplotlib.figure.Figure | None
        self._panels: Dict[str, Panel] = {}
        self._panels_ordered: List[Panel] = []  # insertion-order for int indexing
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
    def __getitem__(self, key) -> Panel:
        if isinstance(key, int) and not isinstance(key, bool):
            if not self._panels_ordered:
                raise KeyError("no panels yet; call add_row() first")
            try:
                return self._panels_ordered[key]
            except IndexError:
                raise KeyError(
                    f"panel index {key} out of range; "
                    f"only {len(self._panels_ordered)} panel(s) exist"
                )
        if key not in self._panels:
            raise KeyError(
                f"no panel with label {key!r}; "
                f"known labels: {sorted(self._panels)}"
            )
        return self._panels[key]

    def label_style(self, **kwargs: Any) -> None:
        """Update the canvas-wide label style.

        Accepts any subset of: ``weight``, ``size``, ``family``, ``loc``,
        ``pad_mm``, ``border``, ``bbox``. Missing keys are unchanged.
        ``loc`` must be one of the 8 ultraplot locs (``'ul'``, ``'ur'``,
        ``'ll'``, ``'lr'``, ``'uc'``, ``'lc'``, ``'cl'``, ``'cr'``).
        """
        from publiplots.composer.abc_labels import VALID_LOCS

        if "loc" in kwargs and kwargs["loc"] not in VALID_LOCS:
            raise ValueError(
                f"loc must be one of {sorted(VALID_LOCS)}, got {kwargs['loc']!r}"
            )
        self._label_style.update(kwargs)

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
            # Only str labels need uniqueness; None/False are fine to repeat.
            if not isinstance(lbl, str):
                continue
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

        # Raw widths — may contain 'flex' sentinels which we'll resolve next.
        raw_widths = tuple(p.size[0] for p in panels)
        # All panels in one row share the row height; PR 2 still requires
        # numeric heights (flex is width-only).
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

        # --- resolve flex sizing ------------------------------------
        from publiplots.composer._flex import resolve_flex_widths
        try:
            col_widths, n_flex = resolve_flex_widths(
                raw_widths,
                canvas_width_mm=self._width_mm,
                decorations_width_mm=decorations_width,
            )
        except ValueError as e:
            # The resolver raises ValueError when pinned widths alone
            # overflow. Convert to ComposerOverflowError so the user
            # gets the suggested-scale-factor advisor.
            pinned_total = sum(float(w) for w in raw_widths if w != "flex")
            requested = pinned_total + decorations_width
            raise ComposerOverflowError(
                str(e),
                requested_mm=requested,
                available_mm=self._width_mm,
            )

        # --- pinned-only overflow check (PR 1 path) -----------------
        if n_flex == 0:
            panels_width = sum(col_widths)
            requested_width = panels_width + decorations_width
            if requested_width > self._width_mm + 1e-6:  # 1µm tolerance
                raise ComposerOverflowError(
                    f"row width {requested_width:.2f}mm exceeds canvas width "
                    f"{self._width_mm:.2f}mm; reduce panel widths or use a wider canvas",
                    requested_mm=requested_width,
                    available_mm=self._width_mm,
                )
        # When n_flex >= 1, the resolver guarantees figure width == canvas
        # width exactly (modulo float noise). No additional check needed.

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
        # Resolve labels per the canvas's abc mode.
        from publiplots.composer.abc_labels import (
            resolve_labels, merge_label_style, render_label,
        )
        raw_panel_labels = [p.label for p in panels]
        resolved_labels = resolve_labels(
            panel_labels=raw_panel_labels,
            abc=self._abc,
        )

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

            resolved_label = resolved_labels[col_idx]
            resolved_style = merge_label_style(
                self._label_style, panel_input.label_style
            )

            panel = Panel(
                label=resolved_label,
                kind="axes",
                ax=ax,
                size_mm=(col_widths[col_idx], panel_input.size[1]),
                bbox_mm=bbox_mm,
                resolved_label_style=resolved_style,
            )

            # Render the label if it's a non-empty string.
            if isinstance(resolved_label, str) and resolved_label:
                render_label(ax, resolved_label, style=resolved_style)

            # Register by resolved label (so canvas['A'] works for an
            # auto-letter panel that started as label=None) AND by
            # insertion order. False/None labels skip the dict registry.
            if isinstance(resolved_label, str) and resolved_label:
                self._panels[resolved_label] = panel
            self._panels_ordered.append(panel)

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

    # ------------------------------------------------------------------
    # savefig — raster only in PR 1
    # ------------------------------------------------------------------
    def savefig(self, path, **kwargs) -> None:
        """Save the canvas to a file.

        PR 1 supports raster formats (PNG, JPG, TIFF). PDF and SVG raise
        :class:`NotImplementedError` until PR 5 / PR 6 land the vector
        compositing pipelines.

        Parameters
        ----------
        path : str or Path
            Output file path. Extension determines the format.
        **kwargs
            Forwarded to :func:`publiplots.savefig`.

        Raises
        ------
        RuntimeError
            If :meth:`add_row` has not been called yet.
        NotImplementedError
            If ``path`` ends in ``.pdf`` (PR 5) or ``.svg`` (PR 6).
        ValueError
            If the path's extension is not a known raster or vector type.
        """
        if self._figure is None:
            raise RuntimeError(
                "Canvas has no figure yet; call add_row() before savefig()"
            )
        from publiplots.composer._save import dispatch_savefig
        dispatch_savefig(self._figure, path, **kwargs)
