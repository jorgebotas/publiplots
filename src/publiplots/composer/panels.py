"""Panel constructors and result types for the Composer.

PR 1 ships only :class:`PanelAxes`. PR 3 adds :class:`PanelGrid` /
:class:`PanelText`; PR 5 adds :class:`PanelImage`.

Pure-Python module — no matplotlib imports. The result type
:class:`Panel` carries an opaque ``ax`` reference that the canvas
populates; that's the only object that crosses the Python/matplotlib
boundary, and it's typed as ``Any`` here for layering purity.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Tuple, Union


PanelKind = Literal["axes", "axesgrid", "image", "text"]  # only "axes" used in PR 1


def _validate_panel_label(label: Any) -> None:
    """Validate that label is str, None, or False. Raises TypeError otherwise."""
    if label is True:
        raise TypeError(
            "label must be a str, None (auto-letter), or False "
            "(no label); got True (did you mean abc=True on the Canvas?)"
        )
    if label is not None and label is not False:
        if not isinstance(label, str):
            raise TypeError(
                f"label must be a str, None (auto-letter), or False "
                f"(no label); got {type(label).__name__}"
            )


def _validate_panel_size(
    size: Any,
    *,
    allow_flex_width: bool = True,
) -> Tuple[Any, float]:
    """Validate a panel size 2-tuple. Returns the normalized tuple
    (with int→float coercion on numeric width)."""
    try:
        n = len(size)
    except TypeError:
        raise ValueError(
            f"size must be a 2-tuple of (width_mm, height_mm), got {size!r}"
        )
    if n != 2:
        raise ValueError(
            f"size must be a 2-tuple of (width_mm, height_mm), got length {n}"
        )

    w, h = size

    # Width: 'flex' string OR positive numeric
    if isinstance(w, str):
        if w != "flex" or not allow_flex_width:
            raise ValueError(
                f"size width must be numeric or 'flex', got {w!r}"
            )
        # 'flex' is OK — leave the string sentinel.
    else:
        try:
            fw = float(w)
        except (TypeError, ValueError):
            raise ValueError(f"size width must be numeric or 'flex', got {w!r}")
        if fw <= 0:
            raise ValueError(f"size width must be positive, got {fw}")
        w = fw

    # Height: positive numeric (no 'flex')
    if isinstance(h, str):
        raise ValueError(
            f"size height must be numeric, got {h!r} "
            "('flex' is width-only)"
        )
    try:
        fh = float(h)
    except (TypeError, ValueError):
        raise ValueError(f"size height must be numeric, got {h!r}")
    if fh <= 0:
        raise ValueError(f"size height must be positive, got {fh}")

    return (w, fh)


@dataclass(frozen=True)
class PanelAxes:
    """Input record for an axes panel.

    Pass this to :meth:`Canvas.add_row` to declare a panel containing a
    single matplotlib Axes. The canvas allocates the Axes at the panel's
    mm rect and stores the result as a :class:`Panel`.

    Parameters
    ----------
    label : str, None, or False
        Caption-addressable identifier:

        - ``str`` — verbatim label (e.g., ``'A'``, ``'B.i'``).
        - ``None`` — auto-letter slot. The canvas's ``abc=`` template
          resolves it to the next sequence value (``'A'`` if
          ``abc='upper'``, ``'a'`` if ``'lower'``, etc.).
        - ``False`` — explicitly suppress label rendering and skip from
          the auto-letter sequence.
    size : tuple of (width, height_mm)
        Panel mm rect:

        - ``(w_mm, h_mm)`` — both pinned (PR 1 contract).
        - ``('flex', h_mm)`` — width grows to absorb leftover row width.
          Multiple flex panels in a row split the leftover equally.

        The height must always be a positive numeric. PR 2 does NOT
        introduce flex heights.
    label_style : Mapping or None, default None
        Per-panel override of the canvas-wide ``label_style``. Accepts
        any subset of the canvas-wide keys (``loc``, ``size``,
        ``weight``, ``family``, ``pad_mm``, ``border``, ``bbox``).
        Missing keys fall through to the canvas default.

    Notes
    -----
    Frozen dataclass — instances are immutable and hashable.
    """

    label: Any            # str | None | False (type-narrowing in __post_init__)
    size: Tuple[Any, Any]  # (float | 'flex', float)
    label_style: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        _validate_panel_label(self.label)
        normalized = _validate_panel_size(self.size, allow_flex_width=True)
        object.__setattr__(self, "size", normalized)
        if self.label_style is not None:
            if not hasattr(self.label_style, "keys"):
                raise TypeError(
                    f"label_style must be a Mapping or None, "
                    f"got {type(self.label_style).__name__}"
                )


@dataclass(frozen=True)
class Panel:
    """Result type returned by ``canvas[label]``.

    Carries the resolved axes reference and the panel's mm geometry.
    Frozen so users don't accidentally mutate cached metadata.

    For ``kind='axes'``: ``ax`` is a single :class:`matplotlib.axes.Axes`,
    ``axes`` is None.
    For ``kind='axesgrid'``: ``ax`` is None, ``axes`` is a 2D numpy
    ndarray of `Axes`.
    For ``kind='text'``: ``ax`` is the hidden axes hosting the text;
    ``axes`` is None.

    Attributes
    ----------
    label : str
        The caption-addressable identifier.
    kind : str
        One of ``'axes'``, ``'axesgrid'``, ``'image'``, ``'text'``.
        PR 1 only emits ``'axes'``.
    ax : matplotlib.axes.Axes or None
        The underlying axes for ``kind='axes'`` (and the hidden host
        axes for ``kind='text'``). ``None`` for ``kind='axesgrid'``
        (use :attr:`axes` instead) and other non-axes panels (PR 5+).
    size_mm : tuple of (width_mm, height_mm)
        Panel mm rect dimensions.
    bbox_mm : tuple of (x_mm, y_mm, w_mm, h_mm)
        Panel mm rect position relative to the canvas. ``(x, y)`` is the
        bottom-left corner; ``(w, h)`` matches ``size_mm``.
    axes : numpy.ndarray of Axes, or None
        For ``kind='axesgrid'``: 2D ndarray (shape = ``PanelGrid.shape``)
        of the inner sub-grid axes. Row 0 is the TOP of the grid (visual
        convention). ``None`` for all other kinds.
    """

    label: Any                    # str | None | False
    kind: PanelKind
    ax: Optional[Any]  # matplotlib.axes.Axes | None — typed Any for layering
    size_mm: Tuple[float, float]
    bbox_mm: Tuple[float, float, float, float]
    resolved_label_style: Optional[Mapping[str, Any]] = None
    axes: Optional[Any] = None  # numpy ndarray of Axes for kind='axesgrid'


@dataclass(frozen=True)
class PanelText:
    """Input record for a text-only panel.

    Pass to :meth:`Canvas.add_row` to declare a panel containing a
    single block of text (centered by default). Internally rendered as
    a hidden axes (axis off, no patch) with one ``ax.text(...)`` call.
    Supports basic matplotlib mathtext via the underlying renderer.

    Parameters
    ----------
    label : str, None, or False
        Caption-addressable identifier. Same contract as :class:`PanelAxes`.
    text : str
        The text content. Supports mathtext (``r"$\\alpha$"``) and
        newlines.
    size : tuple of (width, height_mm)
        Panel mm rect. ``('flex', h_mm)`` accepted for flex width.
    text_kw : Mapping or None, default None (= empty dict)
        Forwarded to :func:`matplotlib.axes.Axes.text` (e.g.
        ``{'fontsize': 12, 'color': 'gray'}``). The axes text is placed
        at ``(0.5, 0.5)`` with ``ha='center', va='center'`` by default;
        these can be overridden via ``text_kw``.
    label_style : Mapping or None, default None
        Per-panel label_style override, same contract as
        :class:`PanelAxes`.

    Notes
    -----
    Frozen dataclass — instances are immutable and hashable.
    """

    label: Any
    text: str
    size: Tuple[Any, Any]
    text_kw: Optional[Mapping[str, Any]] = None
    label_style: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        _validate_panel_label(self.label)

        if not isinstance(self.text, str):
            raise TypeError(
                f"text must be a str, got {type(self.text).__name__}"
            )

        normalized_size = _validate_panel_size(self.size, allow_flex_width=True)
        object.__setattr__(self, "size", normalized_size)

        # text_kw: Mapping or None (None → empty dict)
        if self.text_kw is None:
            object.__setattr__(self, "text_kw", {})
        elif not hasattr(self.text_kw, "keys"):
            raise TypeError(
                f"text_kw must be a Mapping or None, "
                f"got {type(self.text_kw).__name__}"
            )

        if self.label_style is not None:
            if not hasattr(self.label_style, "keys"):
                raise TypeError(
                    f"label_style must be a Mapping or None, "
                    f"got {type(self.label_style).__name__}"
                )


_VALID_SHARE = frozenset({True, False, "all", "row", "col", "none"})


@dataclass(frozen=True)
class PanelGrid:
    """Input record for an axes-grid panel.

    A sub-grid of axes laid out inside the panel's mm rect. The panel's
    outer dimensions are computed from ``shape`` and ``axes_size`` plus
    inter-cell spacing — there is no ``size`` kwarg.

    Parameters
    ----------
    label : str, None, or False
        Caption-addressable identifier. Same contract as :class:`PanelAxes`.
    shape : tuple of (nrows, ncols)
        Inner grid shape. Both must be positive integers.
    axes_size : tuple of (width_mm, height_mm)
        Per-cell axes mm dimensions. Both must be positive floats.
        ``'flex'`` is NOT supported here (would require resolving inside
        the panel's slot, which complicates the geometry; users can
        compute the per-cell width from the available slot mm by hand
        if they need it).
    sharex, sharey : bool or {'all', 'row', 'col', 'none'}, default False
        Axis-sharing semantics, matching :func:`matplotlib.pyplot.subplots`.
    hspace, wspace : float, default 2.0 mm
        Inter-cell spacing in millimeters.
    label_style : Mapping or None, default None
        Per-panel label_style override.

    Notes
    -----
    Frozen dataclass — instances are immutable and hashable.

    The panel's outer mm rect is exposed via :attr:`size_mm`:
    ``(ncols*w + (ncols-1)*wspace, nrows*h + (nrows-1)*hspace)``.
    """

    label: Any
    shape: Tuple[int, int]
    axes_size: Tuple[float, float]
    sharex: Any = False
    sharey: Any = False
    hspace: float = 2.0
    wspace: float = 2.0
    label_style: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        _validate_panel_label(self.label)

        # shape: 2-tuple of positive ints
        try:
            ns = len(self.shape)
        except TypeError:
            raise ValueError(
                f"shape must be a 2-tuple of (nrows, ncols), got {self.shape!r}"
            )
        if ns != 2:
            raise ValueError(
                f"shape must be a 2-tuple of (nrows, ncols), got length {ns}"
            )
        nr, nc = self.shape
        for name, v in (("nrows", nr), ("ncols", nc)):
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                raise ValueError(
                    f"shape must be (positive int, positive int); "
                    f"{name}={v!r} is not a positive int"
                )

        # axes_size: 2-tuple of positive numerics, NO flex
        normalized_axes = _validate_panel_size(
            self.axes_size, allow_flex_width=False
        )
        nw, nh = normalized_axes
        # _validate_panel_size with allow_flex_width=False raises on
        # 'flex' string; nw should always be float here.
        if isinstance(nw, str):  # paranoia
            raise ValueError(
                f"axes_size width must be numeric (no 'flex' for grid), got {nw!r}"
            )
        object.__setattr__(self, "axes_size", (float(nw), nh))

        # sharex/sharey
        for name, v in (("sharex", self.sharex), ("sharey", self.sharey)):
            if v not in _VALID_SHARE:
                raise ValueError(
                    f"{name} must be bool or one of 'all'/'row'/'col'/'none', "
                    f"got {v!r}"
                )

        # hspace/wspace: non-negative numerics
        for name, v in (("hspace", self.hspace), ("wspace", self.wspace)):
            if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
                raise ValueError(
                    f"{name} must be a non-negative number, got {v!r}"
                )

        # label_style
        if self.label_style is not None:
            if not hasattr(self.label_style, "keys"):
                raise TypeError(
                    f"label_style must be a Mapping or None, "
                    f"got {type(self.label_style).__name__}"
                )

    @property
    def size_mm(self) -> Tuple[float, float]:
        """Outer mm rect computed from shape + axes_size + spacing."""
        nr, nc = self.shape
        w, h = self.axes_size
        outer_w = nc * w + max(nc - 1, 0) * self.wspace
        outer_h = nr * h + max(nr - 1, 0) * self.hspace
        return (outer_w, outer_h)


_VALID_IMAGE_EXTS = {".pdf", ".svg", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
_VALID_IMAGE_ALIGN = {
    "top-left", "top", "top-right",
    "left", "center", "right",
    "bottom-left", "bottom", "bottom-right",
}
_VALID_IMAGE_CLIP = {"fit", "fill", "stretch"}


@dataclass(frozen=True)
class PanelImage:
    """Input record for an external-schematic panel.

    The schematic file is referenced by ``path``; vector-preserving
    insertion happens at savefig time via the compositing pipeline
    (PR 5: PDF; PR 6: SVG). Raster sources (PNG/JPG/TIFF) take a raster
    fallback path that's still valid for journal submission as long as
    the schematic was authored at high enough DPI.

    Parameters
    ----------
    label : str | None | False, default None
        Panel label. ``None`` participates in abc auto-sequencing.
    path : str or Path
        Schematic file. Extension must be one of ``.pdf``, ``.svg``,
        ``.png``, ``.jpg``/``.jpeg``, ``.tif``/``.tiff``.
    size : tuple of (width, height)
        Slot size in mm or ``'flex'`` for the width.
    align : str, default 'center'
        Schematic alignment within the slot when its aspect ratio
        differs from the slot's. One of nine CSS-``object-position``
        values.
    clip : str, default 'fit'
        How to handle aspect mismatch. ``'fit'`` (preserve aspect, fit
        inside slot), ``'fill'`` (preserve aspect, fill slot, crop
        overflow), ``'stretch'`` (ignore aspect; ``align`` ignored).
    label_style : Mapping, optional
        Per-panel override of canvas label_style.
    """

    label: Any = None
    path: Union[str, Path] = ""
    size: Tuple[Union[float, str], Union[float, str]] = (0.0, 0.0)
    align: str = "center"
    clip: str = "fit"
    label_style: Optional[Mapping[str, Any]] = None

    def __post_init__(self):
        _validate_panel_label(self.label)
        normalized_size = _validate_panel_size(self.size, allow_flex_width=True)
        object.__setattr__(self, "size", normalized_size)
        # Path validation
        if not self.path:
            raise ValueError("PanelImage: path is required")
        p = Path(self.path) if not isinstance(self.path, Path) else self.path
        ext = p.suffix.lower()
        if ext not in _VALID_IMAGE_EXTS:
            raise ValueError(
                f"PanelImage: unsupported extension {ext!r}. "
                f"Supported: {sorted(_VALID_IMAGE_EXTS)}."
            )
        if not p.exists():
            raise FileNotFoundError(
                f"PanelImage: schematic not found: {p}"
            )
        # Normalize path to Path so downstream callers (Canvas finalize,
        # _resources.py loader) don't have to re-wrap. Frozen dataclass
        # → use object.__setattr__.
        object.__setattr__(self, "path", p)
        # align/clip validation. align is validated regardless of clip;
        # with clip='stretch' the value is recorded but unused at
        # composite time (architect's #8 — keep defensive validation).
        if self.align not in _VALID_IMAGE_ALIGN:
            raise ValueError(
                f"PanelImage: align={self.align!r} invalid. "
                f"Expected one of 'center'/'top-left'/... (9 values); "
                f"got {self.align!r}."
            )
        if self.clip not in _VALID_IMAGE_CLIP:
            raise ValueError(
                f"PanelImage: clip={self.clip!r} invalid. "
                f"Expected 'fit', 'fill', or 'stretch'."
            )

        if self.label_style is not None:
            if not hasattr(self.label_style, "keys"):
                raise TypeError(
                    f"label_style must be a Mapping or None, "
                    f"got {type(self.label_style).__name__}"
                )
