"""Panel constructors and result types for the Composer.

PR 1 ships only :class:`PanelAxes`. PR 3 adds :class:`PanelGrid` /
:class:`PanelText`; PR 5 adds :class:`PanelImage`.

Pure-Python module — no matplotlib imports. The result type
:class:`Panel` carries an opaque ``ax`` reference that the canvas
populates; that's the only object that crosses the Python/matplotlib
boundary, and it's typed as ``Any`` here for layering purity.
"""

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Tuple


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

    For ``kind='axes'``, ``ax`` is a single :class:`matplotlib.axes.Axes`.
    PR 3 adds ``kind='axesgrid'`` (where ``axes`` is a numpy array) and
    other panel kinds.

    Attributes
    ----------
    label : str
        The caption-addressable identifier.
    kind : str
        One of ``'axes'``, ``'axesgrid'``, ``'image'``, ``'text'``.
        PR 1 only emits ``'axes'``.
    ax : matplotlib.axes.Axes or None
        The underlying axes for ``kind='axes'``. ``None`` for non-axes
        panels (PR 5+).
    size_mm : tuple of (width_mm, height_mm)
        Panel mm rect dimensions.
    bbox_mm : tuple of (x_mm, y_mm, w_mm, h_mm)
        Panel mm rect position relative to the canvas. ``(x, y)`` is the
        bottom-left corner; ``(w, h)`` matches ``size_mm``.
    """

    label: Any                    # str | None | False
    kind: PanelKind
    ax: Optional[Any]  # matplotlib.axes.Axes | None — typed Any for layering
    size_mm: Tuple[float, float]
    bbox_mm: Tuple[float, float, float, float]
    resolved_label_style: Optional[Mapping[str, Any]] = None


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
