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
        # Validate label: must be str, None, or False (NOT True).
        # Reject the ambiguous bool-True case BEFORE the str check
        # because bool is a subclass of int, but `isinstance(True, str)`
        # is False — so the ambiguity is in the "is not False" path.
        if self.label is True:
            raise TypeError(
                "label must be a str, None (auto-letter), or False "
                "(no label); got True (did you mean abc=True on the Canvas?)"
            )
        if self.label is not None and self.label is not False:
            if not isinstance(self.label, str):
                raise TypeError(
                    f"label must be a str, None (auto-letter), or False "
                    f"(no label); got {type(self.label).__name__}"
                )

        # Validate size: must be a 2-tuple
        try:
            n = len(self.size)
        except TypeError:
            raise ValueError(
                f"size must be a 2-tuple of (width_mm, height_mm), got {self.size!r}"
            )
        if n != 2:
            raise ValueError(
                f"size must be a 2-tuple of (width_mm, height_mm), got length {n}"
            )

        w, h = self.size

        # Validate width: 'flex' string OR positive numeric
        if isinstance(w, str):
            if w != "flex":
                raise ValueError(
                    f"size width must be numeric or 'flex', got {w!r}"
                )
            # 'flex' width is OK — leave as the string sentinel.
        else:
            try:
                fw = float(w)
            except (TypeError, ValueError):
                raise ValueError(f"size width must be numeric or 'flex', got {w!r}")
            if fw <= 0:
                raise ValueError(f"size width must be positive, got {fw}")
            w = fw  # coerce int → float

        # Validate height: must be positive numeric (no 'flex' in PR 2)
        if isinstance(h, str):
            raise ValueError(
                f"size height must be numeric, got {h!r} "
                "('flex' is width-only in PR 2)"
            )
        try:
            fh = float(h)
        except (TypeError, ValueError):
            raise ValueError(f"size height must be numeric, got {h!r}")
        if fh <= 0:
            raise ValueError(f"size height must be positive, got {fh}")
        h = fh

        # Validate label_style: must be Mapping or None
        if self.label_style is not None:
            if not hasattr(self.label_style, "keys"):
                raise TypeError(
                    f"label_style must be a Mapping or None, "
                    f"got {type(self.label_style).__name__}"
                )

        # Coerce normalized values onto the frozen instance.
        object.__setattr__(self, "size", (w, h))


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
