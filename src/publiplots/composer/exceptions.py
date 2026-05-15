"""Composer-specific exceptions.

All composer errors inherit from ComposerError (which itself is a
ValueError) so callers can catch with either the composer-specific or
standard-library type.
"""

from typing import Optional


class ComposerError(ValueError):
    """Base class for all publiplots Composer errors."""


class ComposerOverflowError(ComposerError):
    """Raised when a row's panels overflow the canvas width budget.

    Carries ``requested_mm`` and ``available_mm`` attributes so callers
    can format helpful messages without re-parsing the string. Also
    exposes ``scale_to_fit`` — a multiplicative factor that, when
    applied to all panel widths in the offending row, makes the row
    fit the canvas width budget.

    The default ``__str__`` appends the suggested scale factor to the
    user's message so it surfaces in tracebacks without further work.
    """

    def __init__(
        self,
        message: str,
        *,
        requested_mm: float,
        available_mm: float,
    ) -> None:
        self.requested_mm = float(requested_mm)
        self.available_mm = float(available_mm)
        # Scale factor: clamp degenerate cases (requested <= 0 → 1.0;
        # requested == available → 1.0). Otherwise available/requested.
        if self.requested_mm <= 0.0:
            self.scale_to_fit = 1.0
        else:
            self.scale_to_fit = self.available_mm / self.requested_mm
        # Build the augmented message. If scale < 1, suggest shrinkage;
        # if scale >= 1, just keep the user's message (no advisor needed).
        if self.scale_to_fit < 1.0:
            advisor = (
                f" — multiply panel widths by {self.scale_to_fit:.3f} to fit"
            )
            super().__init__(message + advisor)
        else:
            super().__init__(message)


class ComposerAlignmentError(ComposerError):
    """Raised when a `canvas.align()` request can't be satisfied.

    The most common cause is that the requested shift would push a
    panel's content outside its (inviolate) slot mm-rect. The panels
    + edge attributes let callers format helpful messages without
    re-parsing the message text.
    """

    def __init__(
        self,
        message: str,
        *,
        panels: tuple,
        edge: str,
    ) -> None:
        super().__init__(message)
        self.panels = tuple(panels)
        self.edge = str(edge)


class ComposerVectorError(ComposerError):
    """Raised when the vector compositing pipeline can't preserve a schematic.

    The most common causes are: a corrupt or unsupported schematic file,
    a cairosvg failure parsing an Illustrator-exported SVG, a pypdf
    failure on a malformed source PDF, or a missing optional dep (pypdf
    / cairosvg / Pillow). The ``panel_label`` + ``path`` + ``source_error``
    attributes let callers format helpful messages without re-parsing
    the message text.
    """

    def __init__(
        self,
        message: str,
        *,
        panel_label: Optional[str] = None,
        path: Optional[str] = None,
        source_error: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.panel_label = panel_label
        self.path = path
        self.source_error = source_error
