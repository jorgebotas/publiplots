"""Composer-specific exceptions.

All composer errors inherit from ComposerError (which itself is a
ValueError) so callers can catch with either the composer-specific or
standard-library type.
"""


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
