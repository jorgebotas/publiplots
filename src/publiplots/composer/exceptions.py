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
    (and PR 2's overflow advisor) can compute the suggested per-row
    scaling factor without re-parsing the message text.
    """

    def __init__(
        self,
        message: str,
        *,
        requested_mm: float,
        available_mm: float,
    ) -> None:
        super().__init__(message)
        self.requested_mm = float(requested_mm)
        self.available_mm = float(available_mm)
