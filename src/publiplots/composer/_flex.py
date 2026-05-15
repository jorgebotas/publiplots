"""Pure-geometry flex resolver for PR 2 sizing.

Given a row of panel widths (some pinned mm, some 'flex' sentinels) and
a canvas budget minus decorations, resolve every flex panel to a
concrete mm width such that:

  sum(resolved_widths) + decorations + (n-1) * hpad == canvas_width

Multiple flex panels split the leftover width equally. If no flex panels
exist, the function is a no-op (returns the input pinned widths) and
the caller is responsible for raising overflow if pinned + decorations
> canvas_width.

No matplotlib imports. Pure math.
"""

from typing import Any, Sequence, Tuple


def resolve_flex_widths(
    raw_widths: Sequence[Any],   # mix of float and the literal 'flex'
    *,
    canvas_width_mm: float,
    decorations_width_mm: float,
) -> Tuple[Tuple[float, ...], int]:
    """Resolve 'flex' entries to concrete mm widths.

    Parameters
    ----------
    raw_widths : Sequence
        Mix of floats (pinned mm widths) and the string ``'flex'``.
    canvas_width_mm : float
        The canvas's declared width budget.
    decorations_width_mm : float
        Sum of outer_pad×2 + ncols×ylabel_space + ncols×right + (n-1)×hpad.
        Pre-computed by the caller from rcParams.

    Returns
    -------
    resolved_widths : tuple of float
        All entries are concrete floats; flex entries replaced with
        their share of the leftover width.
    n_flex : int
        Count of flex entries (0 when no flex). When ``n_flex == 0``
        the caller treats the widths as pinned-only and may raise
        :class:`ComposerOverflowError`. When ``n_flex >= 1`` the
        resolver fills the canvas budget exactly (modulo float noise).

    Raises
    ------
    ValueError
        If a flex entry would resolve to a non-positive width (i.e.
        the pinned widths plus decorations already exceed the canvas
        budget; flex panels would need to be 0 mm or negative).
    """
    n_flex = sum(1 for w in raw_widths if w == "flex")
    if n_flex == 0:
        # No flex; just coerce + return.
        return tuple(float(w) for w in raw_widths), 0

    pinned_total = sum(float(w) for w in raw_widths if w != "flex")
    leftover = canvas_width_mm - decorations_width_mm - pinned_total
    per_flex = leftover / n_flex

    if per_flex <= 0.0:
        raise ValueError(
            f"flex panels would resolve to non-positive width "
            f"({per_flex:.2f} mm each); pinned panels {pinned_total:.2f}mm "
            f"+ decorations {decorations_width_mm:.2f}mm already exceed "
            f"canvas budget {canvas_width_mm:.2f}mm"
        )

    resolved = tuple(per_flex if w == "flex" else float(w) for w in raw_widths)
    return resolved, n_flex
