"""Utilities for custom error-bar / error-band rendering.

When callers already have precomputed confidence bounds (e.g. from a
LOESS bootstrap, a GAM fit, or a Bayesian posterior), they can pass
``errorbar=('custom', (lower_col, upper_col))`` to
:func:`publiplots.pointplot` or :func:`publiplots.lineplot` and have
the bounds rendered as error bars / shaded bands without running
seaborn's own bootstrap.

The plot functions convert that request into a seaborn-native
``estimator='median'`` + ``errorbar=('pi', 100)`` call by triplicating
each row of the data so the 100% percentile interval spans exactly the
requested ``(lo, hi)`` bounds. :func:`format_for_custom_errorbar`
performs that triplication.
"""

from typing import Optional, Tuple

import pandas as pd

from publiplots.utils.validation import is_categorical


def format_for_custom_errorbar(
    data: pd.DataFrame,
    x: str,
    y: str,
    cols: Tuple[str, str],
    orient: Optional[str] = None,
) -> pd.DataFrame:
    """Triplicate rows so a 100% percentile interval spans ``(lo, hi)``.

    Returns a long-format DataFrame with 3x the input row count: one
    copy with the value column overwritten by the lower bound, one
    unchanged copy, and one copy with the value column overwritten by
    the upper bound. When passed to seaborn with
    ``estimator='median'`` + ``errorbar=('pi', 100)``, the resulting
    interval spans exactly ``[lo, hi]``.

    Parameters
    ----------
    data : pd.DataFrame
        Input data containing ``x``, ``y`` and both bound columns.
    x, y : str
        Column names for the x- and y-axis variables.
    cols : (str, str)
        Column names ``(lower, upper)`` holding the precomputed bounds.
    orient : {'x', 'y', None}, optional
        Which axis carries the aggregated numeric value.

        - ``'x'`` (seaborn's default) means the aggregation happens
          within each x group; ``y`` is the value axis and ``(lo, hi)``
          are y-bounds.
        - ``'y'`` swaps that: ``x`` is the value axis.
        - ``None`` auto-detects from which of ``x`` / ``y`` is
          categorical. If both are numeric, falls back to ``y`` as the
          value axis (matching seaborn's ``orient='x'`` default).

    Returns
    -------
    pd.DataFrame
        Long-format frame with 3x the row count of ``data``. The value
        column is overwritten with ``lo`` in the first third and
        ``hi`` in the last third; all other columns are unchanged
        (rows repeated).

    Raises
    ------
    KeyError
        If either ``cols[0]`` or ``cols[1]`` is not a column of
        ``data``.
    """
    lower, upper = cols
    if lower not in data.columns or upper not in data.columns:
        missing = [c for c in cols if c not in data.columns]
        raise KeyError(
            f"Custom errorbar columns not found in data: {missing}. "
            f"Available columns: {list(data.columns)}"
        )

    # Decide which column carries the aggregated numeric value.
    if orient == "x":
        value_col = y
    elif orient == "y":
        value_col = x
    elif orient is None and is_categorical(data[x]):
        value_col = y
    elif orient is None and is_categorical(data[y]):
        value_col = x
    else:
        # Both numeric and no orient given: default to y as the value
        # axis, matching seaborn's ``orient='x'`` default (aggregation
        # happens within x groups, so y carries the numeric value).
        value_col = y

    return pd.concat(
        [
            data.assign(**{value_col: data[lower]}),
            data,
            data.assign(**{value_col: data[upper]}),
        ],
        ignore_index=True,
    )
