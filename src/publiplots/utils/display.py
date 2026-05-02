"""
Display-side helpers for publiplots.

Thin wrappers around ``matplotlib.pyplot`` so users don't need to import
``matplotlib.pyplot`` alongside ``publiplots`` for common display tasks.
"""

from typing import Any, Optional

import matplotlib.pyplot as plt
from matplotlib.text import Text


def show(*args: Any, **kwargs: Any) -> None:
    """Display figures.

    Thin wrapper around :func:`matplotlib.pyplot.show`. Behavior depends on
    the active matplotlib backend: interactive backends block until the
    window closes; inline / Agg backends return immediately.

    Parameters
    ----------
    *args, **kwargs
        Forwarded to ``matplotlib.pyplot.show``. The most common kwarg is
        ``block`` (``True`` by default in interactive mode).

    Examples
    --------
    >>> import publiplots as pp
    >>> ax = pp.barplot(data=df, x='category', y='value')
    >>> pp.show()
    """
    plt.show(*args, **kwargs)


def suptitle(text: str, **kwargs: Any) -> Text:
    """Add a figure-level title to the current figure.

    Thin wrapper around :func:`matplotlib.pyplot.suptitle`. Returns the
    created :class:`~matplotlib.text.Text` artist.

    Parameters
    ----------
    text : str
        The title text.
    **kwargs
        Forwarded to ``matplotlib.pyplot.suptitle`` (e.g., ``fontsize``,
        ``y``, ``fontweight``).

    Returns
    -------
    matplotlib.text.Text
        The suptitle artist.

    Examples
    --------
    >>> import publiplots as pp
    >>> fig, axes = pp.subplots(1, 2)
    >>> pp.barplot(data=df1, x='x', y='y', ax=axes[0])
    >>> pp.barplot(data=df2, x='x', y='y', ax=axes[1])
    >>> pp.suptitle('Comparison of two experiments')
    """
    return plt.suptitle(text, **kwargs)
