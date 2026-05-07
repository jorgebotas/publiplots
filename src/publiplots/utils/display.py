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

    Thin wrapper around :func:`matplotlib.pyplot.show`. Behaviour depends
    on the active matplotlib backend: interactive backends block until
    the window closes; inline / Agg backends return immediately.

    Parameters
    ----------
    *args, **kwargs
        Forwarded to :func:`matplotlib.pyplot.show`. The most common
        kwarg is ``block`` (``True`` by default in interactive mode).

    Examples
    --------
    >>> import publiplots as pp
    >>> fig, ax = pp.subplots()
    >>> pp.barplot(data=df, x='category', y='value', ax=ax)
    >>> pp.show()
    """
    plt.show(*args, **kwargs)


def suptitle(text: str, **kwargs: Any) -> Text:
    """Add a figure-level title to the current figure.

    Thin wrapper around :func:`matplotlib.pyplot.suptitle` that also
    hooks into publiplots' auto-layout engine: the figure grows
    vertically to reserve room for the title (no overlap with top-row
    axis titles), and repeated calls replace the prior suptitle rather
    than accumulating texts on the figure. Returns the created
    :class:`~matplotlib.text.Text` artist. Uses
    ``matplotlib.pyplot.gcf()`` to locate the target figure.

    Parameters
    ----------
    text : str
        The title text.
    **kwargs
        Forwarded to :func:`matplotlib.pyplot.suptitle` (e.g.,
        ``fontsize``, ``y``, ``fontweight``).

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
    fig = plt.gcf()
    prior = getattr(fig, "_publiplots_suptitle", None)
    if prior is not None:
        try:
            prior.remove()
        except (NotImplementedError, ValueError):
            # Already detached from the figure, or remove unsupported;
            # proceed — the important invariant is that only the new
            # artist ends up in fig._publiplots_suptitle.
            pass
        # Matplotlib caches the suptitle on ``fig._suptitle`` and
        # re-uses it on subsequent ``plt.suptitle`` calls. After we
        # remove the prior Text from ``fig.texts``, null the cache so
        # the next ``plt.suptitle`` creates a fresh, attached artist
        # rather than silently updating the detached one.
        fig._suptitle = None
    artist = plt.suptitle(text, **kwargs)
    fig._publiplots_suptitle = artist
    # Trigger a re-measure so the figure grows before the user sees it.
    al = getattr(fig, "_publiplots_auto_layout", None)
    if al is not None:
        al.settle()
    return artist
