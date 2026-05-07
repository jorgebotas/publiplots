"""
Style utilities for publiplots.

publiplots is publication-first by design: simply importing the package
applies the publication-grade rcParams (compact fonts, high DPI, thin
strokes, mm-based layout defaults). There is no separate "notebook mode"
— the same styling works for interactive and print output. Users who
want matplotlib's defaults can call :func:`reset_style`.

The functions here are escape hatches for inspecting or overriding the
defaults from user code; the styling itself is installed by
:func:`publiplots.themes.rcparams.init_rcparams` on import.
"""

from typing import Dict, Any
import matplotlib.pyplot as plt


def reset_style() -> None:
    """
    Reset matplotlib rcParams to matplotlib's own defaults.

    publiplots applies its publication-grade rcParams (Arial font,
    0.75 pt strokes, 600 savefig DPI, compact 8 pt labels, etc.)
    during ``import publiplots``. Call :func:`reset_style` to revert
    matplotlib rcParams to matplotlib's stock defaults — for example
    when embedding a publiplots plot inside a larger figure that
    should follow the host project's styling.

    This does **not** affect publiplots-specific parameters accessed
    via ``pp.rcParams`` (``'alpha'``, ``'palette'``,
    ``'subplots.axes_size'``, etc.). Those continue to drive plot
    functions regardless of the matplotlib rcParams state.

    Examples
    --------
    >>> import publiplots as pp
    >>> pp.reset_style()  # revert to matplotlib's defaults

    Restore publiplots' defaults afterwards:

    >>> from publiplots.themes.rcparams import init_rcparams
    >>> init_rcparams()

    See Also
    --------
    publiplots.rcParams : Unified access to matplotlib and publiplots params.
    """
    plt.rcdefaults()


def get_current_style() -> Dict[str, Any]:
    """
    Get the current matplotlib rcParams as a plain dict.

    Useful for debugging or snapshotting the active style. Only returns
    matplotlib rcParams, not publiplots-specific parameters.

    Returns
    -------
    Dict[str, Any]
    """
    return dict(plt.rcParams)


def apply_custom_style(style_dict: Dict[str, Any]) -> None:
    """
    Apply a custom matplotlib rcParams dict on top of the current style.

    Only affects matplotlib rcParams. To change publiplots-specific
    parameters, assign to ``pp.rcParams`` directly.

    Parameters
    ----------
    style_dict : Dict[str, Any]

    Examples
    --------
    >>> import publiplots as pp
    >>> pp.apply_custom_style({'font.size': 10, 'lines.linewidth': 1.5})
    """
    plt.rcParams.update(style_dict)
