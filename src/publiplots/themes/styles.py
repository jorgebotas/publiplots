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

    Useful when you want to revert to matplotlib's default styling
    entirely (disabling publiplots' publication-grade rcParams). Does
    not affect publiplots custom parameters (``pp.rcParams['alpha']``,
    etc.).

    Examples
    --------
    >>> import publiplots as pp
    >>> pp.reset_style()
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
