"""
File I/O utilities for publiplots.

This module provides functions for saving figures with publication-ready
defaults and other file operations.
"""

from typing import Optional, Any
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from publiplots.themes.rcparams import resolve_param


def savefig(
    filepath: str,
    dpi: Optional[int] = None,
    format: Optional[str] = None,
    bbox_inches: Optional[str] = None,
    transparent: bool = True,
    facecolor: Optional[str] = None,
    edgecolor: Optional[str] = None,
    pad_inches: float = 0.1,
    **kwargs: Any
) -> None:
    """
    Save figure with publication-ready defaults.

    This function wraps matplotlib's savefig with sensible defaults for
    publication-quality output. It automatically creates parent directories
    if they don't exist.

    Parameters
    ----------
    filepath : str
        Output file path. The file extension determines the format if
        format parameter is not specified.
    dpi : int, optional
        Dots per inch for rasterized output. If None, uses DEFAULT_DPI (300).
        Ignored for vector formats (PDF, SVG, EPS).
    format : str, optional
        File format (e.g., 'png', 'pdf', 'svg', 'eps'). If None, inferred
        from filepath extension.
    bbox_inches : str, optional
        Bounding box setting. ``None`` (default) preserves publiplots'
        mm-precise figure layout — the ``pp.subplots`` geometry and any
        ``pp.legend_group`` bands are already measured exactly. Passing
        ``'tight'`` re-crops the figure to the union of artist bboxes,
        which shifts figure-anchored legend bands off-canvas for
        ``side='top'``/``'bottom'`` setups (see issue #TBD).
    transparent : bool, default=False
        If True, make background transparent (PNG, PDF, SVG).
    facecolor : str, optional
        Figure face color. If None, uses current figure facecolor.
    edgecolor : str, optional
        Figure edge color. If None, uses current figure edgecolor.
    pad_inches : float, default=0.1
        Padding around the figure when bbox_inches='tight'.
    **kwargs : Any
        Additional keyword arguments passed to matplotlib.pyplot.savefig().

    Examples
    --------
    Save a figure with default settings:
    >>> import publiplots as pp
    >>> ax = pp.scatterplot(data, x='x', y='y')
    >>> pp.savefig('output.png')

    Save with higher DPI:
    >>> pp.savefig('output.png', dpi=600)

    Save as PDF (vector format):
    >>> pp.savefig('output.pdf')

    Save with transparency:
    >>> pp.savefig('output.png', transparent=True)

    Notes
    -----
    - For publications, use DPI >= 300 for rasterized formats (PNG, JPEG)
    - For presentations, DPI = 150 is usually sufficient
    - For vector formats (PDF, SVG, EPS), DPI is ignored
    - PNG format is recommended for web and presentations
    - PDF format is recommended for publications (vector graphics)
    """
    # Use default DPI if not specified
    dpi = resolve_param("savefig.dpi", dpi)

    # Infer format from filepath if not specified
    if format is None:
        format = Path(filepath).suffix.lstrip('.')
        if not format:
            format = resolve_param("savefig.format", None)

    # Create parent directories if they don't exist
    filepath_obj = Path(filepath)
    filepath_obj.parent.mkdir(parents=True, exist_ok=True)

    # Save the figure
    plt.savefig(
        filepath,
        dpi=dpi,
        format=format,
        bbox_inches=bbox_inches,
        transparent=transparent,
        facecolor=facecolor,
        edgecolor=edgecolor,
        pad_inches=pad_inches,
        **kwargs
    )

    print(f"Figure saved to: {filepath}")


def save_multiple(
    basename: str,
    formats: list = None,
    **kwargs: Any
) -> None:
    """
    Save the same figure in multiple formats.

    Convenient function for saving a figure in multiple formats with
    the same base name (e.g., both PNG for presentations and PDF for
    publications).

    Parameters
    ----------
    basename : str
        Base filename without extension (e.g., 'figure1').
    formats : list, optional
        List of file formats (e.g., ['png', 'pdf', 'svg']).
        If None, saves as both PNG and PDF.
    **kwargs : Any
        Additional keyword arguments passed to savefig().

    Examples
    --------
    Save in default formats (PNG and PDF):
    >>> pp.save_multiple('results/figure1')

    Save in custom formats:
    >>> pp.save_multiple('figure1', formats=['png', 'svg', 'eps'])

    Save with custom DPI:
    >>> pp.save_multiple('figure1', formats=['png'], dpi=600)
    """
    if formats is None:
        formats = ['png', 'pdf']

    for fmt in formats:
        filepath = f"{basename}.{fmt}"
        savefig(filepath, format=fmt, **kwargs)


def close_all() -> None:
    """
    Close all open figures.

    Useful for cleaning up after creating multiple figures in a script.

    Examples
    --------
    >>> pp.close_all()
    """
    plt.close('all')


def get_figure_size(fig: Figure) -> tuple:
    """
    Get the current figure size in inches.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure object.

    Returns
    -------
    tuple
        (width, height) in inches.

    Examples
    --------
    >>> ax = pp.scatterplot(data, x='x', y='y')
    >>> width, height = pp.get_figure_size(ax.get_figure())
    >>> print(f"Figure size: {width} x {height} inches")
    """
    return fig.get_size_inches()


def set_figure_size(fig: Figure, width: float, height: float) -> None:
    """
    Set the figure size in inches.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure object.
    width : float
        Width in inches.
    height : float
        Height in inches.

    Examples
    --------
    >>> ax = pp.scatterplot(data, x='x', y='y')
    >>> pp.set_figure_size(ax.get_figure(), 8, 6)
    """
    fig.set_size_inches(width, height)
