"""
Legend handlers for publiplots.

This module provides custom legend handlers for creating publication-ready legends
that match the double-layer plotting style used in publiplots (transparent fill +
solid edge). The handlers automatically create legend markers that match the
visual style of scatterplots and barplots.
"""

from typing import List, Dict, Optional, Tuple, Any, Union

from publiplots.themes.rcparams import resolve_param
from publiplots.utils.legend_entries import (
    get_entries,
    is_continuous_hue,
)
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerBase, HandlerPatch
from matplotlib.patches import Circle, Rectangle, Patch
import matplotlib.pyplot as plt

# =============================================================================
# Custom Legend Handlers
# =============================================================================


def _normalize_dash_linestyle(ls: Any) -> Any:
    """Normalize a bare on-off dash tuple to matplotlib's canonical form.

    Seaborn accepts ``dashes={label: (on, off)}`` and stores the bare tuple
    verbatim on the handle; matplotlib's ``Line2D._get_dash_pattern`` only
    accepts named strings or the ``(offset, (on, off, ...))`` form, so a
    ``(on, off)`` tuple fed straight back into a new ``Line2D`` crashes with
    ``TypeError: 'int' object is not iterable``.

    Returns anything other than a bare numeric 2-tuple unchanged.
    """
    if (
        isinstance(ls, tuple)
        and len(ls) == 2
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in ls)
    ):
        return (0, ls)
    return ls


class RectanglePatch(Patch):
    """
    Custom rectangle patch object for legend handles.
    """
    def __init__(self, **kwargs):
        if "markersize" in kwargs:
            del kwargs["markersize"]
        super().__init__(**kwargs)
class MarkerPatch(Patch):
    """
    Custom marker patch object for legend handles.
    Embeds marker symbol and markersize properties.
    """
    def __init__(self, marker='o', **kwargs):
        markersize = kwargs.pop("markersize", resolve_param("lines.markersize"))
        markeredgewidth = kwargs.pop("markeredgewidth", resolve_param("edgewidth"))
        self.marker = marker
        self.markersize = markersize
        self.markeredgewidth = markeredgewidth
        super().__init__(**kwargs)

    def get_marker(self) -> str:
        return self.marker

    def set_marker(self, marker: str):
        self.marker = marker

    def get_markersize(self) -> float:
        return self.markersize
    
    def set_markersize(self, markersize: float):
        if markersize is None or markersize == 0:
            markersize = resolve_param("lines.markersize")
        self.markersize = markersize
    
    def get_markeredgewidth(self) -> float:
        return self.markeredgewidth

    def set_markeredgewidth(self, markeredgewidth: float):
        if markeredgewidth is None or markeredgewidth == 0:
            markeredgewidth = resolve_param("edgewidth")
        self.markeredgewidth = markeredgewidth


class LinePatch(Patch):
    """
    Custom patch for line-only legend handles (lineplot with hue/style).

    Draws a horizontal colored line with optional dash pattern. Used when the
    legend entry represents a line series with no distinguishing marker —
    typically hue (color) or style (dash) on a lineplot.
    """
    def __init__(self, linestyle="-", **kwargs):
        # Remove kwargs that Patch doesn't understand
        for _k in ("markersize", "markeredgewidth"):
            kwargs.pop(_k, None)
        super().__init__(**kwargs)
        self.linestyle = linestyle

    def get_linestyle(self) -> str:
        return self.linestyle

    def set_linestyle(self, linestyle: str):
        self.linestyle = linestyle


class LineMarkerPatch(Patch):
    """
    Custom patch for line+marker legend handles (pointplot, lineplot, etc.).
    Embeds marker symbol, markersize, linestyle, and all styling properties.
    """
    def __init__(self, marker='o', linestyle=None, **kwargs):
        markersize = kwargs.pop("markersize", resolve_param("lines.markersize"))
        markeredgewidth = kwargs.pop("markeredgewidth", resolve_param("edgewidth"))
        self.marker = marker
        self.markersize = markersize
        self.markeredgewidth = markeredgewidth
        super().__init__(**kwargs)
        # Override linestyle if provided
        self.linestyle = linestyle

    def get_marker(self) -> str:
        return self.marker

    def set_marker(self, marker: str):
        self.marker = marker

    def get_markersize(self) -> float:
        return self.markersize

    def set_markersize(self, markersize: float):
        if markersize is None or markersize == 0:
            markersize = resolve_param("lines.markersize")
        self.markersize = markersize

    def get_markeredgewidth(self) -> float:
        return self.markeredgewidth

    def set_markeredgewidth(self, markeredgewidth: float):
        if markeredgewidth is None or markeredgewidth == 0:
            markeredgewidth = resolve_param("edgewidth")
        self.markeredgewidth = markeredgewidth

    def get_linestyle(self) -> str:
        return self.linestyle

    def set_linestyle(self, linestyle: str):
        self.linestyle = linestyle


class HandlerRectangle(HandlerPatch):
    """
    Custom legend handler for double-layer rectangle markers.
    
    Automatically extracts alpha, linewidth, hatches, and colors from handles.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_artists(
        self,
        legend: Legend,
        orig_handle: Any,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Any
    ) -> List[Rectangle]:
        """Create the legend marker artists."""
        # Rectangle position and size
        x = -xdescent
        y = -ydescent

        # Extract all properties from the handle
        color, alpha, linewidth, edgecolor, hatch_pattern = self._extract_properties(
            orig_handle
        )

        # Create filled rectangle with transparency
        rect_fill = Rectangle(
            (x, y),
            width,
            height,
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
            linewidth=0,
            transform=trans,
            hatch=None,
            zorder=2
        )

        # Create edge rectangle with hatch pattern
        rect_edge = Rectangle(
            (x, y),
            width,
            height,
            alpha=1,
            facecolor="none",
            edgecolor=edgecolor,
            linewidth=linewidth,
            transform=trans,
            hatch=hatch_pattern,
            zorder=3
        )

        return [rect_fill, rect_edge]

    def _extract_properties(
        self,
        orig_handle: Any
    ) -> Tuple[str, float, float, str, Optional[str]]:
        """
        Extract all properties from the handle.
        
        Returns
        -------
        Tuple[str, float, float, str, Optional[str]]
            (color, alpha, linewidth, edgecolor, hatch_pattern)
        """
        # Defaults
        color = resolve_param("color")
        alpha = resolve_param("alpha", None)
        linewidth = resolve_param("edgewidth", None)
        edgecolor = None
        hatch_pattern = None

        # Extract from Patch
        if hasattr(orig_handle, "get_facecolor"):
            color = orig_handle.get_facecolor()
        if hasattr(orig_handle, "get_edgecolor"):
            edgecolor = orig_handle.get_edgecolor()
        if hasattr(orig_handle, "get_alpha") and orig_handle.get_alpha() is not None:
            alpha = orig_handle.get_alpha()
        if hasattr(orig_handle, "get_linewidth") and orig_handle.get_linewidth():
            linewidth = orig_handle.get_linewidth()
        if hasattr(orig_handle, "get_hatch"):
            hatch_pattern = orig_handle.get_hatch()

        # Handle tuple format (color, hatch, alpha, linewidth)
        if isinstance(orig_handle, tuple):
            if len(orig_handle) >= 1:
                color = orig_handle[0]
            if len(orig_handle) >= 2:
                hatch_pattern = orig_handle[1]
            if len(orig_handle) >= 3:
                alpha = orig_handle[2]
            if len(orig_handle) >= 4:
                linewidth = orig_handle[3]

        # Use face color as edge color if not specified
        if edgecolor is None:
            edgecolor = color

        return color, alpha, linewidth, edgecolor, hatch_pattern


class HandlerMarker(HandlerBase):
    """
    Generic legend handler for any matplotlib marker type.

    Automatically creates double-layer markers (transparent fill + opaque edge)
    for all marker symbols: 'o', '^', 's', 'D', '*', etc.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_artists(
        self,
        legend: Legend,
        orig_handle: Any,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Any
    ) -> List:
        """Create the legend marker artists."""
        from matplotlib.lines import Line2D
        from matplotlib.colors import to_rgba

        # Center point for the marker
        cx = 0.5 * width - 0.5 * xdescent
        cy = 0.5 * height - 0.5 * ydescent

        # Extract all properties from the handle
        marker, color, size, alpha, linewidth, markeredgewidth, edgecolor = self._extract_properties(
            orig_handle, fontsize
        )

        # Create filled marker with transparency
        marker_artist = Line2D(
            [cx], [cy],
            marker=marker,
            markersize=size,
            markerfacecolor=to_rgba(color, alpha),
            markeredgecolor=to_rgba(edgecolor, 1.0),
            markeredgewidth=markeredgewidth,
            linestyle='none',
            transform=trans,
            zorder=2
        )

        return [marker_artist]

    def _extract_properties(
        self,
        orig_handle: Any,
        fontsize: float
    ) -> Tuple[str, str, float, float, float, str]:
        """
        Extract all properties from the handle.

        Returns
        -------
        Tuple[str, str, float, float, float, str]
            (marker, color, size, alpha, linewidth, edgecolor)
        """
        from matplotlib.lines import Line2D

        # Defaults
        marker = 'o'
        color = resolve_param("color")
        size = resolve_param("lines.markersize")
        alpha = resolve_param("alpha")
        linewidth = resolve_param("edgewidth")
        markeredgewidth = resolve_param("edgewidth")
        edgecolor = None

        # Extract from MarkerPatch (created by create_legend_handles)
        if isinstance(orig_handle, MarkerPatch):
            marker = orig_handle.get_marker()
            color = orig_handle.get_facecolor()
            edgecolor = orig_handle.get_edgecolor()
            alpha = orig_handle.get_alpha() if orig_handle.get_alpha() is not None else alpha
            linewidth = orig_handle.get_linewidth() if orig_handle.get_linewidth() else linewidth
            size = orig_handle.get_markersize() if orig_handle.get_markersize() is not None else size
            markeredgewidth = orig_handle.get_markeredgewidth()

        # Extract from Line2D (standard matplotlib)
        elif isinstance(orig_handle, Line2D):
            marker = orig_handle.get_marker() or 'o'
            color = orig_handle.get_color() or orig_handle.get_markerfacecolor()
            size = orig_handle.get_markersize() or size
            markeredgewidth = orig_handle.get_markeredgewidth() or linewidth
            # Line2D doesn't store alpha separately - use default
            # edgecolor will default to face color below

        # Use face color as edge color if not specified
        if edgecolor is None:
            edgecolor = color

        return marker, color, size, alpha, linewidth, markeredgewidth, edgecolor


class HandlerLineMarker(HandlerBase):
    """
    Legend handler for line+marker combinations (pointplot, lineplot, etc.).

    Draws a horizontal line with a marker on top using double-layer styling
    (transparent fill + opaque edge). This handler is designed for plots that
    show both lines and markers (e.g., pointplot, lineplot with markers).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_artists(
        self,
        legend: Legend,
        orig_handle: Any,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Any
    ) -> List:
        """Create the legend line+marker artists."""
        from matplotlib.lines import Line2D
        from matplotlib.colors import to_rgba

        # Extract all properties from the handle
        marker, color, size, alpha, linewidth, markeredgewidth, edgecolor, linestyle = self._extract_properties(
            orig_handle, fontsize
        )

        # Line coordinates (horizontal line across the legend entry)
        line_y = 0.5 * height - 0.5 * ydescent
        line_x_start = -xdescent
        line_x_end = width - xdescent

        # Marker position (center of the line)
        marker_x = 0.5 * width - 0.5 * xdescent
        marker_y = line_y

        # Create the connecting line
        line = Line2D(
            [line_x_start, line_x_end],
            [line_y, line_y],
            color=to_rgba(color, 1.0),
            linewidth=linewidth,
            linestyle=linestyle,
            transform=trans,
            zorder=1
        )

        # Layer 1: White background marker (covers the line)
        marker_background = Line2D(
            [marker_x], [marker_y],
            marker=marker,
            markersize=size,
            markerfacecolor='white',
            markeredgecolor=color,
            markeredgewidth=0,
            linestyle='none',
            transform=trans,
            zorder=2
        )

        # Layer 2: Semi-transparent filled marker
        marker_artist = Line2D(
            [marker_x], [marker_y],
            marker=marker,
            markersize=size,
            markerfacecolor=to_rgba(color, alpha),
            markeredgecolor=to_rgba(edgecolor, 1.0),
            markeredgewidth=markeredgewidth,
            linestyle='none',
            transform=trans,
            zorder=3
        )

        return [line, marker_background, marker_artist]

    def _extract_properties(
        self,
        orig_handle: Any,
        fontsize: float
    ) -> Tuple[str, str, float, float, float, str, str]:
        """
        Extract all properties from the handle.

        Returns
        -------
        Tuple[str, str, float, float, float, str, str]
            (marker, color, size, alpha, linewidth, edgecolor, linestyle)
        """
        from matplotlib.lines import Line2D

        # Defaults
        marker = 'o'
        color = resolve_param("color")
        size = resolve_param("lines.markersize")
        alpha = resolve_param("alpha")
        # The line half of a line+marker swatch is a DATA line, so it falls
        # back to lines.linewidth -- not edgewidth. Only markeredgewidth
        # below is an outline. Note this particular default is unreachable
        # for a LineMarkerPatch (i.e. for every pointplot/lineplot handle),
        # because the branch below overwrites it from the patch; the
        # load-bearing version of this distinction is `linewidth_line` in
        # create_legend_handles.
        linewidth = resolve_param("lines.linewidth")
        markeredgewidth = resolve_param("edgewidth")
        edgecolor = None
        linestyle = None

        # Extract from LineMarkerPatch (created by create_legend_handles)
        if isinstance(orig_handle, LineMarkerPatch):
            marker = orig_handle.get_marker()
            color = orig_handle.get_facecolor()
            edgecolor = orig_handle.get_edgecolor()
            alpha = orig_handle.get_alpha() if orig_handle.get_alpha() is not None else alpha
            linestyle = _normalize_dash_linestyle(orig_handle.get_linestyle())
            linewidth = orig_handle.get_linewidth()
            markeredgewidth = orig_handle.get_markeredgewidth()
            # Use actual markersize from patch (already in correct units)
            patch_size = orig_handle.get_markersize()
            if patch_size is not None:
                size = patch_size

        # Extract from Line2D (standard matplotlib - fallback)
        elif isinstance(orig_handle, Line2D):
            marker = orig_handle.get_marker() or marker
            linestyle = _normalize_dash_linestyle(orig_handle.get_linestyle())
            color = orig_handle.get_color() or orig_handle.get_markerfacecolor()
            line_size = orig_handle.get_markersize()
            if line_size:
                size = line_size
            linewidth = orig_handle.get_linewidth()
            # Line2D doesn't store alpha separately - use default
            # edgecolor will default to face color below

        # Use face color as edge color if not specified
        if edgecolor is None:
            edgecolor = color

        return marker, color, size, alpha, linewidth, markeredgewidth, edgecolor, linestyle


class HandlerLine(HandlerBase):
    """Legend handler for line-only swatches (lineplot hue/style entries).

    Draws a single horizontal line with the handle's color, alpha,
    linewidth, and linestyle.
    """

    def create_artists(
        self,
        legend: Legend,
        orig_handle: Any,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Any,
    ) -> List:
        from matplotlib.lines import Line2D
        from matplotlib.colors import to_rgba

        color = orig_handle.get_facecolor()
        linewidth = orig_handle.get_linewidth()
        linestyle = orig_handle.get_linestyle() if hasattr(orig_handle, "get_linestyle") else "-"
        linestyle = _normalize_dash_linestyle(linestyle)
        if not linewidth:
            # A line swatch represents a data line, so it falls back to
            # lines.linewidth -- NOT edgewidth like the outline swatches do.
            # Only reached for a handle carrying no width at all;
            # create_legend_handles always sets one (`linewidth_line`).
            linewidth = resolve_param("lines.linewidth")

        # Legend line is always fully opaque for readability, matching
        # HandlerLineMarker's behavior. The stored alpha on the handle
        # reflects the plot's fill transparency, not the legend swatch.
        line_y = 0.5 * height - 0.5 * ydescent
        line = Line2D(
            [-xdescent, width - xdescent],
            [line_y, line_y],
            color=to_rgba(color, 1.0),
            linewidth=linewidth,
            linestyle=linestyle,
            transform=trans,
        )
        return [line]


# =============================================================================
# Helper Functions
# =============================================================================


def get_legend_handler_map() -> Dict[type, HandlerBase]:
    """
    Get a handler map for automatic legend styling.

    Returns
    -------
    Dict[type, HandlerBase]
        Dictionary mapping matplotlib types to handler instances.
    """
    handler_rectangle = HandlerRectangle()
    handler_marker = HandlerMarker()
    handler_line_marker = HandlerLineMarker()
    handler_line = HandlerLine()

    return {
        Rectangle: handler_rectangle,
        MarkerPatch: handler_marker,
        LineMarkerPatch: handler_line_marker,
        LinePatch: handler_line,
        Patch: handler_rectangle,
    }

def create_legend_handles(
    labels: List[str],
    colors: Optional[List[str]] = None,
    edgecolors: Optional[Union[str, List[str]]] = None,
    hatches: Optional[List[str]] = None,
    sizes: Optional[List[float]] = None,
    markers: Optional[List[str]] = None,
    linestyles: Optional[List[str]] = None,
    alpha: Optional[float] = None,
    linewidth: Optional[float] = None,
    markeredgewidth: Optional[float] = None,
    style: str = "rectangle",
    color: Optional[str] = None
) -> List[Patch]:
    """
    Create custom legend handles with alpha and linewidth embedded.

    Parameters
    ----------
    labels : List[str]
        Labels for each legend entry.
    colors : List[str], optional
        Colors for each legend entry.
    edgecolors : str or List[str], optional
        Edge colors for each legend entry. If None, defaults to colors.
        If str, broadcasts to all entries. If list, one per entry.
    hatches : List[str], optional
        Hatch patterns for each legend entry (for rectangles).
    sizes : List[float], optional
        Sizes for each legend entry (markersizes).
    markers : List[str], optional
        Marker symbols for each legend entry (e.g., ['o', '^', 's']).
        If provided with linestyles, creates LineMarkerPatch handles.
        If provided without linestyles, creates MarkerPatch handles.
    linestyles : List[str], optional
        Line styles for each legend entry (e.g., ['-', '--', ':']).
        If provided with markers, creates LineMarkerPatch handles.
    alpha : float, optional
        Transparency level for fill layers. When omitted it falls back to
        ``rcParams["alpha"]`` (0.1).
    linewidth : float, optional
        Stroke width of the swatch. Applies to whichever stroke the chosen
        patch type draws: the shape outline for rectangle/marker swatches,
        or the line itself for line and line+marker swatches. When omitted
        it falls back per patch type -- ``rcParams["edgewidth"]`` for an
        outline, ``rcParams["lines.linewidth"]`` for a line.
    markeredgewidth : float, optional
        Width of marker edges — an outline, so when omitted it falls back
        to ``rcParams["edgewidth"]`` (0.75).
    style : str, default="rectangle"
        Style of legend markers: "rectangle", "circle", "marker", or "line".
        Ignored if markers parameter is provided.
    color : str, optional
        Single color for all entries if colors not provided.

    Returns
    -------
    List[Patch]
        List of Patch objects with embedded properties.
    """
    # Read defaults from rcParams if not provided.
    #
    # `linewidth` means two different things depending on which patch type
    # this call produces, so it resolves against two different knobs:
    #
    #   linewidth_outline -- the stroke that OUTLINES a shape
    #                        (RectanglePatch, MarkerPatch) -> edgewidth
    #   linewidth_line    -- the stroke that IS the data line (LinePatch,
    #                        and the connecting line of a LineMarkerPatch)
    #                        -> lines.linewidth
    #
    # Resolving one shared value here and handing it to all four branches
    # renders a line swatch at edgewidth (0.75) for lines actually drawn at
    # lines.linewidth (1.0) -- the same swatch/figure mismatch this module's
    # handlers exist to prevent, just on the line half. This is the
    # load-bearing site for that distinction: HandlerLineMarker and
    # HandlerLine both overwrite their own defaults from the patch, so it is
    # the patch's linewidth that actually reaches the canvas.
    alpha = resolve_param("alpha", alpha)
    linewidth_outline = resolve_param("edgewidth", linewidth)
    linewidth_line = resolve_param("lines.linewidth", linewidth)
    markeredgewidth = resolve_param("edgewidth", markeredgewidth)

    if colors is None:
        default_color = resolve_param("color", None)
        colors = [color if color is not None else default_color] * len(labels)

    if edgecolors is None:
        edgecolors = colors
    elif isinstance(edgecolors, str):
        edgecolors = [edgecolors] * len(labels)

    if hatches is None or len(hatches) == 0 or style == "circle" or markers is not None:
        hatches = [""] * len(labels)

    if sizes is None or len(sizes) < len(labels):
        sizes = sizes or [resolve_param("lines.markersize")]
        sizes = [sizes[i % len(sizes)] for i in range(len(labels))]

    if markers is not None:
        if isinstance(markers, str):
            markers = [markers] * len(labels)
        if len(markers) == 0:
            markers = None

    if linestyles is not None and len(linestyles) < len(labels):
        linestyles = linestyles or [resolve_param("lines.linestyle")]
        linestyles = [linestyles[i % len(linestyles)] for i in range(len(labels))]

    handles = []

    # Determine patch type
    if markers is not None and linestyles is not None:
        # Use LineMarkerPatch when both markers and linestyles are specified
        for label, col, edge_col, size, marker, linestyle in zip(labels, colors, edgecolors, sizes, markers, linestyles):
            handle = LineMarkerPatch(
                marker=marker,
                linestyle=linestyle,
                facecolor=col,
                edgecolor=edge_col,
                alpha=alpha,
                linewidth=linewidth_line,
                label=label,
                markersize=size,
                markeredgewidth=markeredgewidth,
            )
            handles.append(handle)
    elif markers is not None:
        # Use MarkerPatch when only markers are specified
        for label, col, edge_col, hatch, size, marker in zip(labels, colors, edgecolors, hatches, sizes, markers):
            handle = MarkerPatch(
                marker=marker,
                facecolor=col,
                edgecolor=edge_col,
                alpha=alpha,
                linewidth=linewidth_outline,
                label=label,
                markersize=size,
                markeredgewidth=markeredgewidth,
            )
            handles.append(handle)
    elif linestyles is not None or style == "line":
        # Line-only swatch: one horizontal colored line per entry. Used by
        # lineplot for hue (solid line, distinguish by color) and style
        # (distinguish by dash pattern).
        if linestyles is None:
            linestyles = ["-"] * len(labels)
        for label, col, edge_col, linestyle in zip(labels, colors, edgecolors, linestyles):
            handle = LinePatch(
                linestyle=linestyle,
                facecolor=col,
                edgecolor=edge_col,
                alpha=alpha,
                linewidth=linewidth_line,
                label=label,
            )
            handles.append(handle)
    else:
        # Use MarkerPatch for circles, RectanglePatch for rectangles
        if style == "circle":
            # Circle is just a marker with 'o' symbol
            for label, col, edge_col, hatch, size in zip(labels, colors, edgecolors, hatches, sizes):
                handle = MarkerPatch(
                    marker='o',
                    facecolor=col,
                    edgecolor=edge_col,
                    alpha=alpha,
                    linewidth=linewidth_outline,
                    label=label,
                    markersize=size,
                    markeredgewidth=markeredgewidth,
                )
                handles.append(handle)
        else:
            # Rectangle patches (for bar plots with hatches)
            for label, col, edge_col, hatch, size in zip(labels, colors, edgecolors, hatches, sizes):
                handle = RectanglePatch(
                    facecolor=col,
                    edgecolor=edge_col,
                    alpha=alpha,
                    linewidth=linewidth_outline,
                    label=label,
                    hatch=hatch,
                    markersize=size,
                )
                handles.append(handle)

    return handles


def compute_min_labelspacing(
    handles: List,
    fontsize: float,
    default: float = 0.3,
    breathing: float = 0.5,
) -> float:
    """Return a ``labelspacing`` (font-size units) large enough to avoid
    row overlap given the tallest handle in ``handles``.

    Matplotlib's legend packs rows using a fixed-height handle slot of
    ``fontsize * handleheight`` (~4.9 pt at the 7 pt default font).
    Oversized markers overflow that slot on both sides and bleed into
    adjacent rows when ``labelspacing`` is a small constant. We model the
    required spacing directly:

    ::

        row_center_to_center ≈ fontsize * (1 + labelspacing)
        row_center_to_center ≥ tallest_marker + gap

    → ``labelspacing ≥ (marker / fontsize) - 1 + (gap / fontsize)``.

    With ``breathing`` as ``gap / fontsize``, the edge-to-edge clearance
    between markers stays constant at every size — both the smallest
    and largest adjacent swatches get the same whitespace.

    Handles without a ``get_markersize`` method don't count as oversized,
    so text-only legends stay at ``default``.

    Parameters
    ----------
    handles : list
        Legend handles (MarkerPatch, LineMarkerPatch, LinePatch,
        RectanglePatch, ...). Each may or may not carry a markersize.
    fontsize : float
        Legend text font size in points.
    default : float, default=0.3
        Baseline matplotlib labelspacing used for text-only legends.
    breathing : float, default=0.5
        Edge-to-edge clearance between markers in font-size units.
        0.5 at the 7 pt default font = 3.5 pt of whitespace between
        adjacent swatches.

    Returns
    -------
    float
        ``labelspacing`` in font-size units; always ``>= default``.
    """
    tallest_pt = 0.0
    for h in handles:
        if hasattr(h, "get_markersize"):
            ms = h.get_markersize()
            if ms is not None and ms > tallest_pt:
                tallest_pt = float(ms)

    if tallest_pt <= fontsize:
        return default

    required = (tallest_pt / fontsize) - 1.0 + breathing
    return max(default, required)


# =============================================================================
# Legend Builder (Primary Interface)
# =============================================================================


class _AxesFractionLocator:
    """Axes locator pinning an axes to a rectangle of a parent's axes fraction.

    ``Axes.inset_axes`` installs an equivalent (private) locator; this one
    exists so an inset's rectangle can be *replaced* after the fact —
    :meth:`LegendBuilder._nudge_inside_cbar` measures the drawn colorbar
    and slides it back inside the parent. Being a locator rather than a
    fixed position, the rectangle is re-solved against the parent on
    every draw, so the strip follows the axes without any
    ``LayoutReactor`` registration.
    """

    def __init__(self, parent: Axes, bounds: Tuple[float, float, float, float]):
        self._parent = parent
        self.bounds = tuple(bounds)

    def __call__(self, ax: Axes, renderer):
        from matplotlib.transforms import Bbox, TransformedBbox
        fig = self._parent.get_figure()
        bbox = TransformedBbox(
            Bbox.from_bounds(*self.bounds), self._parent.transAxes
        )
        return bbox.transformed(fig.transSubfigure.inverted())


class LegendBuilder:
    """
    Publication-ready legend builder with automatic column overflow.

    **All dimensions are in millimeters** for precise positioning in
    publication-quality plots. The builder automatically creates new
    columns when vertical space is exhausted.

    This is the primary interface for creating legends in publiplots.

    Parameters
    ----------
    ax : Axes
        Main plot axes to attach legends to.
    x_offset : float, default=2
        Horizontal distance from the right edge of axes (millimeters).
    y_offset : float, optional
        Vertical position from top of axes (millimeters). If None, starts at
        axes height minus vpad.
    gap : float, default=2
        Vertical spacing between legend elements (millimeters).
    column_spacing : float, default=5
        Horizontal spacing between columns (millimeters).
    vpad : float, default=5
        Padding from top of axes (millimeters).
    max_width : float, optional
        Maximum width for legends (millimeters). If None, auto-estimated from content.

    Examples
    --------
    >>> ax = pp.scatterplot(df, x='x', y='y', hue='group', legend=False)
    >>> builder = pp.legend(ax, auto=False, x_offset=2, gap=2)
    >>> builder.add_legend(handles, label="Treatment")
    >>> builder.add_colorbar(mappable, label="Expression", height=15)

    Notes
    -----
    All dimensions are in millimeters. New columns are created automatically
    when vertical space is exhausted.
    """

    # Conversion constants
    MM2INCH = 1 / 25.4
    PT2MM = 25.4 / 72

    def __init__(
        self,
        ax: Axes,
        x_offset: float = 2,
        y_offset: Optional[float] = None,
        gap: float = 2,
        column_spacing: float = 5,
        vpad: Optional[float] = None,
        max_width: Optional[float] = None,
        anchor_ax: Optional[Axes] = None,
        external_to_axis: bool = False,
        side: str = "right",
        orientation: str = "vertical",
    ):
        """Initialize legend builder. All dimensions in millimeters.

        Parameters
        ----------
        ax : Axes
            Axes the legend/colorbar artist is attached to (for picking,
            ``ax.legend_`` association, etc.).
        anchor_ax : Axes, optional
            Axes whose chosen edge is used as the origin for mm-based
            placement math and for reactor registration. Defaults to
            ``ax``. Used by MultiAxesLegendGroup to attach artists to one
            axes while positioning them relative to another (or to a
            virtual grid anchor for figure-anchored groups).
        side : {'right', 'left', 'bottom', 'top'}, default 'right'
            Which edge of ``anchor_ax`` the legend grows outward from.
            'right' matches the historical placement (columns fill
            rightward, rows downward).
        orientation : {'vertical', 'horizontal'}, default 'vertical'
            Primary stacking direction of successive legends and of
            entries within each legend. ``'vertical'`` stacks entries
            downward and advances successive legends downward; overflow
            (exhausted along-edge length) starts a new band *outward*.
            ``'horizontal'`` lays entries along the edge (default
            ``ncol = len(handles)``) and advances successive legends
            rightward; overflow starts a new band further outward.
        """
        from publiplots.utils.legend_layout import LegendLayout
        from publiplots.utils.layout_reactor import LayoutReactor

        if side not in ("right", "left", "bottom", "top"):
            raise ValueError(
                f"side must be 'right' | 'left' | 'bottom' | 'top', got {side!r}"
            )
        if orientation not in ("vertical", "horizontal"):
            raise ValueError(
                f"orientation must be 'vertical' | 'horizontal', got {orientation!r}"
            )

        self.ax = ax
        self._anchor_ax = anchor_ax if anchor_ax is not None else ax
        self.fig = self._anchor_ax.get_figure()
        self._side = side
        self._orientation = orientation

        # Default vpad: when the anchor is a real Axes (per-axis legend
        # or axes-anchored legend_group), vpad=0 lands the legend's top
        # flush with the axes rectangle top — title_space lives above
        # axes.y1 so it's already accounted for by pp.subplots. When
        # the anchor is a _GridAnchor (figure-anchored group),
        # ``anchor.y1`` is the decorated-grid top which sits ABOVE every
        # axes' title_space; vpad=5 keeps the legend from hugging that
        # top border visually.
        if vpad is None:
            from publiplots.utils.legend_group import _GridAnchor
            vpad = 5 if isinstance(self._anchor_ax, _GridAnchor) else 0

        self._layout = LegendLayout(
            x_offset=x_offset,
            y_offset=y_offset,
            gap=gap,
            column_spacing=column_spacing,
            vpad=vpad,
            max_width=max_width,
            orientation=orientation,
        )
        self._layout.reset_to(edge_length_mm=self._get_edge_length())
        self._reactor = LayoutReactor.get(self.fig)
        self._external_to_axis = external_to_axis
        # Element storage: list of (type, object) tuples
        self.elements = []
        # id(Colorbar) -> the floating ``Text`` that labels it. A colorbar
        # label is a standalone figure text with its own reactor
        # registration, so nothing else records which strip it belongs to.
        # ``MultiAxesLegendGroup._apply_along_alignment`` needs that link to
        # keep the pair together when the band holds more than one element
        # (#214). Keyed by ``id`` to match how that pass already identifies
        # reactor registrations; both objects stay alive in ``self.elements``
        # for as long as the mapping is consulted.
        self._colorbar_labels = {}
        # id(Colorbar) -> the ``label`` this strip was created with. For a
        # per-axes colorbar that label IS the LegendEntry name, which is
        # what ``MultiAxesLegendGroup._evict_claimed_per_axis_legends``
        # matches on when a band claims the entry and must drop the
        # already-rendered per-axes copy (#217). Recorded explicitly
        # rather than read back off the artist: a ``title_position='top'``
        # colorbar clears ``cbar.set_label`` and paints the name into a
        # separate object, so the strip itself no longer carries it.
        self._colorbar_names = {}

        # id(Colorbar) -> mm the strip's registered outward offset was
        # padded so its INWARD-hanging decorations (a horizontal strip's
        # tick labels on a ``side='top'`` band) clear the axes edge. The
        # pad sits between the band's base outward offset and the colour
        # rectangle, so the block's visible extent still starts at the
        # base — which is what ``_apply_along_alignment`` must key its
        # rows on, or the strip lands in a row of its own and gets
        # centred on top of a categorical legend sharing the band (#213).
        self._colorbar_inward_pad = {}

    def _get_edge_length(self) -> float:
        """Along-edge length of the anchor in mm.

        For side='right'|'left' that's the axes height; for
        side='top'|'bottom' it's the axes width. The ``LegendLayout``
        cursor treats this uniformly as the 'down the edge' distance
        available before overflowing to a new column.
        """
        if self._side in ("right", "left"):
            return self._get_axes_height()
        return self._get_axes_width()

    def _get_axes_width(self) -> float:
        ax_pos = self._anchor_ax.get_position()
        fig_width_px = self.fig.get_window_extent().width
        axes_width_px = ax_pos.width * fig_width_px
        return axes_width_px / self.fig.dpi / self.MM2INCH

    # =========================================================================
    # Conversion Utilities
    # =========================================================================

    def _get_axes_height(self) -> float:
        """Get axes height in millimeters."""
        ax_pos = self._anchor_ax.get_position()
        fig_height_px = self.fig.get_window_extent().height
        axes_height_px = ax_pos.height * fig_height_px
        return axes_height_px / self.fig.dpi / self.MM2INCH

    def _mm_to_figure_coords(self, x_mm: float, y_mm: float) -> Tuple[float, float]:
        """
        Convert mm position to figure coordinates at the chosen edge.

        Parameters
        ----------
        x_mm : float
            Outward distance from the edge (mm). Rightward for
            ``side='right'``, leftward for ``'left'``, downward for
            ``'bottom'``, upward for ``'top'``.
        y_mm : float
            Remaining along-edge space (mm). Converted to position from
            the starting corner internally.

        Returns
        -------
        x_fig, y_fig : float
            Position in figure coordinates.
        """
        ax_pos = self._anchor_ax.get_position()
        fig_extent = self.fig.get_window_extent()

        # y_mm represents remaining along-edge space; position_from_start
        # is how far we've advanced from the corner where the cursor
        # began (top for right/left, left for bottom/top).
        edge_length = self._get_edge_length()
        position_from_start = edge_length - y_mm

        if self._side in ("right", "left"):
            outward_frac = (x_mm * self.MM2INCH * self.fig.dpi) / fig_extent.width
            along_frac = (position_from_start * self.MM2INCH * self.fig.dpi) / fig_extent.height
            y_fig = ax_pos.y1 - along_frac
            x_fig = (ax_pos.x1 + outward_frac) if self._side == "right" \
                    else (ax_pos.x0 - outward_frac)
        else:  # bottom | top
            outward_frac = (x_mm * self.MM2INCH * self.fig.dpi) / fig_extent.height
            along_frac = (position_from_start * self.MM2INCH * self.fig.dpi) / fig_extent.width
            x_fig = ax_pos.x0 + along_frac
            y_fig = (ax_pos.y0 - outward_frac) if self._side == "bottom" \
                    else (ax_pos.y1 + outward_frac)

        return x_fig, y_fig

    def _fig_canvas_draw_for_measure(self) -> None:
        """Force a full figure redraw so artist window extents become
        current. Isolated into its own method so callers that are already
        mid-draw (e.g. the alignment pass, where matplotlib's renderer
        cache is already populated) can skip it — see
        ``_measure_object_dimensions(force_draw=...)``.
        """
        self.fig.canvas.draw()

    def _measure_object_dimensions(
        self,
        obj: Union[Legend, Colorbar, Any],
        force_draw: bool = True,
    ) -> Tuple[float, float]:
        """
        Measure actual dimensions of matplotlib object.

        Parameters
        ----------
        obj : Legend or Colorbar or Text
            Matplotlib object to measure
        force_draw : bool, default True
            When True, force a full ``fig.canvas.draw()`` first so the
            artist's cached window extent is current. Callers that are
            already inside a draw (e.g. the legend_group alignment pass,
            which runs as a post-refresh reactor callback) pass
            ``force_draw=False``: matplotlib's renderer cache is already
            populated at that point, so ``get_window_extent()`` returns
            sensible values without an O(panels) nested figure redraw.

        Returns
        -------
        width_mm, height_mm : float
            Object dimensions in millimeters
        """
        if force_draw:
            self._fig_canvas_draw_for_measure()

        # Get bounding box. Call get_window_extent() without a renderer:
        # the draw() above populates matplotlib's cached renderer on the
        # active canvas, and get_window_extent() falls back to that
        # cache when no renderer is passed. This keeps measurement
        # working on non-AGG canvases (PDF / PS / SVG), which don't
        # expose ``canvas.get_renderer()`` — previously a legend_group
        # that worked under PNG save crashed under PDF save. See #115.
        if hasattr(obj, 'ax'):  # Colorbar
            bbox = obj.ax.get_window_extent()
        elif hasattr(obj, 'get_window_extent'):
            bbox = obj.get_window_extent()
        else:
            return 0, 0

        # Convert pixels to mm
        width_mm = bbox.width / self.fig.dpi / self.MM2INCH
        height_mm = bbox.height / self.fig.dpi / self.MM2INCH

        return width_mm, height_mm

    # =========================================================================
    # Estimation Utilities (for overflow detection)
    # =========================================================================

    def _estimate_legend_height(
        self,
        handles: List,
        label: str,
        **kwargs
    ) -> float:
        """
        Estimate legend height before creation.

        Returns
        -------
        float
            Estimated height in millimeters
        """
        fontsize = resolve_param("legend.fontsize", resolve_param("font.size"))
        title_fontsize = resolve_param("legend.title_fontsize", fontsize)

        # Get legend parameters. If labelspacing isn't set, fall back to
        # the adaptive minimum so oversized markers are budgeted for.
        ncol = kwargs.get('ncol', 1)
        labelspacing = kwargs.get(
            'labelspacing',
            compute_min_labelspacing(handles, fontsize),
        )
        borderpad = resolve_param("legend.borderpad", kwargs.get('borderpad'))

        # Calculate rows
        n_items = len(handles)
        n_rows = (n_items + ncol - 1) // ncol  # Ceiling division

        # Title height
        title_height = (title_fontsize * self.PT2MM * 1.3) if label else 0

        # Items height (rows * item_height)
        item_height = fontsize * self.PT2MM
        spacing_height = (n_rows - 1) * labelspacing * fontsize * self.PT2MM
        items_height = n_rows * item_height + spacing_height

        # Padding (top + bottom)
        padding_height = 2 * borderpad * fontsize * self.PT2MM

        total = title_height + items_height + padding_height
        return total

    def _estimate_legend_width(
        self,
        handles: List,
        labels: Optional[List[str]] = None,
        **kwargs
    ) -> float:
        """
        Estimate legend width from text content.

        Returns
        -------
        float
            Estimated width in millimeters
        """
        if labels is None:
            labels = [h.get_label() for h in handles if hasattr(h, 'get_label')]

        if not labels:
            return 20  # Fallback default

        fontsize = resolve_param("legend.fontsize", resolve_param("font.size"))

        # Estimate character width (rough approximation)
        # Typical sans-serif: ~0.6 * fontsize per character
        max_label_length = max(len(str(label)) for label in labels)
        text_width = max_label_length * fontsize * 0.6 * self.PT2MM

        # Add space for handle
        handlelength = resolve_param(
            "legend.handlelength", kwargs.get('handlelength')
        )  # in font-size units
        handletextpad = resolve_param(
            "legend.handletextpad", kwargs.get('handletextpad')
        )
        handle_width = (handlelength + handletextpad) * fontsize * self.PT2MM

        # Add padding
        borderpad = resolve_param("legend.borderpad", kwargs.get('borderpad'))
        padding = 2 * borderpad * fontsize * self.PT2MM

        return handle_width + text_width + padding

    # =========================================================================
    # Band Management
    # =========================================================================

    def _check_overflow(self, required_along: float) -> bool:
        """True if an element of this along-edge size overflows the current band."""
        return self._layout.check_overflow(required_along)

    def _start_new_band(self):
        """Create a new band outward when along-edge space is exhausted."""
        self._layout.start_new_band()

    def _adjust_legend_ncol_for_height(
        self,
        handles: List,
        label: str,
        max_height: float,
        **kwargs
    ) -> int:
        """
        Auto-adjust ncol to fit within max_height (PyComplexHeatmap behavior).

        Returns
        -------
        int
            Optimal number of columns
        """
        ncol = kwargs.get('ncol', 1)
        max_ncol = 3  # Cap at 3 columns

        while ncol <= max_ncol:
            kwargs_test = kwargs.copy()
            kwargs_test['ncol'] = ncol
            estimated_height = self._estimate_legend_height(handles, label, **kwargs_test)

            if estimated_height <= max_height:
                return ncol

            ncol += 1

        # If still doesn't fit at max_ncol, return max_ncol and warn
        print(f"Warning: Legend too tall even with {max_ncol} columns")
        return max_ncol

    # =========================================================================
    # Main Methods
    # =========================================================================

    def add_legend(
        self,
        handles: List,
        label: str = "",
        frameon: bool = False,
        max_height: Optional[float] = None,
        **kwargs
    ) -> Legend:
        """
        Add a legend with automatic overflow handling.

        Creates a new column automatically if the legend doesn't fit
        in the current vertical space.

        Parameters
        ----------
        handles : list
            Legend handles (from create_legend_handles or plot objects).
        label : str
            Legend title.
        frameon : bool
            Whether to show frame around legend.
        max_height : float, optional
            Maximum height in millimeters. If legend exceeds this, increase ncol
            to fit (PyComplexHeatmap behavior).
        **kwargs
            Additional kwargs for legend customization. ``inside`` (bool,
            default ``False``) bypasses the mm-based outside-axes column and
            renders the legend inside the axes using matplotlib's native
            axes-relative placement; pair with ``loc='upper right'`` etc. to
            pick the corner. The rest (``ncol``, ``labelspacing``,
            ``handletextpad``, ``columnspacing``, etc.) are forwarded to
            ``ax.legend()``.

        Returns
        -------
        Legend
            The created legend object.

        Notes
        -----
        All dimensions in millimeters. Columns created automatically on overflow.
        """
        inside = bool(kwargs.pop("inside", False))

        if inside:
            # Inside-axes placement: let matplotlib own the geometry. Don't
            # touch the mm cursor, don't build figure-fraction bbox, don't
            # register with the reactor — the legend lives in axes coords
            # and tracks automatically across axes resizes.
            fontsize = resolve_param("legend.fontsize", resolve_param("font.size"))
            default_labelspacing = compute_min_labelspacing(handles, fontsize)
            legend_kwargs = {
                "frameon": frameon,
                "borderpad": resolve_param("legend.borderpad", None),
                "handletextpad": resolve_param("legend.handletextpad", None),
                "labelspacing": default_labelspacing,
                "handler_map": kwargs.pop("handler_map", get_legend_handler_map()),
                "alignment": "left",
            }
            if label:
                legend_kwargs["title"] = label
            legend_kwargs.update(kwargs)
            # ax.legend() replaces ax.legend_ and evicts earlier Legend
            # children. Preserve prior legends built by this same builder
            # (so multiple add_legend calls stack) AND any legend left on
            # self.ax by a different builder (notably pp.legend_group
            # having attached an outside-right collected entry to this
            # same axes).
            prior = [e[1] for e in self.elements
                     if e[0] == "legend" and e[1].axes is self.ax]
            prior_ids = {id(p) for p in prior}
            if self.ax.legend_ is not None and id(self.ax.legend_) not in prior_ids:
                prior.append(self.ax.legend_)
            legend = self.ax.legend(handles=handles, **legend_kwargs)
            # Exclude the inside-axes legend from layout/tightbbox math.
            # Without this, SubplotsAutoLayout's tightbbox-based per-cell
            # measurement picks up the legend's extent and grows the
            # cell, displacing siblings in the same row/column. Matches
            # matplotlib's own loc=... legend semantics. (Fixes #180.)
            legend.set_in_layout(False)
            for p in prior:
                if p is not legend:
                    self.ax.add_artist(p)
            self.elements.append(("legend", legend))
            return legend

        # Horizontal orientation default: lay out every handle in a
        # single row. User-provided ncol wins.
        if self._orientation == "horizontal" and "ncol" not in kwargs:
            kwargs["ncol"] = max(1, len(handles))

        # Auto-adjust ncol if max_height specified
        if max_height is not None:
            optimal_ncol = self._adjust_legend_ncol_for_height(
                handles, label, max_height, **kwargs
            )
            kwargs['ncol'] = optimal_ncol

        # Estimate the legend's along-edge extent: height when stacking
        # vertically, width when laying out horizontally.
        if self._orientation == "horizontal":
            estimated_along = self._estimate_legend_width(handles, label, **kwargs)
        else:
            estimated_along = self._estimate_legend_height(handles, label, **kwargs)

        # Check overflow
        if self._check_overflow(estimated_along):
            self._start_new_band()

        # Convert current position to figure coordinates
        x_fig, y_fig = self._mm_to_figure_coords(
            self._layout.current_outward, self._layout.current_along
        )

        # Adaptive row spacing: when a handle's marker exceeds the font
        # size (scatter size legends, over-sized circle swatches) the
        # default ``labelspacing=0.3`` produces overlapping rows. Widen
        # the spacing to fit the tallest handle unless the caller set
        # ``labelspacing`` explicitly.
        fontsize = resolve_param("legend.fontsize", resolve_param("font.size"))
        default_labelspacing = compute_min_labelspacing(handles, fontsize)

        # Prepare legend kwargs. ``loc`` maps to the matplotlib corner of
        # the legend box that coincides with bbox_to_anchor — pick it so
        # the legend grows *away from* the anchor edge:
        #   side='right'  → anchor sits at legend's upper-left corner.
        #   side='left'   → anchor sits at legend's upper-right corner.
        #   side='bottom' → anchor sits at legend's upper-left corner (cursor rotates).
        #   side='top'    → anchor sits at legend's lower-left corner.
        loc_by_side = {
            "right": "upper left",
            "left": "upper right",
            "bottom": "upper left",
            "top": "lower left",
        }
        legend_kwargs = {
            "loc": loc_by_side[self._side],
            "bbox_to_anchor": (x_fig, y_fig),
            "bbox_transform": self.fig.transFigure,  # Use figure coords
            "frameon": frameon,
            "borderaxespad": 0,
            "borderpad": resolve_param("legend.borderpad", None),
            "handletextpad": resolve_param("legend.handletextpad", None),
            "labelspacing": default_labelspacing,
            "handler_map": kwargs.pop("handler_map", get_legend_handler_map()),
            "alignment": "left",
        }

        if label:
            legend_kwargs['title'] = label

        legend_kwargs.update(kwargs)

        # Create legend. ax.legend() clears prior Legend children; preserve
        # both legends we built earlier on this axes AND a legend left by
        # a different builder (notably an inside-axes legend that lives on
        # the same axes as a pp.legend_group anchor).
        prior = [e[1] for e in self.elements
                 if e[0] == "legend" and e[1].axes is self.ax]
        prior_ids = {id(p) for p in prior}
        if self.ax.legend_ is not None and id(self.ax.legend_) not in prior_ids:
            prior.append(self.ax.legend_)
        legend = self.ax.legend(handles=handles, **legend_kwargs)
        legend.set_clip_on(False)
        for p in prior:
            if p is not legend:
                self.ax.add_artist(p)

        # Measure actual dimensions
        width, height = self._measure_object_dimensions(legend)

        # Capture position BEFORE advancing the cursor — this is where the
        # legend was actually placed, and what the reactor needs.
        placement_x_mm = self._layout.current_outward
        # mm_y_from_top is tracked directly in the layout (stable across
        # axes height changes from constrained_layout etc).
        mm_y_from_top = self._layout.along_from_start

        # Update layout cursor for the next element.
        # - vertical: height advances along the edge, width is the band's
        #   outward extent.
        # - horizontal: width advances along the edge, height is the
        #   band's outward extent.
        if self._orientation == "horizontal":
            self._layout.update_width(height)
            self._layout.advance_along(width)
        else:
            self._layout.update_width(width)
            self._layout.advance_along(height)

        # Register with the reactor so the anchor follows axes changes.
        self._reactor.register(
            ax=self._anchor_ax,
            artist=legend,
            mm_x_from_right=placement_x_mm,
            mm_y_from_top=mm_y_from_top,
            side=self._side,
            external_to_axis=self._external_to_axis,
        )

        # Store element
        self.elements.append(("legend", legend))

        return legend

    # Matplotlib's legend ``loc='best'`` searches for the emptiest corner
    # using the legend's own handles; a colorbar strip has no equivalent
    # search, so the inside path resolves 'best' to a fixed corner rather
    # than raising. Not a matplotlib precedent: the ``loc = 'upper right'``
    # rewrite in ``Legend.set_loc`` sits inside its ``if loc is None``
    # branch, so it fires only for a default taken from
    # ``rcParams['legend.loc']`` — an *explicit* ``loc=0`` on a figure
    # legend raises "Automatic legend placement (loc='best') not
    # implemented". Resolving is the deliberate choice here, because a
    # strip has no handles to search around and raising would defeat the
    # point of accepting the code at all (#223).
    _INSIDE_CBAR_DEFAULT_LOC = "upper right"

    # Padding between the strip and the axes edge, in mm. Matches the
    # visual weight of matplotlib's ``legend.borderaxespad`` at the
    # publiplots default font size without inheriting its font units.
    _INSIDE_CBAR_PAD_MM = 2.0

    # Integer location code -> name, inverted from the mapping the
    # *installed* matplotlib actually uses (``Legend.codes``, itself
    # ``{'best': 0, **AnchoredOffsetbox.codes}``) rather than a copy of
    # it, so an integer here always resolves to the corner
    # ``ax.legend(loc=<code>)`` would pick. Codes 5 ('right') and 7
    # ('center right') are distinct names that anchor identically —
    # ``offsetbox._get_anchored_bbox`` maps both to "E".
    _INSIDE_CBAR_LOC_NAMES = {code: name for name, code in Legend.codes.items()}

    @classmethod
    def _inside_cbar_loc_error(cls, loc: Any) -> ValueError:
        """Build the shared ``loc`` rejection message."""
        codes = sorted(cls._INSIDE_CBAR_LOC_NAMES)
        return ValueError(
            "inside colorbar loc must be one of 'upper|center|lower' "
            "+ 'left|center|right' (or 'center', 'right', 'best'), or a "
            f"matplotlib location code {codes[0]}-{codes[-1]}, got {loc!r}"
        )

    @classmethod
    def _inside_cbar_anchor(cls, loc: Union[str, int]) -> Tuple[str, str]:
        """Split a matplotlib ``loc`` into (vertical, horizontal).

        Accepts the nine axes-relative legend location strings, the bare
        ``'right'`` alias matplotlib keeps for ``'center right'``, and
        matplotlib's integer location codes — so a hue column switched
        from categorical to continuous keeps the placement it had when
        the same ``loc`` went to ``ax.legend()`` (#223).

        ``'best'``/``0`` has no meaning for a strip (there are no handles
        to search around) and resolves to
        :attr:`_INSIDE_CBAR_DEFAULT_LOC`.
        """
        if isinstance(loc, int):
            # ``bool`` is an ``int`` here exactly as it is in
            # matplotlib's own ``isinstance(loc, int)`` validation, so
            # ``loc=True`` means code 1 in both paths.
            if loc not in cls._INSIDE_CBAR_LOC_NAMES:
                raise cls._inside_cbar_loc_error(loc)
            loc = cls._INSIDE_CBAR_LOC_NAMES[loc]
        if not isinstance(loc, str):
            raise cls._inside_cbar_loc_error(loc)
        if loc == "best":
            loc = cls._INSIDE_CBAR_DEFAULT_LOC
        if loc == "center":
            return "center", "center"
        if loc == "right":
            return "center", "right"
        parts = loc.split()
        if len(parts) != 2:
            raise cls._inside_cbar_loc_error(loc)
        vertical, horizontal = parts
        if vertical not in ("upper", "center", "lower") or \
                horizontal not in ("left", "center", "right"):
            raise cls._inside_cbar_loc_error(loc)
        return vertical, horizontal

    def _nudge_inside_cbar(self, cbar_ax, bounds, pad_mm: float) -> None:
        """Slide an inside colorbar back inside the axes if its decorations spill.

        ``bounds`` positions the *strip*; its tick labels (and the label
        drawn above it) hang off that rectangle, so an anchored corner
        can push them past the axes edge — a right-anchored strip is the
        common case, since the tick labels sit to its right. Measuring
        the drawn tightbbox and translating the rectangle is enough:
        the strip keeps its requested mm size and only moves.
        """
        self._fig_canvas_draw_for_measure()
        tight = cbar_ax.get_tightbbox()
        if tight is None:
            return
        ax_bbox = self.ax.get_window_extent()
        if not (ax_bbox.width and ax_bbox.height):
            return
        pad_px = pad_mm * self.MM2INCH * self.fig.dpi

        def _shift(lo, hi, ax_lo, ax_hi):
            # Pull the high edge in first, then the low edge; a
            # decoration block wider than the axes stays left/bottom
            # aligned rather than oscillating.
            delta = min(0.0, (ax_hi - pad_px) - hi)
            if lo + delta < ax_lo + pad_px:
                delta = (ax_lo + pad_px) - lo
            return delta

        dx = _shift(tight.x0, tight.x1, ax_bbox.x0, ax_bbox.x1)
        dy = _shift(tight.y0, tight.y1, ax_bbox.y0, ax_bbox.y1)
        if dx or dy:
            x0, y0, w, h = bounds
            cbar_ax.set_axes_locator(_AxesFractionLocator(
                self.ax,
                (x0 + dx / ax_bbox.width, y0 + dy / ax_bbox.height, w, h),
            ))

    def _add_colorbar_inside(
        self,
        *,
        mappable: Optional[ScalarMappable],
        label: str,
        height: float,
        width: float,
        title_position: str,
        orientation: str,
        ticks: Optional[List[float]],
        center: Optional[float],
        vmin: Optional[float],
        vmax: Optional[float],
        **kwargs,
    ) -> Colorbar:
        """Render the colorbar inside ``self.ax`` instead of in an outside band.

        The counterpart of the ``inside=True`` branch of
        :meth:`add_legend`. Both hand placement to matplotlib's own
        axes-relative machinery: the legend via ``ax.legend(loc=...)``,
        the colorbar via ``ax.inset_axes``, whose rectangle is expressed
        in axes fractions and therefore re-solved from the parent axes'
        bbox on every draw. That means no mm cursor, no overflow band,
        and no :class:`~publiplots.utils.layout_reactor.LayoutReactor`
        registration — the strip tracks the axes by itself.

        ``height``/``width`` stay mm, converted against the *axes*
        rectangle rather than the figure. That is what keeps the strip
        the requested size through the first draw: ``pp.subplots`` pins
        the axes at ``axes_size`` mm and re-sizes the *figure* around it,
        so an axes-fraction rectangle is stable in mm where a
        figure-relative one would be scaled by the mid-draw resize.

        Like the inside legend, the strip is marked ``in_layout=False``
        so ``SubplotsAutoLayout``'s tightbbox measurement doesn't grow the
        cell around it (same reasoning as #180).
        """
        loc = kwargs.pop("loc", self._INSIDE_CBAR_DEFAULT_LOC)
        vertical, horizontal = self._inside_cbar_anchor(loc)
        # Deliberately not user-tunable through ``borderpad``: in the
        # legend family that key is the padding *inside* the legend
        # frame, in font-size units. A strip has no frame, and the pad
        # here is the mm distance to the axes edge — matplotlib spells
        # that one ``borderaxespad``. Honouring ``borderpad`` would give
        # a single ``legend_kws`` key two meanings in two units on a
        # plot that draws both a legend and a colorbar.
        pad_mm = self._INSIDE_CBAR_PAD_MM

        # mm -> axes fractions, against the axes the strip lands in
        # (``self.ax``; a group in inside mode retargets it per call).
        ax_pos = self.ax.get_position()
        fig_extent = self.fig.get_window_extent()
        ax_width_mm = (
            ax_pos.width * fig_extent.width / self.fig.dpi / self.MM2INCH
        )
        ax_height_mm = (
            ax_pos.height * fig_extent.height / self.fig.dpi / self.MM2INCH
        )
        w_frac = width / ax_width_mm
        h_frac = height / ax_height_mm
        pad_x = pad_mm / ax_width_mm
        pad_y = pad_mm / ax_height_mm

        if horizontal == "left":
            x0 = pad_x
        elif horizontal == "right":
            x0 = 1 - pad_x - w_frac
        else:
            x0 = (1 - w_frac) / 2
        if vertical == "lower":
            y0 = pad_y
        elif vertical == "upper":
            y0 = 1 - pad_y - h_frac
        else:
            y0 = (1 - h_frac) / 2

        cbar_ax = self.ax.inset_axes([x0, y0, w_frac, h_frac])
        cbar_ax.set_xmargin(0)
        cbar_ax.set_ymargin(0)
        cbar_ax.set_in_layout(False)

        cbar = self.fig.colorbar(
            mappable,
            cax=cbar_ax,
            orientation=orientation,
            **kwargs,
        )

        # Outside bands draw the label as a free-standing fig.text and
        # re-anchor it every draw; here the inset's own title rides along
        # with the strip for free.
        if label:
            if title_position == "top":
                cbar_ax.set_title(
                    label,
                    fontsize=resolve_param(
                        "legend.title_fontsize", resolve_param("font.size")
                    ),
                    fontweight="normal",
                )
                cbar.set_label("")
            else:
                cbar.set_label(label)

        if ticks is not None:
            cbar.set_ticks(ticks)
        elif center is not None and vmin is not None and vmax is not None:
            cbar.set_ticks([vmin, center, vmax])

        self._nudge_inside_cbar(cbar_ax, [x0, y0, w_frac, h_frac], pad_mm)

        self.elements.append(("colorbar", cbar))
        self._colorbar_names[id(cbar)] = label
        return cbar

    def add_colorbar(
        self,
        mappable: Optional[ScalarMappable] = None,
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        center: Optional[float] = None,
        label: str = "",
        height: Optional[float] = None,
        width: Optional[float] = None,
        title_position: str = "top",
        orientation: Optional[str] = None,
        ticks: Optional[List[float]] = None,
        **kwargs
    ) -> Colorbar:
        """
        Add a colorbar with automatic overflow handling.

        Supports both ScalarMappable input (standard matplotlib) and
        direct colormap specification (PyComplexHeatmap style).

        Parameters
        ----------
        mappable : ScalarMappable, optional
            Existing ScalarMappable object (standard matplotlib usage).
        cmap : str, optional
            Colormap name (alternative to mappable, PyComplexHeatmap style).
            If provided, creates ScalarMappable internally.
        vmin, vmax : float, optional
            Value range for colormap (used with cmap parameter).
        center : float, optional
            Center value for divergent colormaps. Uses TwoSlopeNorm
            for proper centering (e.g., 0 for red-white-blue).
        label : str
            Colorbar label/title.
        height : float, optional
            Colorbar height (the *vertical* extent) in millimeters. The
            meaning is literal at every orientation; only the default
            follows it — 15 for a vertical strip, 4.5 for a horizontal
            one.
        width : float, optional
            Colorbar width (the *horizontal* extent) in millimeters.
            Literal at every orientation, like ``height``; defaults to
            4.5 for a vertical strip and 15 for a horizontal one.
        title_position : {'top', 'right'}, default='top'
            Position of title. 'top' places label above colorbar
            (horizontal), 'right' uses matplotlib default (vertical).
        orientation : {'vertical', 'horizontal'}, optional
            Colorbar orientation. ``None`` (default) derives it from the
            band it lands in: horizontal on a ``side='top'``/
            ``'bottom'`` band, vertical on ``'left'``/``'right'``. An
            explicit value always wins.
        ticks : list of float, optional
            Custom tick positions. If None and center is provided,
            automatically sets ticks at [vmin, center, vmax].
        **kwargs
            Additional kwargs passed to fig.colorbar(). ``inside`` (bool,
            default ``False``) bypasses the mm-based outside-axes band and
            renders the strip inside the axes rectangle, mirroring
            ``add_legend(inside=True)``; pair with ``loc='upper right'``
            etc. to pick the corner. ``loc`` takes the nine position
            strings ``ax.legend()`` accepts, the bare ``'right'`` alias,
            and matplotlib's integer codes 0-10, and resolves each to the
            corner the categorical legend would use. ``'best'``/``0``
            has no meaning for a strip and resolves to ``'upper right'``.
            ``height``/``width`` keep their mm meaning there, and the
            strip is excluded from layout math.

        Returns
        -------
        Colorbar
            The created colorbar object.

        Notes
        -----
        All dimensions in millimeters. Columns created automatically on overflow.

        Examples
        --------
        Standard matplotlib style:
        >>> builder.add_colorbar(sm, label="Values", height=20)

        PyComplexHeatmap style with divergent colormap:
        >>> builder.add_colorbar(
        ...     cmap='RdBu_r', vmin=-2, vmax=2, center=0,
        ...     label="Log2 FC", ticks=[-2, 0, 2]
        ... )
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm, Normalize
        from matplotlib.cm import ScalarMappable as SM

        # Create mappable if cmap provided (PyComplexHeatmap style)
        if mappable is None and cmap is not None:
            # ``plt.get_cmap`` accepts a name or a Colormap instance and is
            # the supported replacement for ``matplotlib.cm.get_cmap``, which
            # was removed in matplotlib 3.9.
            cmap_obj = plt.get_cmap(cmap)
            if center is not None:
                norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
            else:
                norm = Normalize(vmin=vmin, vmax=vmax)
            mappable = SM(norm=norm, cmap=cmap_obj)

        # Orientation follows the band's own axis unless the caller names
        # one (#213). ``self._orientation`` is the ALREADY-RESOLVED band
        # orientation: ``MultiAxesLegendGroup._DEFAULT_ORIENTATION`` maps
        # top/bottom -> 'horizontal' and left/right -> 'vertical' before
        # constructing this builder. Deriving from it rather than from
        # ``self._side`` is what makes an explicit
        # ``pp.legend(side='bottom', orientation='vertical')`` still
        # produce a vertical strip — a vertical band should hold a
        # vertical strip. A bare ``LegendBuilder`` (the ``inside=True``
        # plot path) stays 'vertical', which is its historical default.
        if orientation is None:
            orientation = self._orientation
        elif orientation not in ("vertical", "horizontal"):
            raise ValueError(
                f"orientation must be None | 'vertical' | 'horizontal', "
                f"got {orientation!r}"
            )

        # ``height``/``width`` keep their LITERAL mm meaning at every
        # orientation — ``height`` is the vertical extent, ``width`` the
        # horizontal one. Only the DEFAULTS swap, so a horizontal strip
        # comes out flat and wide instead of tall and narrow. Both
        # parameters default to ``None`` precisely so this can tell "the
        # caller said nothing" from "the caller passed 15", which is the
        # only way to swap a default without also overriding an explicit
        # value.
        if orientation == "horizontal":
            default_height, default_width = 4.5, 15.0
        else:
            default_height, default_width = 15.0, 4.5
        if height is None:
            height = default_height
        if width is None:
            width = default_width

        # Inside-axes placement short-circuit, the colorbar counterpart of
        # ``add_legend(inside=True)``. Everything below this point is the
        # outside-band machinery (mm cursor, overflow bands, reactor
        # registration) and none of it applies to a strip that lives in
        # the axes rectangle. (Fixes #215.)
        if bool(kwargs.pop("inside", False)):
            return self._add_colorbar_inside(
                mappable=mappable,
                label=label,
                height=height,
                width=width,
                title_position=title_position,
                orientation=orientation,
                ticks=ticks,
                center=center,
                vmin=vmin,
                vmax=vmax,
                **kwargs,
            )

        # Which cursor axis carries the label -> strip stack.
        #
        # ``LegendLayout``'s cursor is (outward, along): *outward* runs
        # AWAY from the anchor edge, *along* runs down the edge's tangent
        # — see ``layout_reactor._Registration`` for the per-side mapping
        # (its ``mm_x_from_right`` is outward, ``mm_y_from_top`` is along,
        # both historical names). "The label sits above the strip" is a
        # statement about *screen* geometry, so which cursor axis carries
        # it depends on the side:
        #   side='right'/'left' — along runs downward, so the strip sits
        #       further ALONG than the label (the historical layout).
        #   side='top'    — outward runs upward, so the label sits further
        #       OUTWARD than the strip; both share one along slot.
        #   side='bottom' — outward runs downward, so the strip sits
        #       further OUTWARD than the label; both share one along slot.
        # Stacking top/bottom bands along the edge instead is what made a
        # per-axes top colorbar render its strip sideways and its label
        # over the axes (#203).
        stack_outward = self._side in ("top", "bottom")

        # Estimate title height for overflow check
        title_pad = 2  # mm
        title_obj = None
        if title_position == "top" and label:
            fontsize = resolve_param("legend.title_fontsize", resolve_param("font.size"))
            estimated_title_height = fontsize * self.PT2MM * 1.3
            total_estimated_height = height + estimated_title_height + title_pad
        else:
            total_estimated_height = height

        # Check overflow using estimate. Overflow is about the *along-edge*
        # budget: for right/left the whole stack consumes it, but for
        # top/bottom the stack grows outward and only the strip's width
        # does. (The label may be wider than the strip; it isn't measured
        # until it exists, and this is only the pre-flight estimate — the
        # cursor advance below uses the real widths.)
        if self._check_overflow(width if stack_outward else total_estimated_height):
            self._start_new_band()

        # Add title if needed and measure actual height
        title_height_actual = 0
        title_width_actual = 0
        if title_position == "top" and label:
            # On a top band the strip is the element closest to the axes,
            # so the label has to step outward past it (plus the pad).
            title_outward_mm = self._layout.current_outward
            if self._side == "top":
                title_outward_mm += height + title_pad
            x_fig, y_fig = self._mm_to_figure_coords(
                title_outward_mm, self._layout.current_along
            )
            title_obj = self.fig.text(
                x_fig, y_fig, label,
                ha="left",
                # ``_mm_to_figure_coords`` returns the point on the outward
                # line; the label must grow AWAY from the axes from there.
                # A top band grows upward, so anchor its bottom; every
                # other side grows downward, so anchor its top.
                va="bottom" if self._side == "top" else "top",
                fontsize=resolve_param("legend.title_fontsize", resolve_param("font.size")),
                fontweight="normal"
            )

            # Measure actual title dimensions
            title_width_actual, title_height_actual = self._measure_object_dimensions(title_obj)

            # On a top/bottom band the label and the strip share ONE
            # along-edge slot (they are stacked outward), and the slot is
            # as wide as whichever of the two is wider. Centre each of
            # them inside it so the label sits over its own strip. Without
            # this the two are merely left-aligned, which reads as a
            # 6.5mm offset for a label wider than the 4.5mm strip and is
            # what a band using ``align='start'`` renders (#214).
            if stack_outward:
                title_along_shift = (
                    max(width, title_width_actual) - title_width_actual
                ) / 2
            else:
                title_along_shift = 0.0
            if title_along_shift:
                x_fig, y_fig = self._mm_to_figure_coords(
                    title_outward_mm,
                    self._layout.current_along - title_along_shift,
                )
                title_obj.set_position((x_fig, y_fig))

            # The reactor registration is deliberately deferred to after
            # the strip exists: a horizontal strip on a top band shifts
            # the WHOLE block outward by its inward tick-label overhang
            # (see below), and that overhang can only be measured once
            # the colorbar has been drawn. Registering here would pin the
            # label to the pre-shift outward line.
            title_mm_y_from_top = self._layout.along_from_start + title_along_shift

        # Place the strip relative to the measured label: further outward
        # on a bottom band, further along the edge on right/left, and
        # unmoved on a top band (there the label was lifted above it).
        cbar_outward_mm = self._layout.current_outward
        cbar_y_start = self._layout.current_along
        if title_height_actual:
            if self._side == "bottom":
                cbar_outward_mm += title_height_actual + title_pad
            elif not stack_outward:
                cbar_y_start -= title_height_actual + title_pad

        # Counterpart of the label's shift above: centre the strip in the
        # shared along-edge slot too. Zero unless the label is wider than
        # the strip on a top/bottom band.
        if stack_outward and title_width_actual:
            cbar_along_shift = (max(width, title_width_actual) - width) / 2
        else:
            cbar_along_shift = 0.0
        cbar_y_start -= cbar_along_shift

        # Create colorbar axes
        x_fig, y_fig = self._mm_to_figure_coords(cbar_outward_mm, cbar_y_start)

        fig_extent = self.fig.get_window_extent()
        cbar_width_fig = (width * self.MM2INCH * self.fig.dpi) / fig_extent.width
        cbar_height_fig = (height * self.MM2INCH * self.fig.dpi) / fig_extent.height

        # ``add_axes`` takes the bottom-left corner, while
        # ``_mm_to_figure_coords`` returns the strip's *outward* edge —
        # its bottom for side='top', its top otherwise; its right edge for
        # side='left', its left otherwise. Mirrors the per-draw arithmetic
        # in ``LayoutReactor._update_artist_anchor``, which owns the
        # position from the first draw onward.
        cbar_bottom_fig = y_fig if self._side == "top" else y_fig - cbar_height_fig
        cbar_left_fig = x_fig - cbar_width_fig if self._side == "left" else x_fig

        cbar_ax = self.fig.add_axes(
            [cbar_left_fig, cbar_bottom_fig, cbar_width_fig, cbar_height_fig],
            xmargin=0,
            ymargin=0
        )

        # Create colorbar
        cbar = self.fig.colorbar(
            mappable,
            cax=cbar_ax,
            orientation=orientation,
            **kwargs
        )

        # Set label (only if not on top)
        if title_position != "top":
            cbar.set_label(label)
        else:
            cbar.set_label("")

        # Set ticks
        if ticks is not None:
            cbar.set_ticks(ticks)
        elif center is not None and vmin is not None and vmax is not None:
            # Auto-set ticks for divergent colormap
            cbar.set_ticks([vmin, center, vmax])

        # The strip's mm footprint is exactly what ``add_axes`` was given:
        # ``fig.colorbar`` never resizes a caller-supplied ``cax``. Reading
        # it back in pixels would report a *stale* size whenever the draw
        # below grew the figure — pp.subplots resizes to fit the band, and
        # the axes' stored fraction then covers more mm than it was built
        # with. That is how a top band's 15mm strip came back as 20.06mm.
        # The draw itself is kept: callers downstream expect the figure to
        # be current after add_colorbar.
        self._fig_canvas_draw_for_measure()
        cbar_width, cbar_height = width, height

        # A horizontal strip carries its tick labels BELOW itself. On a
        # bottom band "below" is further outward, past everything else in
        # the block; on a top band it points back TOWARD the axes, so the
        # block's tight bbox dips below the outward line the strip's
        # rectangle was placed on and the tick numbers land inside the
        # axes rectangle (1.61mm into a 40mm axes, measured). Step the
        # whole block — strip and label together — outward by that
        # overhang, so what clears the axes edge is the block's tight
        # bbox and not just the bare colour rectangle. (#213)
        #
        # Measured rather than derived from ``orientation``: a caller who
        # asks for a vertical strip on a top band, or who passes
        # ``ticklocation='top'``, has no inward overhang and the whole
        # correction collapses to zero. Only side='top' needs it — every
        # other side already carries its tick labels either outward
        # (right, bottom) or along the edge, and a left band's inward
        # overhang is absorbed by the y-decoration offset the band is
        # already pushed past.
        inward_overhang_mm = 0.0
        if self._side == "top":
            tight = cbar_ax.get_tightbbox()
            if tight is not None:
                strip_extent = cbar_ax.get_window_extent()
                inward_overhang_mm = max(
                    0.0,
                    (strip_extent.y0 - tight.y0) / self.fig.dpi / self.MM2INCH,
                )
        if inward_overhang_mm:
            cbar_outward_mm += inward_overhang_mm
            # Re-derive the strip's rectangle at the shifted offset. Only
            # side='top' reaches here, and there ``_mm_to_figure_coords``
            # returns the strip's bottom-left corner directly — which is
            # what ``set_position`` wants. The figure extent is re-read
            # because the draw above may have grown the figure to fit
            # the band.
            fig_extent = self.fig.get_window_extent()
            x_fig, y_fig = self._mm_to_figure_coords(cbar_outward_mm, cbar_y_start)
            cbar_ax.set_position([
                x_fig,
                y_fig,
                (width * self.MM2INCH * self.fig.dpi) / fig_extent.width,
                (height * self.MM2INCH * self.fig.dpi) / fig_extent.height,
            ])
            if title_obj is not None:
                title_outward_mm += inward_overhang_mm
                x_fig, y_fig = self._mm_to_figure_coords(
                    title_outward_mm,
                    self._layout.current_along - title_along_shift,
                )
                title_obj.set_position((x_fig, y_fig))

        # Deferred from the title block above so it picks up the shifted
        # outward offset. The reactor owns the label's position from the
        # first draw onward, so this registration — not the set_position
        # calls — is what keeps the block clear of the axes.
        if title_obj is not None:
            self._reactor.register(
                ax=self._anchor_ax,
                artist=title_obj,
                mm_x_from_right=title_outward_mm,
                mm_y_from_top=title_mm_y_from_top,
                side=self._side,
                external_to_axis=self._external_to_axis,
            )

        # Calculate actual total width (max of title and colorbar)
        actual_width = max(cbar_width, title_width_actual)

        # Calculate actual total height
        if title_position == "top" and label:
            total_height_actual = title_height_actual + title_pad + cbar_height
        else:
            total_height_actual = cbar_height

        # The layout cursor still points at the block's origin, so the
        # strip's own placement is re-derived here from the same offsets
        # used above: outward past the label on a bottom band, along past
        # it on right/left, unchanged on a top band.
        placement_x_mm = cbar_outward_mm
        if title_height_actual and not stack_outward:
            mm_y_from_top = (
                self._layout.along_from_start + title_height_actual + title_pad
            )
        else:
            mm_y_from_top = self._layout.along_from_start + cbar_along_shift

        # Update layout cursor past the full title+colorbar block. The
        # stack's extent lands on whichever axis carries it: outward for
        # top/bottom bands, along the edge for right/left.
        if stack_outward:
            # The inward tick overhang sits between the outward line and
            # the strip, so it is part of the block's outward extent.
            self._layout.update_width(total_height_actual + inward_overhang_mm)
            self._layout.advance_along(actual_width)
        else:
            self._layout.update_width(actual_width)
            self._layout.advance_along(total_height_actual)

        # Register with the reactor. Colorbars need mm_width + mm_height so
        # the reactor dispatches to cbar.ax.set_position instead of
        # set_bbox_to_anchor (Colorbar doesn't implement set_bbox_to_anchor).
        self._reactor.register(
            ax=self._anchor_ax,
            artist=cbar,
            mm_x_from_right=placement_x_mm,
            mm_y_from_top=mm_y_from_top,
            mm_width=width,        # the mm width parameter of add_colorbar
            mm_height=cbar_height, # == the ``height`` argument; see above
            side=self._side,
            external_to_axis=self._external_to_axis,
        )

        # Store elements
        self.elements.append(("colorbar", cbar))
        self._colorbar_names[id(cbar)] = label
        if title_obj:
            self.elements.append(("text", title_obj))
            # Remember which strip this floating label belongs to. The two
            # carry independent reactor registrations, so without this the
            # along-edge alignment pass has no way to tell one band's label
            # from another's (#214).
            self._colorbar_labels[id(cbar)] = title_obj
        if inward_overhang_mm:
            self._colorbar_inward_pad[id(cbar)] = inward_overhang_mm

        return cbar

    def add_legend_for(self, type: str, label: Optional[str] = None, **kwargs):
        """
        Add legend by auto-detecting from self.ax stored metadata.

        Parameters
        ----------
        type : str
            Type of legend: 'hue', 'size', or 'style'
        label : str, optional
            Legend label (overrides default from metadata).
        **kwargs : dict
            Additional customization passed to add_legend() or add_colorbar()
            (frameon, labelspacing, handletextpad, height, width, etc.)

        Examples
        --------
        >>> builder = pp.legend(ax, auto=False)
        >>> builder.add_legend_for('hue', label='Groups')
        >>> builder.add_legend_for('size', label='Magnitude')
        >>> builder.add_legend_for('hue', label='Score')  # Works for colorbar too
        """
        legend_data = _get_legend_data(self.ax)

        if legend_data and type in legend_data:
            # Use stored metadata
            data = legend_data[type].copy()

            # Check if this is a colorbar
            if data.get('type') == 'colorbar':
                # Handle colorbar
                if label is not None:
                    data['label'] = label
                data.update(kwargs)
                # Remove 'type' key as it's not a parameter for add_colorbar
                data.pop('type', None)
                self.add_colorbar(**data)
            else:
                # Handle regular legend
                if label is not None:
                    data['label'] = label
                data.update(kwargs)
                self.add_legend(**data)
        else:
            # Fallback: basic auto-detection
            # This is a simple fallback - may not work for complex cases
            pass

    def get_remaining_height(self) -> float:
        """Get remaining vertical space."""
        return max(0, self._layout.current_along)


__all__ = [
    "HandlerRectangle",
    "HandlerMarker",
    "HandlerLineMarker",
    "HandlerLine",
    "RectanglePatch",
    "MarkerPatch",
    "LineMarkerPatch",
    "LinePatch",
    "get_legend_handler_map",
    "create_legend_handles",
    "LegendBuilder",
]