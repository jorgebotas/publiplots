"""
Legend handlers for publiplots.

This module provides custom legend handlers for creating publication-ready legends
that match the double-layer plotting style used in publiplots (transparent fill +
solid edge). The handlers automatically create legend markers that match the
visual style of scatterplots and barplots.
"""

from typing import List, Dict, Optional, Tuple, Any, Union

from publiplots.themes.rcparams import resolve_param
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
        markeredgewidth = kwargs.pop("markeredgewidth", resolve_param("lines.markeredgewidth"))
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
            markeredgewidth = resolve_param("lines.markeredgewidth")
        self.markeredgewidth = markeredgewidth


class LineMarkerPatch(Patch):
    """
    Custom patch for line+marker legend handles (pointplot, lineplot, etc.).
    Embeds marker symbol, markersize, linestyle, and all styling properties.
    """
    def __init__(self, marker='o', linestyle=None, **kwargs):
        markersize = kwargs.pop("markersize", resolve_param("lines.markersize"))
        markeredgewidth = kwargs.pop("markeredgewidth", resolve_param("lines.markeredgewidth"))
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
            markeredgewidth = resolve_param("lines.markeredgewidth")
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
        color = "gray"
        alpha = resolve_param("alpha", None)
        linewidth = resolve_param("lines.linewidth", None)
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
        color = "gray"
        size = resolve_param("lines.markersize")
        alpha = resolve_param("alpha")
        linewidth = resolve_param("lines.linewidth")
        markeredgewidth = resolve_param("lines.markeredgewidth")
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
            markeredgewidth = orig_handle.get_mairkeredgewidth() or linewidth
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
            markeredgecolor=to_rgba(color, 1.0),
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
        color = "gray"
        size = resolve_param("lines.markersize")
        alpha = resolve_param("alpha")
        linewidth = resolve_param("lines.linewidth")
        markeredgewidth = resolve_param("lines.markeredgewidth")
        edgecolor = None
        linestyle = None

        # Extract from LineMarkerPatch (created by create_legend_handles)
        if isinstance(orig_handle, LineMarkerPatch):
            marker = orig_handle.get_marker()
            color = orig_handle.get_facecolor()
            edgecolor = orig_handle.get_edgecolor()
            alpha = orig_handle.get_alpha() if orig_handle.get_alpha() is not None else alpha
            linestyle = orig_handle.get_linestyle()
            linewidth = orig_handle.get_linewidth()
            markeredgewidth = orig_handle.get_markeredgewidth()
            # Use actual markersize from patch (already in correct units)
            patch_size = orig_handle.get_markersize()
            if patch_size is not None:
                size = patch_size

        # Extract from Line2D (standard matplotlib - fallback)
        elif isinstance(orig_handle, Line2D):
            marker = orig_handle.get_marker() or marker
            linestyle = orig_handle.get_linestyle()
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

    return {
        Rectangle: handler_rectangle,
        MarkerPatch: handler_marker,
        LineMarkerPatch: handler_line_marker,
        Patch: handler_rectangle,
    }

def create_legend_handles(
    labels: List[str],
    colors: Optional[List[str]] = None,
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
    alpha : float, default=DEFAULT_ALPHA
        Transparency level for fill layers.
    linewidth : float, default=DEFAULT_LINEWIDTH
        Width of edge lines.
    markeredgewidth : float, default=DEFAULT_MARKEREDEDGWIDTH
        Width of marker edges.
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
    # Read defaults from rcParams if not provided
    alpha = resolve_param("alpha", alpha)
    linewidth = resolve_param("lines.linewidth", linewidth)
    markeredgewidth = resolve_param("lines.markeredgewidth", markeredgewidth)

    if colors is None:
        default_color = resolve_param("color", None)
        colors = [color if color is not None else default_color] * len(labels)

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
        for label, col, size, marker, linestyle in zip(labels, colors, sizes, markers, linestyles):
            handle = LineMarkerPatch(
                marker=marker,
                linestyle=linestyle,
                facecolor=col,
                edgecolor=col,
                alpha=alpha,
                linewidth=linewidth,
                label=label,
                markersize=size,
                markeredgewidth=markeredgewidth,
            )
            handles.append(handle)
    elif markers is not None:
        # Use MarkerPatch when only markers are specified
        for label, col, hatch, size, marker in zip(labels, colors, hatches, sizes, markers):
            handle = MarkerPatch(
                marker=marker,
                facecolor=col,
                edgecolor=col,
                alpha=alpha,
                linewidth=linewidth,
                label=label,
                markersize=size,
                markeredgewidth=markeredgewidth,
            )
            handles.append(handle)
    else:
        # Use MarkerPatch for circles, RectanglePatch for rectangles
        if style == "circle":
            # Circle is just a marker with 'o' symbol
            for label, col, hatch, size in zip(labels, colors, hatches, sizes):
                handle = MarkerPatch(
                    marker='o',
                    facecolor=col,
                    edgecolor=col,
                    alpha=alpha,
                    linewidth=linewidth,
                    label=label,
                    markersize=size,
                    markeredgewidth=markeredgewidth,
                )
                handles.append(handle)
        else:
            # Rectangle patches (for bar plots with hatches)
            for label, col, hatch, size in zip(labels, colors, hatches, sizes):
                handle = RectanglePatch(
                    facecolor=col,
                    edgecolor=col,
                    alpha=alpha,
                    linewidth=linewidth,
                    label=label,
                    hatch=hatch,
                    markersize=size,
                )
                handles.append(handle)

    return handles


# =============================================================================
# Legend Builder (Primary Interface)
# =============================================================================


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
    >>> fig, ax = pp.scatterplot(df, x='x', y='y', hue='group', legend=False)
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
        vpad: float = 5,
        max_width: Optional[float] = None,
        mode: str = 'external',
    ):
        """Initialize legend builder. All dimensions in millimeters."""
        self.ax = ax
        self.fig = ax.get_figure()

        # Store parameters (all in mm)
        self.x_offset = x_offset
        self.gap = gap
        self.column_spacing = column_spacing
        self.vpad = vpad
        self.max_width = max_width
        self.mode = mode  # 'external' (figure coords) or 'internal' (axes coords)

        # Initialize position tracking (all in mm)
        self.current_x = x_offset
        self.current_y = y_offset if y_offset is not None else self._get_axes_height()
        self.column_start_y = self.current_y

        # Column tracking
        self.current_column_width = 0
        self.columns = []  # List of column widths

        # Element storage
        self.elements = []  # (type, object) tuples

    # =========================================================================
    # Conversion Utilities
    # =========================================================================

    def _get_axes_height(self) -> float:
        """Get axes height in millimeters."""
        ax_pos = self.ax.get_position()
        fig_height_px = self.fig.get_window_extent().height
        axes_height_px = ax_pos.height * fig_height_px
        return axes_height_px / self.fig.dpi / self.MM2INCH

    def _mm_to_figure_coords(self, x_mm: float, y_mm: float) -> Tuple[float, float]:
        """
        Convert mm position to figure coordinates (external mode).

        Parameters
        ----------
        x_mm : float
            Horizontal distance from right edge of axes (mm)
        y_mm : float
            Remaining vertical space (mm). Converted to position from top internally.

        Returns
        -------
        x_fig, y_fig : float
            Position in figure coordinates
        """
        ax_pos = self.ax.get_position()
        fig_extent = self.fig.get_window_extent()

        # y_mm represents remaining space, convert to position from top
        axes_height = self._get_axes_height()
        position_from_top = axes_height - y_mm

        # Convert mm to figure fraction
        x_offset_fig = (x_mm * self.MM2INCH * self.fig.dpi) / fig_extent.width
        y_offset_fig = (position_from_top * self.MM2INCH * self.fig.dpi) / fig_extent.height

        # Position relative to axes
        x_fig = ax_pos.x1 + x_offset_fig
        y_fig = ax_pos.y1 - y_offset_fig

        return x_fig, y_fig

    def _mm_to_axes_coords(self, x_mm: float, y_mm: float) -> Tuple[float, float]:
        """
        Convert mm position to axes coordinates (internal mode).

        Parameters
        ----------
        x_mm : float
            Horizontal distance from left edge of axes (mm)
        y_mm : float
            Remaining vertical space (mm). Converted to position from top internally.

        Returns
        -------
        x_axes, y_axes : float
            Position in axes coordinates (0-1 range)
        """
        # Get axes dimensions in mm
        ax_pos = self.ax.get_position()
        fig_extent = self.fig.get_window_extent()

        axes_width_px = ax_pos.width * fig_extent.width
        axes_height_px = ax_pos.height * fig_extent.height

        axes_width_mm = axes_width_px / self.fig.dpi / self.MM2INCH
        axes_height_mm = axes_height_px / self.fig.dpi / self.MM2INCH

        # Convert mm to axes-relative coordinates (0-1)
        x_axes = x_mm / axes_width_mm

        # y_mm represents remaining space from top, convert to position from bottom (matplotlib axes coords)
        position_from_top_mm = axes_height_mm - y_mm
        y_axes = position_from_top_mm / axes_height_mm

        return x_axes, y_axes

    def _mm_to_coords(self, x_mm: float, y_mm: float) -> Tuple[float, float, Any]:
        """
        Convert mm position to appropriate coordinates based on mode.

        Returns
        -------
        x, y, transform : Tuple[float, float, Transform]
            Position coordinates and the appropriate transform
        """
        if self.mode == 'internal':
            x, y = self._mm_to_axes_coords(x_mm, y_mm)
            transform = self.ax.transAxes
        else:  # 'external'
            x, y = self._mm_to_figure_coords(x_mm, y_mm)
            transform = self.fig.transFigure

        return x, y, transform

    def _axes_to_figure_coords(self, x_axes: float, y_axes: float) -> Tuple[float, float]:
        """
        Convert axes coordinates (0-1) to figure coordinates.

        This is needed for creating colorbar axes and text, which require figure coordinates.
        """
        ax_pos = self.ax.get_position()

        x_fig = ax_pos.x0 + x_axes * ax_pos.width
        y_fig = ax_pos.y0 + y_axes * ax_pos.height

        return x_fig, y_fig

    def _measure_object_dimensions(self, obj: Union[Legend, Colorbar, Any]) -> Tuple[float, float]:
        """
        Measure actual dimensions of matplotlib object.

        Parameters
        ----------
        obj : Legend or Colorbar or Text
            Matplotlib object to measure

        Returns
        -------
        width_mm, height_mm : float
            Object dimensions in millimeters
        """
        self.fig.canvas.draw()

        # Get bounding box
        if hasattr(obj, 'ax'):  # Colorbar
            bbox = obj.ax.get_window_extent(self.fig.canvas.get_renderer())
        elif hasattr(obj, 'get_window_extent'):
            bbox = obj.get_window_extent(self.fig.canvas.get_renderer())
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

        # Get legend parameters
        ncol = kwargs.get('ncol', 1)
        labelspacing = kwargs.get('labelspacing', 0.3)  # in font-size units
        borderpad = kwargs.get('borderpad', 0.4)

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
        handlelength = kwargs.get('handlelength', 2)  # in font-size units
        handletextpad = kwargs.get('handletextpad', 0.8)
        handle_width = (handlelength + handletextpad) * fontsize * self.PT2MM

        # Add padding
        borderpad = kwargs.get('borderpad', 0.4)
        padding = 2 * borderpad * fontsize * self.PT2MM

        return handle_width + text_width + padding

    # =========================================================================
    # Column Management
    # =========================================================================

    def _check_overflow(self, required_height: float) -> bool:
        """
        Check if element fits in current column.

        Parameters
        ----------
        required_height : float
            Height needed in millimeters

        Returns
        -------
        bool
            True if overflow (doesn't fit), False if fits
        """
        return self.current_y < required_height

    def _start_new_column(self):
        """Create a new column when vertical space exhausted."""
        # Record current column width
        self.columns.append(self.current_column_width)

        # Move horizontally
        self.current_x += self.current_column_width + self.column_spacing

        # Reset vertical position
        self.current_y = self.column_start_y

        # Reset width tracking
        self.current_column_width = 0

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
            Additional kwargs for legend customization:
            - ncol: number of columns (auto-adjusted if max_height exceeded)
            - labelspacing: vertical space between entries
            - handletextpad: space between handle and text
            - columnspacing: space between columns (in font-size units)

        Returns
        -------
        Legend
            The created legend object.

        Notes
        -----
        All dimensions in millimeters. Columns created automatically on overflow.
        """
        # Auto-adjust ncol if max_height specified
        if max_height is not None:
            optimal_ncol = self._adjust_legend_ncol_for_height(
                handles, label, max_height, **kwargs
            )
            kwargs['ncol'] = optimal_ncol

        # Estimate height
        estimated_height = self._estimate_legend_height(handles, label, **kwargs)

        # Check overflow
        if self._check_overflow(estimated_height):
            self._start_new_column()

        # Convert current position to appropriate coordinates based on mode
        x, y, transform = self._mm_to_coords(self.current_x, self.current_y)

        # Prepare legend kwargs
        legend_kwargs = {
            "loc": "upper left",
            "bbox_to_anchor": (x, y),
            "bbox_transform": transform,
            "frameon": frameon,
            "borderaxespad": 0,
            "borderpad": 0.4,
            "handletextpad": 0.8,
            "labelspacing": 0.3,
            "handler_map": kwargs.pop("handler_map", get_legend_handler_map()),
            "alignment": "left",
        }

        if label:
            legend_kwargs['title'] = label

        legend_kwargs.update(kwargs)

        # Create legend
        existing_legends = [e[1] for e in self.elements if e[0] == "legend"]
        legend = self.ax.legend(handles=handles, **legend_kwargs)
        legend.set_clip_on(False)

        # Re-add existing legends (matplotlib limitation)
        for existing_legend in existing_legends:
            self.ax.add_artist(existing_legend)

        # Measure actual dimensions
        width, height = self._measure_object_dimensions(legend)

        # Update position tracking
        self.current_column_width = max(self.current_column_width, width)
        self.current_y -= height + self.gap

        # Store element
        self.elements.append(("legend", legend))

        return legend
    
    def add_colorbar(
        self,
        mappable: Optional[ScalarMappable] = None,
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        center: Optional[float] = None,
        label: str = "",
        height: float = 15,
        width: float = 4.5,
        title_position: str = "top",
        orientation: str = "vertical",
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
        height : float, default=15
            Colorbar height in millimeters.
        width : float, default=4.5
            Colorbar width in millimeters.
        title_position : {'top', 'right'}, default='top'
            Position of title. 'top' places label above colorbar
            (horizontal), 'right' uses matplotlib default (vertical).
        orientation : {'vertical', 'horizontal'}, default='vertical'
            Colorbar orientation.
        ticks : list of float, optional
            Custom tick positions. If None and center is provided,
            automatically sets ticks at [vmin, center, vmax].
        **kwargs
            Additional kwargs passed to fig.colorbar().

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
        from matplotlib.colors import TwoSlopeNorm, Normalize
        from matplotlib.cm import ScalarMappable as SM, get_cmap

        # Create mappable if cmap provided (PyComplexHeatmap style)
        if mappable is None and cmap is not None:
            cmap_obj = get_cmap(cmap)
            if center is not None:
                norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
            else:
                norm = Normalize(vmin=vmin, vmax=vmax)
            mappable = SM(norm=norm, cmap=cmap_obj)

        # Estimate title height for overflow check
        title_pad = 2  # mm
        title_obj = None
        if title_position == "top" and label:
            fontsize = resolve_param("legend.title_fontsize", resolve_param("font.size"))
            estimated_title_height = fontsize * self.PT2MM * 1.3
            total_estimated_height = height + estimated_title_height + title_pad
        else:
            total_estimated_height = height

        # Check overflow using estimate
        if self._check_overflow(total_estimated_height):
            self._start_new_column()

        # Add title if needed and measure actual height
        title_height_actual = 0
        if title_position == "top" and label:
            # Get position (mode-aware, but always convert to figure coords for text)
            if self.mode == 'internal':
                x_axes, y_axes = self._mm_to_axes_coords(self.current_x, self.current_y)
                x_fig, y_fig = self._axes_to_figure_coords(x_axes, y_axes)
            else:
                x_fig, y_fig = self._mm_to_figure_coords(self.current_x, self.current_y)

            title_obj = self.fig.text(
                x_fig, y_fig, label,
                ha="left", va="top",
                fontsize=resolve_param("legend.title_fontsize", resolve_param("font.size")),
                fontweight="normal"
            )

            # Measure actual title dimensions
            title_width_actual, title_height_actual = self._measure_object_dimensions(title_obj)

            # Position colorbar below measured title
            cbar_y_start = self.current_y - title_height_actual - title_pad
        else:
            cbar_y_start = self.current_y
            title_width_actual = 0

        # Create colorbar axes (mode-aware, but always convert to figure coords for axes creation)
        if self.mode == 'internal':
            x_axes, y_axes_top = self._mm_to_axes_coords(self.current_x, cbar_y_start)
            x_fig, y_fig_top = self._axes_to_figure_coords(x_axes, y_axes_top)
        else:
            x_fig, y_fig_top = self._mm_to_figure_coords(self.current_x, cbar_y_start)

        fig_extent = self.fig.get_window_extent()
        cbar_width_fig = (width * self.MM2INCH * self.fig.dpi) / fig_extent.width
        cbar_height_fig = (height * self.MM2INCH * self.fig.dpi) / fig_extent.height

        cbar_ax = self.fig.add_axes(
            [x_fig, y_fig_top - cbar_height_fig, cbar_width_fig, cbar_height_fig],
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

        # Measure actual colorbar dimensions
        cbar_width, cbar_height = self._measure_object_dimensions(cbar)

        # Calculate actual total width (max of title and colorbar)
        actual_width = max(cbar_width, title_width_actual)

        # Calculate actual total height
        if title_position == "top" and label:
            total_height_actual = title_height_actual + title_pad + cbar_height
        else:
            total_height_actual = cbar_height

        # Update position tracking
        self.current_column_width = max(self.current_column_width, actual_width)
        self.current_y -= total_height_actual + self.gap

        # Store elements
        self.elements.append(("colorbar", cbar))
        if title_obj:
            self.elements.append(("text", title_obj))

        return cbar

    def add_existing_legend(self, legend: Legend, **kwargs) -> Legend:
        """
        Add an already-created legend by repositioning it (no re-rendering).

        This method takes an existing Legend object and repositions it within
        the legend builder's layout. It's more efficient than recreating legends
        as it preserves all original styling, handlers, and formatting.

        Parameters
        ----------
        legend : Legend
            Existing matplotlib Legend object to reposition.
        **kwargs
            Additional customization options (currently unused, for future extension).

        Returns
        -------
        Legend
            The repositioned legend object.

        Notes
        -----
        This method calculates dimensions from the existing legend object,
        checks for column overflow, and updates its position without recreating
        it. All legend properties (handles, labels, styling) are preserved.
        """
        # Measure existing legend dimensions (before removal)
        width, height = self._measure_object_dimensions(legend)

        # Remove legend from its original axes
        # This is necessary because matplotlib doesn't allow artists to belong to multiple axes
        legend.remove()

        # Check overflow
        if self._check_overflow(height):
            self._start_new_column()

        # Calculate new position in appropriate coordinates based on mode
        x, y, transform = self._mm_to_coords(self.current_x, self.current_y)

        # Update legend position
        legend.set_bbox_to_anchor((x, y), transform=transform)

        # Re-add existing legends (matplotlib limitation)
        existing_legends = [e[1] for e in self.elements if e[0] == "legend"]
        for existing_legend in existing_legends:
            self.ax.add_artist(existing_legend)

        # Add to target axes
        self.ax.add_artist(legend)
        legend.set_clip_on(False)

        # Update position tracking
        self.current_column_width = max(self.current_column_width, width)
        self.current_y -= height + self.gap

        # Store element
        self.elements.append(("legend", legend))

        return legend

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
        return max(0, self.current_y)


def _get_legend_data(ax: Axes) -> dict:
    """
    Get stored legend data from axes collections/patches/lines.

    Parameters
    ----------
    ax : Axes
        Axes to retrieve legend data from

    Returns
    -------
    dict
        Dictionary with legend data for 'hue', 'size', 'style' if available
    """
    # Check collections first
    for collection in ax.collections:
        if hasattr(collection, '_legend_data'):
            return collection._legend_data

    # Check patches
    for patch in ax.patches:
        if hasattr(patch, '_legend_data'):
            return patch._legend_data

    # Check lines (for pointplot)
    for line in ax.lines:
        if hasattr(line, '_legend_data'):
            return line._legend_data

    return {}


def legend(
    ax: Optional[Axes] = None,
    handles: Optional[List] = None,
    labels: Optional[List[str]] = None,
    auto: bool = True,
    x_offset: float = 2,
    gap: float = 2,
    column_spacing: float = 5,
    vpad: float = 5,
    from_axes: Optional[List[Axes]] = None,
    **kwargs
) -> LegendBuilder:
    """
    Create publication-ready legends with automatic positioning.

    **All dimensions in millimeters** for precise control in publication plots.
    Returns LegendBuilder for adding multiple legends with automatic column
    overflow handling.

    This is the primary interface for legend creation in publiplots.

    Parameters
    ----------
    ax : Axes, optional
        Axes to create legend for. If None, uses plt.gca() (current axes).
    handles : list, optional
        Manual legend handles. If provided, auto is ignored.
    labels : list, optional
        Manual legend labels (used with handles).
    auto : bool, default=True
        If True, auto-creates all legends from metadata stored on plot objects.
        If False, returns empty builder for manual control.
    x_offset : float, default=2
        Horizontal distance from right edge of axes (millimeters).
    gap : float, default=2
        Vertical spacing between legend elements (millimeters).
    column_spacing : float, default=5
        Horizontal spacing between columns (millimeters).
    vpad : float, default=5
        Top padding from axes edge (millimeters).
    from_axes : list of Axes, optional
        List of axes to collect and reconcile legends from.
        Used by complex_heatmap to create unified legends from multiple axes.
    **kwargs
        Additional kwargs passed to add_legend() if handles provided,
        or legend customization options.

    Returns
    -------
    LegendBuilder
        Builder object for adding more legends.

    Examples
    --------
    Auto mode (no axes needed, uses current axes):
    >>> fig, ax = pp.scatterplot(df, x='x', y='y', hue='group', legend=False)
    >>> builder = pp.legend()  # Auto-creates legends on current axes

    Auto mode with explicit axes:
    >>> builder = pp.legend(ax)  # Auto-creates legends

    Manual sequential mode:
    >>> builder = pp.legend(auto=False, x_offset=3, gap=3)
    >>> builder.add_legend(hue_handles, label="Treatment")
    >>> builder.add_colorbar(cmap='RdBu_r', vmin=-2, vmax=2, center=0,
    ...                      label="Log2 FC", height=20)
    >>> builder.add_legend(marker_handles, label="Cell Type")

    Manual handles mode:
    >>> builder = pp.legend(handles=custom_handles, labels=custom_labels,
    ...                     label='My Legend')

    Notes
    -----
    All dimensions are in millimeters. New columns are created automatically
    when vertical space is exhausted. This ensures legends never overlap with
    the plot area and maintains consistent spacing for publication-quality figures.

    Multiple calls to legend() on the same axes will reuse the same LegendBuilder,
    allowing legends to stack properly without overriding each other.
    """
    # Get current axes if not provided (like plt.legend())
    if ax is None:
        ax = plt.gca()

    # Check if a builder already exists for this axes
    if hasattr(ax, '_legend_builder') and ax._legend_builder is not None:
        builder = ax._legend_builder
    else:
        # Detect mode: 'internal' if collecting from other axes, 'external' otherwise
        mode = 'internal' if from_axes is not None else 'external'

        # Initialize new builder
        builder = LegendBuilder(
            ax,
            x_offset=x_offset,
            gap=gap,
            column_spacing=column_spacing,
            vpad=vpad,
            mode=mode,
        )
        # Store on axes for reuse
        ax._legend_builder = builder

    # Auto-apply handler_map
    if 'handler_map' not in kwargs:
        kwargs['handler_map'] = get_legend_handler_map()

    # Reconcile mode - collect legends from multiple axes
    if from_axes is not None:
        for source_ax in from_axes:
            if not hasattr(source_ax, '_legend_builder'):
                continue

            source_builder = source_ax._legend_builder
            if source_builder is None:
                continue

            # Iterate through stored elements in the source builder
            for element_type, element in source_builder.elements:
                if element_type == "legend":
                    # Reposition existing legend (no re-rendering needed)
                    builder.add_existing_legend(element)
                elif element_type == "colorbar":
                    # Add colorbar to unified legend
                    # Extract colorbar properties from the stored colorbar object
                    cbar = element

                    # Get the mappable (ScalarMappable) from the colorbar
                    mappable = cbar.mappable

                    # Get the label (could be from ax.yaxis or ax.xaxis depending on orientation)
                    cbar_ax = cbar.ax
                    if cbar.orientation == 'vertical':
                        label = cbar_ax.get_ylabel()
                    else:
                        label = cbar_ax.get_xlabel()

                    # Get ticks
                    ticks = cbar.get_ticks()

                    # Extract actual dimensions from colorbar axes
                    # Measure the colorbar axes bounding box and convert to mm
                    self.fig.canvas.draw()
                    cbar_bbox = cbar_ax.get_window_extent(self.fig.canvas.get_renderer())
                    cbar_width_mm = cbar_bbox.width / self.fig.dpi / builder.MM2INCH
                    cbar_height_mm = cbar_bbox.height / self.fig.dpi / builder.MM2INCH

                    # Determine dimensions based on orientation
                    if cbar.orientation == 'vertical':
                        height = cbar_height_mm
                        width = cbar_width_mm
                    else:  # horizontal
                        height = cbar_height_mm
                        width = cbar_width_mm

                    # Add colorbar to unified builder with preserved dimensions
                    builder.add_colorbar(
                        mappable=mappable,
                        label=label,
                        height=height,
                        width=width,
                        orientation=cbar.orientation,
                        ticks=list(ticks) if ticks is not None else None,
                    )

        # Hide axes frame and ticks for clean legend panel
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        return builder

    # Manual mode with handles
    if handles is not None:
        builder.add_legend(handles=handles, labels=labels, **kwargs)
        return builder

    # Auto mode
    if auto:
        legend_data = _get_legend_data(ax)
        if legend_data:
            # Add hue legend/colorbar
            if 'hue' in legend_data:
                hue_data = legend_data['hue'].copy()
                if hue_data.get('type') == 'colorbar':
                    hue_data.pop('type', None)
                    builder.add_colorbar(**hue_data)
                else:
                    builder.add_legend(**hue_data, **kwargs)

            # Add size legend
            if 'size' in legend_data:
                size_data = legend_data['size'].copy()
                builder.add_legend(**size_data, **kwargs)

            # Add style legend
            if 'style' in legend_data:
                style_data = legend_data['style'].copy()
                builder.add_legend(**style_data, **kwargs)

    return builder

__all__ = [
    "HandlerRectangle",
    "HandlerMarker",
    "HandlerLineMarker",
    "RectanglePatch",
    "MarkerPatch",
    "LineMarkerPatch",
    "get_legend_handler_map",
    "create_legend_handles",
    "LegendBuilder",
    "legend",
]