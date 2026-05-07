"""Shared helpers for plot functions that stash + render LegendEntry objects.

Two primitives:
- ``stash_hue_legend``: one-shot hue stash used by plots with a single
  categorical hue dimension (box, violin, heatmap categorical path).
- ``render_entries``: per-axis render of any already-stashed entries that
  aren't claimed by a figure-level ``pp.legend_group``. Used by plots with
  multi-kind stashing (scatter-style: hue + size + style, or bar: hue + hatch).

Plots that do their own kind-specific stashing should call ``render_entries``
directly after stashing.
"""

from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
import numpy as np

from publiplots.utils import create_legend_handles
from publiplots.utils.legend_entries import (
    LegendEntry,
    stash_entry,
    get_entries,
    resolve_legend_flags,
    entry_is_in_group,
    is_continuous_hue,
)

if TYPE_CHECKING:
    import matplotlib.axes
    import pandas as pd
    from matplotlib.colors import Normalize


def stash_hue_legend(
    ax: Axes,
    *,
    hue: Optional[str],
    palette: Optional[Union[str, Dict, List]],
    edgecolor: Optional[str],
    alpha: Optional[float],
    linewidth: Optional[float],
    legend: Union[bool, Dict],
    legend_kws: Optional[Dict],
) -> None:
    """Stash one hue LegendEntry and render it per-axis if not in a group.

    Short-circuits (no stash, no render) when any of these hold:
      - ``legend is False``
      - ``hue is None``
      - ``palette`` is not a dict (caller should resolve it first)
    """
    if legend is False or hue is None or not isinstance(palette, dict):
        return

    flags = resolve_legend_flags(legend)
    legend_kws = dict(legend_kws or {})
    hue_label = legend_kws.pop("hue_label", hue)

    if flags["hue"]:
        labels = list(palette.keys())
        handles = create_legend_handles(
            labels=labels,
            colors=list(palette.values()),
            edgecolors=[edgecolor] * len(palette) if edgecolor else None,
            alpha=alpha,
            linewidth=linewidth,
        )
        stash_entry(
            ax,
            LegendEntry.build(
                name=hue_label,
                kind="hue",
                handles=handles,
                labels=labels,
            ),
        )

    render_entries(ax, flags=flags, legend_kws=legend_kws)


def stash_continuous_hue(
    ax: "matplotlib.axes.Axes",
    *,
    name: str,
    palette,
    hue_norm,
) -> None:
    """Stash one continuous-hue LegendEntry (ScalarMappable) on the axes.

    The stashed handle is a matplotlib ScalarMappable(norm=hue_norm, cmap=palette)
    wrapped in a list with empty labels — the sentinel shape that
    ``is_continuous_hue`` detects and ``render_entries`` routes through
    ``add_colorbar`` instead of ``add_legend``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to stash the entry on.
    name : str
        The legend title (typically the hue column name).
    palette : str | Colormap
        Colormap for the mappable. Must be accepted by ScalarMappable.
    hue_norm : matplotlib.colors.Normalize
        Normalization for the mappable.
    """
    from matplotlib.cm import ScalarMappable
    mappable = ScalarMappable(norm=hue_norm, cmap=palette)
    stash_entry(
        ax,
        LegendEntry.build(
            name=name,
            kind="hue",
            handles=[mappable],
            labels=[],
        ),
    )


def merge_categorical_entries(
    *,
    name: str,
    labels: List[str],
    colors: Optional[List[str]] = None,
    markers: Optional[List[str]] = None,
    linestyles: Optional[List[str]] = None,
    sizes: Optional[List[float]] = None,
    edgecolors: Optional[Union[str, List[str]]] = None,
    handle_kwargs: Optional[Dict] = None,
) -> LegendEntry:
    """Build one LegendEntry whose swatches encode multiple categorical kinds.

    Used when hue / style (and optionally categorical size) share the same
    column — the intuitive rendering is one legend column whose swatch
    encodes every dimension simultaneously (colored + shaped + dashed line
    for a lineplot, colored shaped marker for a scatter), not N duplicate
    columns with the same row labels.

    The kind is ``"hue"`` so the merged entry is claimed by the ``hue``
    flag of ``legend={...}`` / ``legend_group(kind="hue")`` — ``hue`` is
    the anchor dimension when present, and this keeps behavior predictable.

    Only call this for the all-categorical case. Continuous hue (colorbar)
    and continuous size (size-tick swatches) cannot merge into a single
    composite artist — leave those as separate entries.

    Parameters
    ----------
    name : str
        Legend title (typically the shared column name).
    labels : list of str
        Labels for each legend row (same across all merged kinds).
    colors : list of str, optional
        Per-row fill colors (from the categorical hue palette).
    markers : list of str, optional
        Per-row marker symbols (from the style → marker map).
    linestyles : list of str or tuple, optional
        Per-row linestyles (from the style → linestyle map).
    sizes : list of float, optional
        Per-row marker sizes (only when a size dimension also merges).
    edgecolors : str or list of str, optional
        Per-row edge colors. Broadcast when passed as a string.
    handle_kwargs : dict, optional
        Extra kwargs forwarded to :func:`create_legend_handles`
        (``alpha``, ``linewidth``, ``markeredgewidth``, ``color``).

    Returns
    -------
    LegendEntry
        ``kind="hue"`` entry ready to stash via :func:`stash_entry`.
    """
    handle_kwargs = dict(handle_kwargs or {})
    handles = create_legend_handles(
        labels=labels,
        colors=colors,
        edgecolors=edgecolors,
        markers=markers,
        linestyles=linestyles,
        sizes=sizes,
        **handle_kwargs,
    )
    return LegendEntry.build(
        name=name,
        kind="hue",
        handles=handles,
        labels=labels,
    )


def resolve_size_map(
    size: Optional[str],
    data: "pd.DataFrame",
    size_order: Optional[List] = None,
    sizes: Optional[Union[List, Dict, Tuple[float, float]]] = None,
) -> Dict:
    """Resolve a ``{category: width}`` map for categorical size variables.

    Parallel to :func:`resolve_style_maps` but for the ``size`` dimension.
    Only call this after verifying the size column is categorical —
    numeric size uses the continuous path via :func:`get_size_ticks`.

    Parameters
    ----------
    size : str | None
        Name of the size column in ``data``. If None, returns an empty dict.
    data : pd.DataFrame
        Source data used to extract unique size values.
    size_order : list | None
        Optional ordering of size values. If None, uses
        ``data[size].unique()``.
    sizes : list | dict | tuple | None
        Size specification:

        - dict: explicit ``{category: width}`` map (returned as-is,
          filtered to ``size_order``).
        - list: widths to cycle through in order of the categories.
        - tuple ``(min, max)``: interpolate linearly across the categories
          (first → min, last → max).
        - None: falls back to a default ``(1.0, 4.0)`` interpolation.

    Returns
    -------
    dict
        Ordered ``{category: width}`` mapping.
    """
    if size is None:
        return {}

    # When the user passes an explicit ``sizes`` dict, its key order is a
    # clear signal of the legend order they want — ``data[size].unique()``
    # returns values in encounter order, which is rarely meaningful.
    # Explicit ``size_order`` still wins when provided.
    if isinstance(sizes, dict) and size_order is None:
        values = list(sizes.keys())
    else:
        values = list(data[size].unique() if size_order is None else size_order)
    n = len(values)
    if n == 0:
        return {}

    if isinstance(sizes, dict):
        return {v: sizes[v] for v in values if v in sizes}

    if isinstance(sizes, (list, tuple)) and len(sizes) == 2 and not isinstance(sizes, list):
        lo, hi = sizes
        if n == 1:
            return {values[0]: float(hi)}
        step = (hi - lo) / (n - 1)
        return {v: float(lo + i * step) for i, v in enumerate(values)}

    if isinstance(sizes, list):
        return {v: float(sizes[i % len(sizes)]) for i, v in enumerate(values)}

    # Default: interpolate between 1.0 and 4.0 (reasonable line/marker range).
    lo, hi = 1.0, 4.0
    if n == 1:
        return {values[0]: hi}
    step = (hi - lo) / (n - 1)
    return {v: float(lo + i * step) for i, v in enumerate(values)}


def resolve_style_maps(
    style: Optional[str],
    data: "pd.DataFrame",
    style_order: Optional[List],
    markers: Optional[Union[bool, List, Dict]] = None,
    dashes: Optional[Union[bool, List, Dict]] = None,
) -> Tuple[Dict, Dict]:
    """Resolve (marker_map, linestyle_map) for a categorical style variable.

    Either returned dict may be empty when the corresponding input is
    falsy / not requested.

    Parameters
    ----------
    style : str | None
        Name of the style column in ``data``. If None, both returned
        dicts are empty.
    data : pd.DataFrame
        Source data used to extract unique style values.
    style_order : list | None
        Optional ordering of style values. If None, uses ``data[style].unique()``.
    markers : bool | list | dict | None
        If truthy (non-False, non-None), resolve a marker map via
        :func:`publiplots.themes.markers.resolve_marker_map`. Pass-through
        when it's already a list/dict; otherwise the helper picks defaults.
    dashes : bool | list | dict | None
        Same as ``markers`` but for linestyles via
        :func:`publiplots.themes.linestyles.resolve_linestyle_map`.

    Returns
    -------
    (marker_map, linestyle_map) : tuple of dict
    """
    if style is None:
        return {}, {}

    style_values = list(data[style].unique() if style_order is None else style_order)

    marker_map: Dict = {}
    linestyle_map: Dict = {}

    if markers is not None and markers is not False:
        from publiplots.themes.markers import resolve_marker_map
        marker_param = markers if isinstance(markers, (list, dict)) else None
        marker_map = resolve_marker_map(values=style_values, marker_map=marker_param)

    if dashes is not None and dashes is not False:
        from publiplots.themes.linestyles import resolve_linestyle_map
        dash_param = dashes if isinstance(dashes, (list, dict)) else None
        linestyle_map = resolve_linestyle_map(values=style_values, linestyle_map=dash_param)

    return marker_map, linestyle_map


def get_size_ticks(
        values: np.ndarray,
        sizes: Tuple[float, float],
        size_norm: "Normalize",
        nbins: int = 4,
        min_n_ticks: int = 3,
        include_min_max: bool = False,
    ) -> Tuple[List[str], List[float]]:
    """
    Get size ticks for size legend.
    Uses MaxNLocator to generate ticks.
    Includes actual min and max from data.
    Rounds to reasonable precision.
    Falls back to min and max if no ticks are generated.

    Parameters
    ----------
    values : np.ndarray
        Values to get size ticks for.
    sizes : Tuple[float, float]
        (min_size, max_size) in points^2.
    size_norm : Normalize
        Size normalization object.
    nbins : int, default=4
        Number of bins used by MaxNLocator.
    min_n_ticks : int, default=3
        Minimum number of ticks to generate.
    include_min_max : bool, default=False
        Whether to include actual min and max from data.
        If True, the first and last tick will be the actual min and max.
        This is useful if the data is very small and the min and max are not representive.
    Returns
    -------
    tick_labels : List[str]
        Tick labels.
    tick_sizes : List[float]
        Tick sizes.
    """
    unique_vals = np.unique(values[~np.isnan(values)])
    v_min, v_max = size_norm.vmin, size_norm.vmax

    if len(unique_vals) <= 4:
        # If few unique values, show them all
        ticks = unique_vals
    else:
        # Use MaxNLocator but ensure we capture extremes
        locator = MaxNLocator(nbins=nbins, min_n_ticks=min_n_ticks)
        ticks = locator.tick_values(v_min, v_max)
        ticks = ticks[(ticks >= v_min) & (ticks <= v_max)]

        # Include actual min and max from data
        if include_min_max:
            ticks = np.unique(np.concatenate([[v_min], ticks, [v_max]]))

    # Round to reasonable precision
    if v_max - v_min > 10:
        ticks = np.array([int(np.round(t)) for t in ticks])
        ticks = np.unique(ticks)
    else:
        ticks = np.unique(np.round(ticks, 1))

    if ticks.size == 0:  # Fallback
        ticks = np.array([v_min, v_max])

    def _get_markersize(size: float) -> float:
        normalized_size = size_norm(size)
        actual_size = min(sizes[0] + normalized_size * (sizes[1] - sizes[0]), sizes[1])
        # Convert to markersize for legend
        return np.sqrt(actual_size / np.pi) * 2

    return [str(t) for t in ticks], [_get_markersize(t) for t in ticks]


_BUILDER_FORWARD_KEYS = frozenset({
    "inside", "loc", "bbox_to_anchor", "bbox_transform", "ncol",
    "frameon", "title_fontsize", "prop", "alignment", "borderpad",
    "borderaxespad", "handletextpad", "handlelength", "columnspacing",
    "labelspacing", "markerscale", "markerfirst",
})


def _builder_kwargs(legend_kws: Optional[Dict]) -> Dict:
    """Extract the subset of ``legend_kws`` meant for ``LegendBuilder``.

    The plot ``_legend`` helpers consume kind-specific keys (``hue_label``,
    ``hatch_label``, ``size_label``, ``size_nbins``, etc.) and leave the
    matplotlib ``ax.legend()`` passthrough keys behind. This filter picks
    those up and hands them to ``builder.add_legend``/``add_colorbar`` so
    users can pass e.g. ``legend_kws={'inside': True, 'loc': 'upper right'}``.
    """
    if not legend_kws:
        return {}
    return {k: v for k, v in legend_kws.items() if k in _BUILDER_FORWARD_KEYS}


def render_entries(
    ax: Axes,
    *,
    flags: Dict[str, bool],
    legend_kws: Optional[Dict] = None,
) -> None:
    """Render all stashed entries on ``ax`` not claimed by a figure-level group.

    For each stashed entry whose kind flag is True and that isn't claimed by
    the figure's legend_group, the handle is routed through the publiplots
    legend builder — continuous-hue (ScalarMappable) handles become a
    colorbar, everything else becomes a standard legend entry.

    If nothing remains to render, no builder is instantiated.

    ``legend_kws`` is the plot-function argument; kind-specific keys like
    ``hue_label`` should already have been popped by the caller. Remaining
    keys in :data:`_BUILDER_FORWARD_KEYS` are forwarded to the builder.
    """
    fig = ax.get_figure()
    to_render = [
        e for e in get_entries(ax)
        if flags[e.kind] and not entry_is_in_group(fig, e, ax=ax)
    ]
    if not to_render:
        return
    builder_kws = _builder_kwargs(legend_kws)
    # Route non-inside legends through a cached per-axes group. Entries
    # already claimed by a figure-level or multi-axes group registered by
    # the user are filtered out above via entry_is_in_group; what remains
    # is this axes' own legend contribution.
    #
    # inside=True short-circuits to a plain LegendBuilder — those legends
    # must not register with the reactor (tests pinned:
    # test_inside_true_skips_reactor_registration,
    # test_inside_coexists_with_legend_group).
    inside_mode = bool(legend_kws and legend_kws.get('inside'))
    if inside_mode:
        from publiplots.utils.legend import LegendBuilder
        builder = LegendBuilder(ax, external_to_axis=False)
    else:
        from publiplots.utils.legend_group import _get_or_create_per_axes_group
        group = _get_or_create_per_axes_group(ax)
        builder = group._builder
    for entry in to_render:
        if entry.kind == "hue" and is_continuous_hue(entry.handles):
            builder.add_colorbar(
                mappable=entry.handles[0],
                label=entry.name,
                **builder_kws,
            )
        else:
            builder.add_legend(
                handles=list(entry.handles),
                label=entry.name,
                **builder_kws,
            )
