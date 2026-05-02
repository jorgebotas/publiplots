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
from publiplots.utils.legend import legend as legend_fn
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

    render_entries(ax, flags=flags)


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


def render_entries(ax: Axes, *, flags: Dict[str, bool]) -> None:
    """Render all stashed entries on ``ax`` not claimed by a figure-level group.

    For each stashed entry whose kind flag is True and that isn't claimed by
    the figure's legend_group, the handle is routed through the publiplots
    legend builder — continuous-hue (ScalarMappable) handles become a
    colorbar, everything else becomes a standard legend entry.

    If nothing remains to render, no builder is instantiated.
    """
    fig = ax.get_figure()
    to_render = [
        e for e in get_entries(ax)
        if flags[e.kind] and not entry_is_in_group(fig, e)
    ]
    if not to_render:
        return
    builder = legend_fn(ax=ax, auto=False)
    for entry in to_render:
        if entry.kind == "hue" and is_continuous_hue(entry.handles):
            builder.add_colorbar(
                mappable=entry.handles[0],
                label=entry.name,
            )
        else:
            builder.add_legend(
                handles=list(entry.handles),
                label=entry.name,
            )
