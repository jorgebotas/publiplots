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

from typing import Dict, List, Optional, Union

from matplotlib.axes import Axes

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
