"""Wedge artist construction, axes framing and meta emission."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from matplotlib.axes import Axes
from matplotlib.patches import Wedge

from publiplots.annotate._cache import WedgeMeta, WedgeRecord
from publiplots.plot.wedge.geometry import WedgePlacement

MM_PER_INCH = 25.4


def pull_offset(placement: WedgePlacement, pull: float) -> Tuple[float, float]:
    """Displacement along the wedge's mid-angle, as a fraction of radius."""
    if not pull:
        return (0.0, 0.0)
    mid = math.radians((placement.theta0 + placement.theta1) / 2.0)
    return (pull * math.cos(mid), pull * math.sin(mid))


def draw_wedges(
    ax: Axes,
    placements: List[WedgePlacement],
    colors: Dict[Tuple[Any, ...], Tuple[float, float, float]],
    hatches: Dict[Tuple[Any, ...], Optional[str]],
    *,
    center: Tuple[float, float],
    pull: Dict[Tuple[Any, ...], float],
    alpha: float,
    edgecolor,
    linewidth: float,
) -> List[Wedge]:
    """Add one Wedge per placement, in draw order, and return them."""
    patches: List[Wedge] = []
    for place in placements:
        dx, dy = pull_offset(place, pull.get(place.node.path, 0.0))
        patch = Wedge(
            (center[0] + dx, center[1] + dy),
            place.r_outer,
            place.theta0,
            place.theta1,
            width=place.r_outer - place.r_inner,
            facecolor=colors[place.node.path],
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
            hatch=hatches.get(place.node.path),
        )
        ax.add_patch(patch)
        patches.append(patch)
    return patches


def build_wedge_meta(
    placements: List[WedgePlacement],
    patches: List[Wedge],
    colors: Dict[Tuple[Any, ...], Tuple[float, float, float]],
    hatches: Dict[Tuple[Any, ...], Optional[str]],
    pull: Dict[Tuple[Any, ...], float],
    *,
    total: float,
    donut: float,
    center: Tuple[float, float],
    levels: Sequence[str],
    hue_active: bool,
    hue_continuous: bool,
    source_frame: Optional[Any],
) -> WedgeMeta:
    """Pair each placement with the artist drawn for it, positionally --
    and safely, because both lists were produced by the loop above in the
    same pass. This is the only place the two meet, and neither is
    recovered from the axes.
    """
    if len(placements) != len(patches):
        raise AssertionError(
            f"wedge meta: {len(placements)} placements against "
            f"{len(patches)} patches"
        )
    parent_values = {p.node.path: p.node.value for p in placements}
    records = []
    for place, patch in zip(placements, patches):
        node = place.node
        parent_total = parent_values.get(node.parent, total) if node.parent \
            else total
        records.append(WedgeRecord(
            patch=patch,
            path=node.path,
            value=node.value,
            fraction_of_total=node.value / total if total else 0.0,
            fraction_of_parent=(node.value / parent_total
                                if parent_total else 0.0),
            theta0=place.theta0,
            theta1=place.theta1,
            r_inner=place.r_inner,
            r_outer=place.r_outer,
            depth=node.depth,
            hue_color=colors[node.path],
            hatch=hatches.get(node.path),
            pull=pull.get(node.path, 0.0),
        ))
    return WedgeMeta(
        wedges=records,
        total=total,
        donut=donut,
        center=center,
        hue_active=hue_active,
        hue_continuous=hue_continuous,
        levels=tuple(levels),
        source_frame=source_frame,
    )


def apply_cell_limits(ax: Axes, *, radius: float, aspect: str) -> None:
    """Frame the axes so the chart is circular.

    ``aspect="cell"`` leaves the aspect unconstrained and sets the data
    limits to the axes' own millimetre ratio. Measured on a 70x50mm cell,
    ``set_aspect("equal")`` shrinks the drawn box to 0.6008 of the figure
    against an allocated 0.8411 -- issue #101 -- while matching the cell
    keeps the full allocation and still renders a disc square to within a
    pixel. The tradeoff is that the limits are computed once, so an
    interactive resize would distort the circle; ``aspect="equal"`` is
    available for that case.
    """
    if aspect not in ("cell", "equal"):
        raise ValueError(
            f"aspect must be 'cell' or 'equal'; got {aspect!r}"
        )
    ax.set_axis_off()
    if aspect == "equal":
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_aspect("equal")
        return

    fig = ax.figure
    box = ax.get_position()
    w_mm = box.width * fig.get_figwidth() * MM_PER_INCH
    h_mm = box.height * fig.get_figheight() * MM_PER_INCH
    ratio = (w_mm / h_mm) if h_mm else 1.0
    ax.set_ylim(-radius, radius)
    ax.set_xlim(-radius * ratio, radius * ratio)
