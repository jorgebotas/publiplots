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


def set_hatch_linewidth(patch: Any, linewidth: float) -> None:
    """Draw ``patch``'s hatch strokes at ``linewidth``.

    A hatch stroke outlines the shape it fills, so it takes the outline's
    ``edgewidth`` rather than ``rcParams["hatch.linewidth"]`` -- the same
    rule ``barplot`` and ``histplot`` apply to their patches. The
    ``hasattr`` guard matches the one in ``utils/rounding.py``: a patch
    without ``set_hatch_linewidth`` keeps the rcParams width rather than
    raising.
    """
    if hasattr(patch, "set_hatch_linewidth"):
        patch.set_hatch_linewidth(linewidth)


def draw_wedges(
    ax: Axes,
    placements: List[WedgePlacement],
    colors: Dict[Tuple[Any, ...], Tuple[float, float, float]],
    hatches: Dict[Tuple[Any, ...], Optional[str]],
    *,
    center: Tuple[float, float],
    pull: Dict[Tuple[Any, ...], float],
    alpha: float,
    edgecolor: Optional[Any],
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
        set_hatch_linewidth(patch, linewidth)
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
        if node.parent is None:
            parent_total = total
        elif node.parent in parent_values:
            parent_total = parent_values[node.parent]
        else:
            raise ValueError(
                f"wedge meta: node {node.path!r} has parent "
                f"{node.parent!r}, which is not among the placements "
                "passed in -- fraction_of_parent cannot be computed "
                "without it"
            )
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
    limits to the axes' own millimetre ratio: the disc is sized to the
    **shorter** side of the cell and the limits on the longer side are
    stretched to the cell's ratio, so the axes box keeps the whole
    allocation on both axes and the surplus on the longer side stays
    free (which is where outside labels and leader lines go).
    ``set_aspect("equal")`` draws the same disc, but has matplotlib
    shrink the drawn *box* to a square inside the allocated one -- issue
    #101 -- keeping only ``min(w, h) / max(w, h)`` of the allocation on
    whichever axis is longer.

    Measured by rendering a pie to an Agg buffer and reading the ink
    bounding box; "box kept" is ``get_position()`` over
    ``get_position(original=True)``:

    ======  =====================  =======  ==============  ===============
    cell    ``"cell"`` ink (mm)    ink w/h  ``"cell"`` box  ``"equal"`` box
    ======  =====================  =======  ==============  ===============
    70x50   47.92 x 47.92           1.0000  1.0000x1.0000   0.7143x1.0000
    90x40   38.44 x 38.61           0.9956  1.0000x1.0000   0.4444x1.0000
    40x70   38.61 x 38.44           1.0044  1.0000x1.0000   1.0000x0.5714
    30x80   28.96 x 29.13           0.9942  1.0000x1.0000   1.0000x0.3750
    60x60   57.57 x 57.57           1.0000  1.0000x1.0000   1.0000x1.0000
    ======  =====================  =======  ==============  ===============

    The ink ratios away from 1 are one rendering pixel on a disc a few
    hundred pixels across; the disc itself is round by construction in
    both modes. The tradeoff for ``"cell"`` is that the limits are
    computed once, so an interactive resize would distort the circle;
    ``aspect="equal"`` is available for that case.
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
    ratio = (w_mm / h_mm) if (w_mm and h_mm) else 1.0
    # Size the disc to the SHORTER side and stretch the other axis' limits
    # to the cell ratio. Sizing unconditionally to the height put the disc
    # wider than the axes on any cell taller than wide, and matplotlib
    # clipped it (13.33mm off each side of a 40x70mm cell).
    if ratio >= 1.0:
        ax.set_ylim(-radius, radius)
        ax.set_xlim(-radius * ratio, radius * ratio)
    else:
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius / ratio, radius / ratio)
