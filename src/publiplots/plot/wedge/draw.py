"""Wedge artist construction, axes framing and meta emission."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Wedge
from matplotlib.path import Path

from publiplots.annotate._cache import WedgeMeta, WedgeRecord
from publiplots.annotate._positioning import mm_to_data
from publiplots.plot.wedge.geometry import WedgePlacement

MM_PER_INCH = 25.4


def pull_offset(placement: WedgePlacement, pull: float) -> Tuple[float, float]:
    """Displacement along the wedge's mid-angle, as a fraction of radius."""
    if not pull:
        return (0.0, 0.0)
    mid = math.radians((placement.theta0 + placement.theta1) / 2.0)
    return (pull * math.cos(mid), pull * math.sin(mid))


def even_explode(pull: float, r_outer: float) -> float:
    """The even explode distance of a scalar ``pull``, in data units.

    ``pull`` is a fraction of the outer radius, so the distance is
    ``t = pull * r_outer``. The explode it describes is not a
    translation: every wedge stays centred on the chart centre, its inner
    and outer radii both grow by ``t``, and both its straight edges are
    inset by ``t``. Every wedge therefore moves out by the same distance
    ``t`` and every pair of neighbours ends up the same perpendicular
    distance ``2 * t`` apart, whatever either wedge's span -- a
    translation along each wedge's own mid-ray opens only
    ``t * sin(half-span)`` per side, so thin wedges barely separate.

    Moving every wedge out by the same distance, giving every pair of
    neighbours the same gap and keeping each wedge's shape cannot all
    hold at once; this keeps the first two. A wedge grows by ``t`` at
    both radii, and none reaches inside ``r_inner + t`` -- a hole of
    radius ``t`` in a pie. Where the inset leaves room there, a wedge
    ends in an arc on that radius; a narrower one ends further out,
    where its inset edges meet. A lone 360-degree wedge has no edges to
    inset and is a full ring from ``r_inner + t`` to ``r_outer + t``.
    A wedge too thin for the full inset is capped so its rim keeps half
    its span (see :class:`GappedWedge`); it is never dropped. It still
    stays outside the hole: a convex wedge ends in the inner arc where
    the apex would fall inside ``r_inner + t``, and in the apex only
    where the apex lies further out.
    For the two halves of a half-disc the
    straight edges, the ``2 * t`` gap and the rim on each mid-ray match a
    translation by ``t`` exactly, but the rim is an arc of radius
    ``r_outer + t`` about the chart centre rather than of ``r_outer``
    about the translated one, so the shapes differ by up to about ``t``
    at the rim's ends: 0.976 ``t`` at ``pull=0.05``.
    """
    return float(pull) * float(r_outer)


class GappedWedge(Wedge):
    """A :class:`~matplotlib.patches.Wedge` whose straight edges are inset.

    Each of the two straight edges is moved ``inset`` data units into the
    wedge, parallel to itself, while the arcs stay on their radii. Two
    neighbouring wedges inset by ``h`` each therefore leave a band of
    constant width ``2 * h`` along the ray they share, whatever either
    wedge's span -- unlike a radial ``pull``, whose sideways separation
    grows with the span.

    **No wedge is inset out of existence.** A wedge of at most 180
    degrees is inset by ``min(inset, r * sin(half / 2))`` instead, with
    ``r`` its outer radius and ``half`` its half-span, so its rim arc
    always keeps at least half of its angular span -- half its arc length
    at ``r``, and so at least half its rim chord too. A wedge that thin
    is still drawn, narrowed, and the band beside it is narrower by what
    it was spared: ``inset + get_drawn_inset()`` to a full-inset
    neighbour, the two drawn insets summed between two capped ones.
    :meth:`get_drawn_inset` returns the inset actually applied.

    With ``inset == 0`` the path is matplotlib's own, vertex for vertex.
    Otherwise, with ``h`` the drawn inset:

    - the outer arc runs from ``theta1 + asin(h / r)`` to
      ``theta2 - asin(h / r)``;
    - an inner radius ``r_in > h`` whose inset arc keeps a positive
      extent (``asin(h / r_in) < half``) is inset the same way;
    - otherwise the two offset edges meet at an apex on the mid-ray, at
      ``h / sin(half)`` from the centre. That holds for a reflex wedge
      too (``90 < half < 180``), whose apex lies past the centre on its
      own side. For a capped wedge the apex is at least
      ``r / (2 * cos(half / 2))``, more than half the outer radius out;
    - a full 360-degree wedge has no straight edges and is not inset.

    A reflex wedge is never capped. Its rim can only vanish when
    everything else in the ring together is too thin for the gap, and
    then there is nothing left for it to keep a gap from, so it is drawn
    uninset, as a lone wedge would be.

    Parameters
    ----------
    center, r, theta1, theta2, width
        As for :class:`~matplotlib.patches.Wedge`.
    inset : float, default 0.0
        Requested perpendicular inset of each straight edge, in data
        units.
    **kwargs
        Forwarded to :class:`~matplotlib.patches.Wedge`.
    """

    def __init__(
        self,
        center: Tuple[float, float],
        r: float,
        theta1: float,
        theta2: float,
        *,
        width: Optional[float] = None,
        inset: float = 0.0,
        **kwargs: Any,
    ) -> None:
        # Set before the base initialiser, which builds the path.
        self.inset = float(inset)
        super().__init__(center, r, theta1, theta2, width=width, **kwargs)

    def set_inset(self, inset: float) -> None:
        """Set the requested perpendicular edge inset, in data units."""
        self._path = None
        self.inset = float(inset)
        self.stale = True

    def get_inset(self) -> float:
        """Return the requested perpendicular edge inset, in data units."""
        return self.inset

    def get_drawn_inset(self) -> float:
        """Return the inset the path is drawn with, in data units.

        The requested inset, capped for a wedge of at most 180 degrees so
        its rim keeps half its span; ``0`` for a full circle, and for a
        reflex wedge drawn uninset because its rim would vanish.
        """
        h = self.inset
        span = self.theta2 - self.theta1
        if h <= 0.0 or abs(span - 360.0) <= 1e-12:
            return 0.0
        half = math.radians(span) / 2.0
        if half <= math.pi / 2.0:
            return min(h, self.r * math.sin(half / 2.0))
        r_in = (self.r - self.width) if self.width is not None else 0.0
        # Its rim vanishes: past the inset, or split by a slot running
        # past it where the inset edges meet beyond the rim.
        if self.r <= h or (r_in <= h and h / math.sin(half) >= self.r):
            return 0.0
        return h

    def _recompute_path(self) -> None:
        h = self.get_drawn_inset()
        if h <= 0.0:
            super()._recompute_path()
            return

        span = self.theta2 - self.theta1
        r_out = self.r
        r_in = (r_out - self.width) if self.width is not None else 0.0
        half = math.radians(span) / 2.0
        mid = math.radians(self.theta1) + half
        center = np.asarray(self.center, dtype=float)

        def arc_angles(radius: float) -> Optional[Tuple[float, float]]:
            """The inset arc's end angles at ``radius``, or None if empty."""
            if radius <= h:
                return None
            shift = math.asin(h / radius)
            if shift >= half:
                return None
            return (self.theta1 + math.degrees(shift),
                    self.theta2 - math.degrees(shift))

        # The cap keeps the outer arc at least half the span, and
        # get_drawn_inset has already sent a vanishing reflex wedge to
        # the uninset path, so the outer arc always exists here.
        outer = arc_angles(r_out)
        inner = arc_angles(r_in) if r_in > 0.0 else None
        arc = Path.arc(*outer)
        if inner is not None:
            back = Path.arc(*inner)
            vertices = np.concatenate([
                arc.vertices * r_out,
                back.vertices[::-1] * r_in,
                [(0.0, 0.0)],
            ])
            codes = [*arc.codes, Path.LINETO, *back.codes[1:],
                     Path.CLOSEPOLY]
        else:
            # sin(half) > 0 for every span short of a full circle.
            apex = (h / math.sin(half)) * np.array([math.cos(mid),
                                                   math.sin(mid)])
            vertices = np.concatenate([arc.vertices * r_out, [apex, apex]])
            codes = [*arc.codes, Path.LINETO, Path.CLOSEPOLY]
        self._path = Path(vertices + center, codes)


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
    explode: float = 0.0,
) -> List[GappedWedge]:
    """Add one wedge per placement, in draw order, and return them.

    Every wedge is drawn with round joins. At a thin wedge's apex the two
    straight edges meet at an acute angle, and the default miter join
    there spikes past the centre into the neighbouring wedges -- visibly
    so at 600 dpi on the 6% wedge of a 52.2/41.8/6/4 pie.

    ``pull`` translates the wedges it names along their own mid-rays.
    ``explode`` is the even explode instead, in data units: every wedge
    stays centred on ``center``, its inner and outer radii both grow by
    ``explode``, and each straight edge is inset by ``explode``, so all
    of them move out by the same distance and every pair of neighbours
    sits ``2 * explode`` apart (see :func:`even_explode`).

    The wedges start inset by ``explode`` alone; :func:`apply_gap` adds
    the gap once the limits are framed.
    """
    patches: List[GappedWedge] = []
    for place in placements:
        dx, dy = pull_offset(place, pull.get(place.node.path, 0.0))
        patch = GappedWedge(
            (center[0] + dx, center[1] + dy),
            place.r_outer + explode,
            place.theta0,
            place.theta1,
            width=place.r_outer - place.r_inner,
            inset=explode,
            facecolor=colors[place.node.path],
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
            hatch=hatches.get(place.node.path),
            joinstyle="round",
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
    explode: float = 0.0,
) -> WedgeMeta:
    """Pair each placement with the artist drawn for it, positionally --
    and safely, because both lists were produced by the loop above in the
    same pass. This is the only place the two meet, and neither is
    recovered from the axes.

    Each record describes the wedge as drawn: an even ``explode`` (data
    units, as passed to :func:`draw_wedges`) is folded into ``r_inner``
    and ``r_outer``, so a label placed from the record lands on the
    radii the wedge actually has.
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
            r_inner=place.r_inner + explode,
            r_outer=place.r_outer + explode,
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


def apply_gap(ax: Axes, patches: Sequence[GappedWedge], gap: float,
              *, explode: float = 0.0) -> None:
    """Inset every wedge so neighbours sit ``gap`` millimetres apart.

    Each straight edge moves ``gap / 2`` millimetres into its wedge, so
    any two wedges sharing a ray leave a band exactly ``gap`` wide
    between their outlines' centre lines (see :class:`GappedWedge`).
    With an even ``explode`` (data units) each edge is inset by
    ``explode`` on top, so the band is ``gap`` plus ``2 * explode``.

    The conversion to data units reads the axes' current transform, so
    it must run after :func:`apply_cell_limits` has framed the limits.
    ``apply_aspect`` comes first so that under ``aspect="equal"`` it
    reads the square matplotlib will draw rather than the full cell, as
    ``annotate``'s wedge strategy does; under ``aspect="cell"`` the call
    changes nothing. Both aspects give equal x and y scales, so the
    y-axis conversion holds in every direction. A ``gap`` of 0 returns
    before touching the axes, leaving the drawing exactly as it was.
    """
    if not gap:
        return
    ax.apply_aspect()
    inset = abs(mm_to_data(gap / 2.0, ax, "y")) + explode
    for patch in patches:
        patch.set_inset(inset)
