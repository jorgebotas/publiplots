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
from publiplots.plot.wedge.geometry import (
    WedgeNode,
    WedgePlacement,
    has_children,
)

MM_PER_INCH = 25.4

#: Relative tolerance at which a radius counts as equal to an edge inset.
INSET_TOLERANCE = 1e-9


def arc_survives_inset(radius: float, inset: float) -> bool:
    """Whether an arc at ``radius`` reaches past straight edges inset by
    ``inset``: the comparison that decides whether a wedge ends in an
    inner arc or in the apex where its inset edges meet.

    Under an even explode a pie wedge's inner radius equals its inset in
    exact arithmetic, and floating point puts the computed radius a hair
    to either side, so a strict ``radius > inset`` drew the same wedge
    with an inner arc at one ``pull`` and an apex at the next. A
    ``radius`` within ``INSET_TOLERANCE * max(radius, inset)`` of
    ``inset`` therefore counts as equal, on the side that keeps the arc.

    :class:`GappedWedge` draws by this rule and the wedge label placement
    reads the drawn wedge by it, so the two cannot disagree.
    """
    return radius >= inset - INSET_TOLERANCE * max(radius, inset)


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

    The two edges can be inset by different amounts (:meth:`set_insets`):
    ``h0`` for the edge at ``theta1`` and ``h1`` for the edge at
    ``theta2``. Each is capped on its own, by the rule above, against the
    outer radius ``r`` as it is when the path is drawn.
    :meth:`get_drawn_insets` returns the pair actually applied.

    With both insets ``0`` the path is matplotlib's own, vertex for
    vertex. Otherwise, with ``h0`` and ``h1`` the drawn insets:

    - the outer arc runs from ``theta1 + asin(h0 / r)`` to
      ``theta2 - asin(h1 / r)``;
    - each half of the wedge, from an edge to the mid-ray, ends inside
      as a wedge inset by that edge's ``h`` on both sides would: an inner
      radius ``r_in >= h`` (to within :func:`arc_survives_inset`'s
      tolerance) whose inset arc keeps a positive extent
      (``asin(h / r_in) < half``) is inset the same way, from the edge to
      the mid-ray;
    - otherwise that half ends where its offset edge meets the mid-ray, at
      ``h / sin(half)`` from the centre. That holds for a reflex wedge too
      (``90 < half < 180``), whose apex lies past the centre on its own
      side. For a capped wedge the apex is at least
      ``r / (2 * cos(half / 2))``, more than half the outer radius out.
      With equal insets the two halves end alike, in one inner arc or in
      one apex;
    - with unequal insets the two halves are joined along the mid-ray, by
      a radial step between their inner ends where those differ. Each
      half lies inside the annular sector and outside the hole as the
      equal-inset wedge does, and each edge keeps its own inset from the
      rim to its half's inner end, so seams through the rings of a
      sunburst stay collinear. Nothing is solved, so the parallel edges
      of a 180-degree wedge need no special case;
    - a full 360-degree wedge has no straight edges and is not inset.

    A reflex wedge is never capped. Its rim can only vanish when
    everything else in the ring together is too thin for the gap, and
    then there is nothing left for it to keep a gap from, so it is drawn
    uninset, as a lone wedge would be. With unequal insets that is
    decided against the larger one.

    :meth:`set_radial_insets` moves the outer radius in and the inner
    radius out, for the gap between the rings of a sunburst.

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
        self.insets: Tuple[float, float] = (float(inset), float(inset))
        super().__init__(center, r, theta1, theta2, width=width, **kwargs)

    @property
    def inset(self) -> float:
        """The larger of the two requested edge insets, in data units."""
        return max(self.insets)

    def set_inset(self, inset: float) -> None:
        """Set the requested perpendicular inset of both straight edges,
        in data units."""
        self.set_insets(inset, inset)

    def set_insets(self, h0: float, h1: float) -> None:
        """Set the requested perpendicular insets of the edge at
        ``theta1`` (``h0``) and the edge at ``theta2`` (``h1``), in data
        units."""
        self._path = None
        self.insets = (float(h0), float(h1))
        self.stale = True

    def get_inset(self) -> float:
        """Return the larger requested edge inset, in data units."""
        return self.inset

    def set_radial_insets(self, inner: float, outer: float) -> None:
        """Move the outer radius in by ``outer`` and the inner radius out
        by ``inner``, in data units."""
        self.set_radius(self.r - outer)
        self.set_width(self.width - inner - outer)

    def get_drawn_insets(self) -> Tuple[float, float]:
        """Return the insets the path is drawn with, in data units: the
        edge at ``theta1``, then the edge at ``theta2``.

        Each requested inset, capped for a wedge of at most 180 degrees
        so its rim keeps half its span -- against the current outer
        radius, so after any radial inset; ``(0, 0)`` for a full circle,
        and for a reflex wedge drawn uninset because its rim would vanish.
        """
        h0, h1 = (max(h, 0.0) for h in self.insets)
        span = self.theta2 - self.theta1
        if max(h0, h1) <= 0.0 or abs(span - 360.0) <= 1e-12:
            return (0.0, 0.0)
        half = math.radians(span) / 2.0
        if half <= math.pi / 2.0:
            cap = self.r * math.sin(half / 2.0)
            return (min(h0, cap), min(h1, cap))
        h = max(h0, h1)
        r_in = (self.r - self.width) if self.width is not None else 0.0
        # Its rim vanishes: past the inset, or split by a slot running
        # past it where the inset edges meet beyond the rim.
        if self.r <= h or (not arc_survives_inset(r_in, h)
                           and h / math.sin(half) >= self.r):
            return (0.0, 0.0)
        return (h0, h1)

    def get_drawn_inset(self) -> float:
        """Return the larger of :meth:`get_drawn_insets`, in data units."""
        return max(self.get_drawn_insets())

    def _recompute_path(self) -> None:
        h0, h1 = self.get_drawn_insets()
        if h0 <= 0.0 and h1 <= 0.0:
            super()._recompute_path()
            return

        span = self.theta2 - self.theta1
        r_out = self.r
        r_in = (r_out - self.width) if self.width is not None else 0.0
        half = math.radians(span) / 2.0
        mid = math.radians(self.theta1) + half
        center = np.asarray(self.center, dtype=float)

        def shift_at(radius: float, h: float) -> float:
            """The angle an edge inset by ``h`` moves in at ``radius``."""
            # Within the tolerance h / radius can be a hair above 1.
            return math.asin(min(h / radius, 1.0))

        def inner_shift(h: float) -> Optional[float]:
            """A half's inner arc shift at ``r_in``, or None where the half
            ends in its corner on the mid-ray."""
            if r_in <= 0.0 or not arc_survives_inset(r_in, h):
                return None
            shift = shift_at(r_in, h)
            return shift if shift < half else None

        # The cap keeps the outer arc at least half the span, and
        # get_drawn_insets has already sent a vanishing reflex wedge to
        # the uninset path, so the outer arc always exists here.
        arc = Path.arc(self.theta1 + math.degrees(shift_at(r_out, h0)),
                       self.theta2 - math.degrees(shift_at(r_out, h1)))
        shift0, shift1 = inner_shift(h0), inner_shift(h1)
        if shift0 is not None and shift1 is not None:
            back = Path.arc(self.theta1 + math.degrees(shift0),
                            self.theta2 - math.degrees(shift1))
            vertices = np.concatenate([
                arc.vertices * r_out,
                back.vertices[::-1] * r_in,
                [(0.0, 0.0)],
            ])
            codes = [*arc.codes, Path.LINETO, *back.codes[1:],
                     Path.CLOSEPOLY]
        else:
            # The path runs in along the theta2 edge, through half 1's
            # inner end, along the mid-ray to half 0's, and out along the
            # theta1 edge.
            mid_deg = self.theta1 + span / 2.0
            # sin(half) > 0 for every span short of a full circle.
            ray = np.array([math.cos(mid), math.sin(mid)])
            pieces = [arc.vertices * r_out]
            codes = [*arc.codes]
            if shift1 is not None:
                back = Path.arc(mid_deg, self.theta2 - math.degrees(shift1))
                pieces.append(back.vertices[::-1] * r_in)
                codes += [Path.LINETO, *back.codes[1:]]
            else:
                pieces.append([(h1 / math.sin(half)) * ray])
                codes.append(Path.LINETO)
            if shift0 is not None:
                back = Path.arc(self.theta1 + math.degrees(shift0), mid_deg)
                pieces.append(back.vertices[::-1] * r_in)
                codes += [Path.LINETO, *back.codes[1:]]
            elif shift1 is not None or h0 != h1:
                pieces.append([(h0 / math.sin(half)) * ray])
                codes.append(Path.LINETO)
            pieces.append(pieces[-1][-1:])
            codes.append(Path.CLOSEPOLY)
            vertices = np.concatenate(pieces)
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
    vectors: Dict[Tuple[Any, ...], Tuple[float, float]],
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

    ``vectors`` translates the wedges it names: each is centred on
    ``center`` plus its vector, in data units (a dict ``pull`` resolved
    by ``render.resolve_pull_vectors``); an unnamed wedge stays on
    ``center``.
    ``explode`` is the even explode instead, in data units: every wedge
    stays centred on ``center``, its inner and outer radii both grow by
    ``explode``, and each straight edge is inset by ``explode``, so all
    of them move out by the same distance and every pair of neighbours
    sits ``2 * explode`` apart (see :func:`even_explode`).

    On a sunburst every ring keeps its width and moves out by ``2 *
    explode`` more than the ring inside it: a depth-``k`` wedge's outer
    radius grows by ``(2 * k + 1) * explode``, so every ring boundary opens
    to ``2 * explode`` as every seam does. Depth 0 is the pie's ``r +
    explode``.

    The wedges start inset by ``explode`` alone; :func:`apply_gap` adds
    the gap once the limits are framed.
    """
    patches: List[GappedWedge] = []
    for place in placements:
        dx, dy = vectors.get(place.node.path, (0.0, 0.0))
        patch = GappedWedge(
            (center[0] + dx, center[1] + dy),
            place.r_outer + (2 * place.node.depth + 1) * explode,
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
    chart: str = "pie",
) -> WedgeMeta:
    """Pair each placement with the artist drawn for it, positionally --
    and safely, because both lists were produced by the loop above in the
    same pass. This is the only place the two meet, and neither is
    recovered from the axes.

    Each record describes the wedge as drawn: its angles and radii are
    read off the patch, after the gap and any even explode have been
    applied, so a label placed from the record lands on the wedge as it
    is.
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
            theta0=patch.theta1,
            theta1=patch.theta2,
            r_inner=patch.r - patch.width,
            r_outer=patch.r,
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
        chart=chart,
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


def ray_key(angle: float) -> float:
    """The ray at ``angle`` degrees as a key in ``[0, 360)``.

    Angles a whole number of turns apart, or apart by float noise (the
    child and parent edges on one ray are computed from different sums),
    give the same key: ``-1e-13`` and ``360 - 1e-13`` both give ``0.0``.
    """
    key = round(float(angle) % 360.0, 9)
    return 0.0 if key == 360.0 else key


def _cap_eligible_half(patch: GappedWedge) -> Optional[float]:
    """The half-span in radians of a wedge the never-vanish cap applies
    to (at most 180 degrees), or None -- the same test as
    :meth:`GappedWedge.get_drawn_insets`."""
    span = patch.theta2 - patch.theta1
    if abs(span - 360.0) <= 1e-12:
        return None
    half = math.radians(span) / 2.0
    return half if half <= math.pi / 2.0 else None


def ray_side_caps(
    patches: Sequence[GappedWedge],
) -> Dict[Tuple[float, int], float]:
    """The never-vanish cap on each side of every ray, in data units.

    Keyed by ``(ray_key(angle), side)``: side ``+1`` is the edge at a
    patch's ``theta1``, whose wedge lies counterclockwise of the ray, and
    ``-1`` the edge at its ``theta2``. Each value is the smallest
    ``r * sin(half / 2)`` over the patches of at most 180 degrees with an
    edge there, read from each patch's current outer radius; ``inf``
    where only wider patches, or full circles, touch that side.
    """
    caps: Dict[Tuple[float, int], float] = {}
    for patch in patches:
        half = _cap_eligible_half(patch)
        cap = (patch.r * math.sin(half / 2.0) if half is not None
               else math.inf)
        for key in ((ray_key(patch.theta1), 1), (ray_key(patch.theta2), -1)):
            caps[key] = min(caps.get(key, math.inf), cap)
    return caps


def apply_gap(ax: Axes, patches: Sequence[GappedWedge], gap: float,
              *, explode: float = 0.0,
              nodes: Optional[Sequence[WedgeNode]] = None,
              n_levels: int = 1) -> None:
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

    ``nodes`` (one per patch, in the same order) makes the patches the
    rings of a sunburst, ``n_levels`` deep; ``None`` is a pie, drawn as
    above. With rings:

    - **Between rings** each band moves its inner radius out (below the
      root ring) and its rim in (where it has children) by ``gap / 2``,
      but never past half its width: at most ``w / 4`` a side when both
      sides move, ``w / 2`` when one does. The rim of a leaf, ragged or
      not, and the edge of the hole stay put. This runs first, so the
      caps below see each wedge's final outer radius.
    - **Between siblings** each side of every ray is capped as a whole:
      an edge is inset by ``min(gap / 2 + explode, cap)``, with ``cap``
      the smallest ``r * sin(half / 2)`` over the wedges of at most 180
      degrees on that side of that ray, in every ring. A parent and its
      first or last child share a side, so the seam runs straight
      through the rings even where a thin wedge narrows it.
    - **A wedge wider than half the circle** whose inner radius lies
      inside its larger edge inset has that inner radius moved out to
      the inset, so it ends in an inner arc and keeps both edge gaps
      instead of being drawn uninset: a root wedge with no hole keeps
      a hole of radius ``gap / 2`` plus any explode. A wedge whose rim
      lies inside the inset is left as it is, drawn uninset.

    An explode without a gap still runs this, so exploded slivers share
    their caps too; the radial insets are then 0.
    """
    if nodes is None:
        if not gap:
            return
        ax.apply_aspect()
        inset = abs(mm_to_data(gap / 2.0, ax, "y")) + explode
        for patch in patches:
            patch.set_inset(inset)
        return

    if not gap and not explode:
        return
    if len(nodes) != len(patches):
        raise AssertionError(
            f"apply_gap: {len(nodes)} nodes against {len(patches)} patches")
    if any(node.depth >= n_levels for node in nodes):
        raise AssertionError(
            f"apply_gap: a node lies deeper than the {n_levels} rings")
    h = 0.0
    if gap:
        ax.apply_aspect()
        h = abs(mm_to_data(gap / 2.0, ax, "y"))

    parents = has_children(list(nodes))
    for patch, node in zip(patches, nodes):
        inner_side = node.depth > 0
        outer_side = node.path in parents
        sides = int(inner_side) + int(outer_side)
        if not sides or not h:
            continue
        step = min(h, patch.width / (2.0 * sides))
        patch.set_radial_insets(step if inner_side else 0.0,
                                step if outer_side else 0.0)

    caps = ray_side_caps(patches)
    full = h + explode
    for patch in patches:
        patch.set_insets(min(full, caps[(ray_key(patch.theta1), 1)]),
                         min(full, caps[(ray_key(patch.theta2), -1)]))
        # A reflex wedge reaching inside its inset would end in an apex
        # past its own rim and be drawn uninset; give it a hole of the
        # inset's radius so its edges keep the gap. Not where the rim
        # itself lies inside the inset: that wedge is drawn uninset.
        span = patch.theta2 - patch.theta1
        if span <= 180.0 or abs(span - 360.0) <= 1e-12:
            continue
        h_max = patch.inset
        r_in = patch.r - patch.width
        if (r_in < h_max and patch.r > h_max
                and arc_survives_inset(patch.r, h_max)):
            patch.set_radial_insets(h_max - r_in, 0.0)
