"""The ``wedge_values`` annotate strategy: labels for pie and donut wedges.

Placement here is decided by *measuring* the rendered label against the
room its wedge actually has, not by a threshold on the wedge's angular
span. A fixed degree threshold is the wrong shape for this problem: the
room a label needs depends on the chart's physical size, on how long the
label string is, and on where the wedge points. At the themed 7pt font,
across the cases measured in :func:`_label_fits`, the span a label needs
to sit inside runs from 14.3 degrees (``"50.0%"`` at three o'clock on a
40mm chart) to 134.2 degrees (``"1,234,567"`` at twelve o'clock on a
25mm one).

Labels that end up outside their wedge are *decorations*, like a title
or a tick label: they reach past the axes frame, and the layout grows
the figure to hold them rather than the strategy shrinking the chart to
fit them in. So nothing here touches the axis limits, and the disc is
the same size whichever anchor its labels took.

An outside label is placed by one of three leader styles (see
:func:`_outside_layout`), each exact in millimetres. Its Text and the
leader that points at it are created and held together, as an
:class:`_OutsideLabel`, so that the pass moving crowded labels apart
(``avoid_overlap``; :func:`_spread_in_columns` for elbow leaders,
:func:`_spread_on_circle` for the others) can move a label and re-route
its leader in one step.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.text import Text

from publiplots.annotate._cache import WedgeMeta, WedgeRecord
from publiplots.annotate._color import resolve_color
from publiplots.annotate._positioning import mm_to_data
from publiplots.annotate._shared import (
    ensure_renderer as _ensure_renderer,
)
from publiplots.annotate._shared import (
    format_value as _format_value,
)
from publiplots.layout.auto_layout import register_decoration
from publiplots.themes.rcparams import resolve_param

_VALID_ANCHORS = {"inside", "outside", "auto"}
_DEFAULT_ANCHOR = "auto"
_VALID_BASES = {"total", "parent"}
_DEFAULT_BASIS = "total"
_VALID_ROTATIONS = {"tangential", "radial"}
_VALID_LEADERS = ("radial", "elbow", None)

# A mid-ray this close to vertical counts as vertical, so the elbow of a
# wedge centred on twelve or six o'clock turns right. `cos(3*pi/2)` is
# -1.8e-16, not 0, and would otherwise send the six o'clock tail left.
_VERTICAL_TOLERANCE = 1e-9

# How far short of `label_spacing` a measured gap between two outside
# labels may fall and still count as kept, in mm. See `_OutsideGeometry`.
_SPACING_TOLERANCE_MM = 1e-3


class _RcLeader:
    """Default for ``leader``: read ``pp.rcParams["pie.leader"]`` at call time.

    A sentinel rather than ``None``, because ``leader=None`` is a value
    the caller may pass -- it means "no line" -- and must not be mistaken
    for "not passed".
    """

    def __repr__(self) -> str:
        return "rcParams['pie.leader']"


_LEADER_DEFAULT = _RcLeader()

# Sentinel the dispatcher hands back when the caller passed no `fmt`. A
# real format spec cannot collide with it, and it is what distinguishes
# "label the share as a percentage" (the default) from "format the value
# with this spec".
_WEDGE_DEFAULT_FMT = "__wedge_percent__"


def _meta(ax: Axes) -> WedgeMeta:
    meta = getattr(ax, "_publiplots_wedge_meta", None)
    if meta is None:
        raise ValueError(
            "annotate(kind='wedge_values') requires an axes drawn by "
            "pp.pieplot. There is no introspection fallback for a foreign "
            "ax.pie(): recovering wedges from the axes is the step that "
            "produced the annotate pairing bugs this design avoids."
        )
    if meta.chart == "sunburst":
        raise NotImplementedError(
            "annotate(kind='wedge_values') is not supported on a "
            "pp.sunburstplot axes yet; wedge labels for sunburst charts "
            "arrive in the next release. Draw the chart without them for "
            "now."
        )
    return meta


def _data_per_pixel(ax: Axes) -> float:
    """Data units spanned by one display pixel, read off the y axis.

    One scalar describes both directions only when the x and y scales
    agree. ``aspect="cell"`` makes them agree by construction: it matches
    the data limits to the axes' millimetre ratio, which is what makes the
    disc render round there.

    ``aspect="equal"`` makes them agree only once ``apply_aspect`` has
    shrunk the drawn box to a square, and matplotlib otherwise does that
    on the first draw -- after the labels are placed. Until then the box
    is still the full cell against square limits, so the y scale is the
    right one on a cell wider than tall and wrong by the cell's own
    ratio on one taller than wide. The strategy therefore calls
    ``ax.apply_aspect()`` before reading this. Measured on the frame
    ``tests/annotate/test_wedge_values.py::df`` uses (``kind=ABCD``,
    ``n=[50, 25, 15, 10]``) with ``legend=False`` and
    ``annotate={"anchor": "outside", "offset": 3.0}``: without that call
    the rim-to-label gap renders at 3.000mm on a 70x40mm cell and at
    1.714mm -- three times 40/70 -- on a 40x70mm one; with it, at
    3.000mm on both. The call changes nothing under ``aspect="cell"``,
    whose drawn box already is the whole cell.
    """
    inverse = ax.transData.inverted()
    y0 = inverse.transform((0.0, 0.0))[1]
    y1 = inverse.transform((0.0, 1.0))[1]
    return abs(y1 - y0)


def _upright(angle: float) -> float:
    """Keep derived text readable by flipping anything upside-down."""
    angle = (angle + 180.0) % 360.0 - 180.0
    if angle > 90.0:
        return angle - 180.0
    if angle < -90.0:
        return angle + 180.0
    return angle


def _rotation_for(rotation, mid_deg: float) -> float:
    """Resolve the requested rotation for a wedge at ``mid_deg``.

    Flipping upside-down text applies only to the derived orientations.
    An explicit ``rotation=120`` is the caller's choice and is passed
    through untouched.
    """
    if rotation == "tangential":
        return _upright(mid_deg - 90.0)
    if rotation == "radial":
        return _upright(mid_deg)
    return float(rotation)


def _inside_point(record, mid: float) -> Tuple[float, float, str]:
    """Anchor at the middle of the wedge's radial band."""
    cx, cy = record.patch.center
    radius = (record.r_inner + record.r_outer) / 2.0
    return cx + radius * math.cos(mid), cy + radius * math.sin(mid), "center"


def _resolve_leader(leader) -> Optional[str]:
    """Validate a ``leader`` value, reading the rcParam for the default.

    ``False`` is accepted as ``None`` -- "no line" -- both here and in the
    rcParam, since ``annotate={"leader": False}`` is the natural spelling.
    """
    if leader is _LEADER_DEFAULT:
        leader = resolve_param("pie.leader")
    if leader is False:
        leader = None
    if leader not in _VALID_LEADERS:
        raise ValueError(
            f"wedge_values: leader= must be 'radial', 'elbow', or None or "
            f"False for no line; got {leader!r}."
        )
    return leader


def check_length(value, name: str, *, owner: str) -> float:
    """Validate a length in millimetres and return it as a float.

    The one rule for every wedge length -- ``pieplot``'s ``gap`` and
    ``wedge_values``' ``offset``, ``leader_gap`` and ``elbow_length``:
    anything but a real number raises :class:`TypeError`, and so does a
    bool, which is a number to Python but never a length; a negative,
    NaN or infinite one raises :class:`ValueError`. ``owner`` prefixes
    the message with the function the caller called.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(
            f"{owner}: {name}= must be a number of millimetres; got "
            f"{type(value).__name__} {value!r}."
        )
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"{owner}: {name}= must be a finite, non-negative number of "
            f"millimetres; got {value!r}."
        )
    return value


def _resolve_avoid_overlap(value: Optional[bool]) -> bool:
    """Validate ``avoid_overlap``, reading ``pie.avoid_overlap`` for ``None``.

    Only a bool is accepted, NumPy's included: a number or a string here
    is far more likely a misplaced length than a deliberate switch.
    """
    value = resolve_param("pie.avoid_overlap", value)
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(
            f"wedge_values: avoid_overlap= must be True or False; got "
            f"{type(value).__name__} {value!r}."
        )
    return bool(value)


def _resolve_length(name: str, key: str, value: Optional[float]) -> float:
    """A non-negative finite length in mm, defaulting to ``rcParams[key]``."""
    return check_length(resolve_param(key, value), name, owner="wedge_values")


@dataclass
class _OutsideGeometry:
    """Everything outside placement needs, converted to data units once.

    Attributes
    ----------
    leader : {'radial', 'elbow'} or None
        The leader style.
    offset : float
        Rim to the leader's end -- to its elbow, for ``'elbow'`` -- or,
        with no leader, rim to the label's nearest point. Data units.
    leader_gap : float
        Leader's end to the label's nearest point. Data units.
    elbow_length : float
        Length of an elbow leader's horizontal tail. Data units.
    label_spacing : float
        Least gap between two outside labels' boxes. Data units.
    tolerance : float
        How far short of ``label_spacing`` a measured gap may fall and
        still count as kept: 0.001mm, in data units, far below anything
        visible, so rounding in the measured boxes does not trigger
        another round of de-collision.
    scale : float
        Data units per display pixel, for the label's rendered box.
    renderer : matplotlib.backend_bases.RendererBase
        Renderer the label's box is measured with.
    """

    leader: Optional[str]
    offset: float
    leader_gap: float
    elbow_length: float
    scale: float
    renderer: object
    label_spacing: float = 0.0
    tolerance: float = 0.0


@dataclass
class _OutsideLabel:
    """One outside label and the leader that points at it, kept together.

    A later pass that moves crowded labels apart must move a label and
    re-route its leader as one step; holding the pair, with the wedge and
    mid-ray they hang off, is what lets it. :func:`_apply_outside` is the
    one place that writes a layout onto both.

    Attributes
    ----------
    text : matplotlib.text.Text
        The label.
    leader : matplotlib.lines.Line2D or None
        Its leader line, or ``None`` when ``leader=None``.
    record : WedgeRecord
        The wedge it labels.
    mid : float
        The wedge's mid-ray angle, in radians.
    phi : float or None
        For ``'radial'`` and no leader, the ray the label was moved onto
        by :func:`_spread_on_circle`, in radians about the wedge's own
        centre; ``None`` while it sits on ``mid``.
    label_y : float or None
        For ``'elbow'``, the height the label was moved to by
        :func:`_spread_in_columns`, in data units; ``None`` while it sits
        level with its elbow.
    slide : float
        For a label moved onto ``phi``, how much further out along it
        :func:`_slide_clear` moved it to take it off a wedge or another
        label's leader, in data units; ``0.0`` otherwise.
    """

    text: Text
    leader: Optional[Line2D]
    record: WedgeRecord
    mid: float
    phi: Optional[float] = None
    label_y: Optional[float] = None
    slide: float = 0.0


def _half_extents(text: Text, renderer, scale: float) -> Tuple[float, float]:
    """Half the rendered label box's width and height, in data units.

    ``get_window_extent`` already folds in any rotation, so this is the
    axis-aligned box around the rotated label.
    """
    box = text.get_window_extent(renderer)
    return box.width * scale / 2.0, box.height * scale / 2.0


def _centre_distance(half_w: float, half_h: float,
                     direction: Tuple[float, float], gap: float) -> float:
    """How far along ``direction`` to centre a box so it is ``gap`` from P.

    A box of half-sizes ``(a, b)`` centred at ``P + t * d`` is at distance

        ``dist(t) = hypot(max(t|dx| - a, 0), max(t|dy| - b, 0))``

    from ``P``: the plain straight-line gap from ``P`` to the nearest
    point of the box. ``dist`` is zero while ``P`` is inside the box and
    strictly increasing once the box clears it, so ``dist(t) = gap`` has
    one solution, found here in closed form from whichever of the box's
    sides or corners is nearest ``P``:

    - its near *vertical* side, ``t|dx| - a = gap``, when that leaves
      ``P`` level with the box (``t|dy| <= b``);
    - its near *horizontal* side, ``t|dy| - b = gap``, when that leaves
      ``P`` beside it (``t|dx| <= a``);
    - otherwise its near corner, the larger root of
      ``(t|dx| - a)**2 + (t|dy| - b)**2 = gap**2``.

    A mid-ray exactly along an axis never divides by zero: the vertical
    side is skipped when ``dx == 0``, and when ``dy == 0`` it always
    applies, so the horizontal side is only reached with ``dy != 0``.
    With ``gap == 0`` the box just touches ``P``.
    """
    ax_, ay_ = abs(direction[0]), abs(direction[1])
    if ax_ > 0.0:
        t = (half_w + gap) / ax_
        if t * ay_ <= half_h:
            return t
    # Reached only with `ay_ > 0`: when it is 0 the branch above returns.
    t = (half_h + gap) / ay_
    if t * ax_ <= half_w:
        return t
    # Both separations are positive at the solution: the nearest point is
    # a corner. `direction` is a unit vector, so the quadratic's leading
    # coefficient is 1.
    p = half_w * ax_ + half_h * ay_
    q = half_w * half_w + half_h * half_h - gap * gap
    return p + math.sqrt(max(p * p - q, 0.0))


def _outside_layout(text: Text, record: WedgeRecord, mid: float,
                    geom: _OutsideGeometry):
    """Where an outside label goes, and the vertices of its leader.

    With ``d`` the unit mid-ray and ``c`` the wedge's own centre -- the
    patch centre, which already carries a dict ``pull``, so a pulled
    wedge keeps its label -- the rim point is ``c + r_outer * d``, and
    ``r_outer`` already includes a scalar ``pull``'s even explode:

    - ``'radial'``: the leader runs from the rim to ``offset`` beyond it
      along ``d``. The label's rendered box is centred on the mid-ray at
      the distance that puts its nearest point exactly ``leader_gap``
      from the leader's end, in a straight line
      (:func:`_centre_distance`), so the gap reads the same at every
      angle. The box's nearest point is on the mid-ray only at three,
      six, nine and twelve o'clock; elsewhere it is a corner beside it,
      and the box reaches back past the leader's end without coming
      nearer than ``leader_gap`` to any point of the line.
    - ``None``: no leader, and the same centred box with its nearest
      point ``offset`` from the rim point, in a straight line.
    - ``'elbow'``: a radial segment from the rim to an elbow ``offset``
      beyond it, then a horizontal tail of ``elbow_length`` toward the
      label's side -- right where ``d`` points right or straight up or
      down, left otherwise. The label is ``ha='left'``/``'right'``,
      ``va='center'``, ``leader_gap`` past the tail's end.

    A rotated label is laid out by its rotated box in every style -- the
    axis-aligned box around the rotated text, so the ink of a rotated
    label can sit a little further off than the gap. For ``'radial'`` and
    ``None`` that box is measured directly; for ``'elbow'``
    it is matplotlib's default ``rotation_mode``, which applies ``ha``
    and ``va`` to the rotated box, so its near side still sits
    ``leader_gap`` past the tail and its middle level with the tail. A
    caller-supplied ``rotation_mode="anchor"`` aligns the unrotated text
    instead and gives up that guarantee.

    A label that :func:`_spread_on_circle` or :func:`_spread_in_columns`
    moved carries the move on its :class:`_OutsideLabel` -- the ray
    ``phi`` or the height ``label_y`` -- and :func:`_apply_outside` lays
    it out with :func:`_moved_layout` instead of this. One that was not
    moved takes exactly this path, so with
    ``avoid_overlap=False``, or where nothing crowds, the layout is the
    one this function has always produced, to the bit.

    Returns
    -------
    position : tuple of float
        The label's ``(x, y)`` in data units.
    ha, va : str
        Its alignment.
    vertices : tuple of list of float, or None
        The leader's ``(xs, ys)``, or ``None`` with no leader.
    """
    cx, cy = record.patch.center
    d = (math.cos(mid), math.sin(mid))
    rim = record.r_outer
    rim_point = (cx + rim * d[0], cy + rim * d[1])
    end = rim + geom.offset
    end_point = (cx + end * d[0], cy + end * d[1])

    if geom.leader == "elbow":
        side = -1.0 if d[0] < -_VERTICAL_TOLERANCE else 1.0
        tail_x = end_point[0] + side * geom.elbow_length
        position = (tail_x + side * geom.leader_gap, end_point[1])
        ha = "left" if side > 0 else "right"
        vertices = ([rim_point[0], end_point[0], tail_x],
                    [rim_point[1], end_point[1], end_point[1]])
        return position, ha, "center", vertices

    half_w, half_h = _half_extents(text, geom.renderer, geom.scale)
    if geom.leader == "radial":
        centre = end + _centre_distance(half_w, half_h, d, geom.leader_gap)
        vertices = ([rim_point[0], end_point[0]],
                    [rim_point[1], end_point[1]])
    else:
        centre = rim + _centre_distance(half_w, half_h, d, geom.offset)
        vertices = None
    position = (cx + centre * d[0], cy + centre * d[1])
    return position, "center", "center", vertices


def _moved_layout(label: "_OutsideLabel", geom: _OutsideGeometry):
    """:func:`_outside_layout` for a label a de-collision pass moved.

    - ``'elbow'``, moved to height ``label_y``: the label keeps the
      ``x`` of an unmoved one, ``leader_gap`` past the tail's end, and
      only its height changes. The leader still starts on the rim at
      the mid-ray, and its elbow keeps its ``x``, that of the point
      ``offset`` out on the mid-ray, but rises or drops with the label
      to ``label_y``: the first leg leans by the move and the tail,
      ``elbow_length`` long, stays level. See
      :func:`_spread_in_columns` for why the elbow, not the tail, takes
      the move.
    - ``'radial'`` and no leader, moved onto the ray ``phi``: the label
      is centred on ``phi`` exactly as :func:`_outside_layout` centres
      an unmoved one on ``mid`` -- its nearest point ``leader_gap`` from
      the point ``offset`` beyond the rim on ``phi`` (``offset`` from the
      rim point on ``phi`` with no leader) -- so it sits where the label
      of a wedge centred on ``phi`` would. A radial leader still starts
      on the wedge's rim at its own mid-ray and runs straight to that
      point on ``phi``, so it leans by the move. A leaning line can pass
      nearer the box than its end does, so the box is then slid further
      out along ``phi`` (:func:`_clear_leader`) until no point of the
      line is nearer than ``leader_gap``.
    """
    text, record, mid = label.text, label.record, label.mid
    cx, cy = record.patch.center
    d_mid = (math.cos(mid), math.sin(mid))
    rim = record.r_outer
    rim_point = (cx + rim * d_mid[0], cy + rim * d_mid[1])
    end = rim + geom.offset

    if geom.leader == "elbow":
        end_point = (cx + end * d_mid[0], cy + end * d_mid[1])
        side = -1.0 if d_mid[0] < -_VERTICAL_TOLERANCE else 1.0
        tail_x = end_point[0] + side * geom.elbow_length
        y = label.label_y if label.label_y is not None else end_point[1]
        position = (tail_x + side * geom.leader_gap, y)
        ha = "left" if side > 0 else "right"
        vertices = ([rim_point[0], end_point[0], tail_x],
                    [rim_point[1], y, y])
        return position, ha, "center", vertices

    phi = label.phi if label.phi is not None else mid
    half_w, half_h = _half_extents(text, geom.renderer, geom.scale)
    position, vertices = _ray_layout(label, phi, geom, half_w, half_h)
    return position, "center", "center", vertices


def _ray_layout(label: "_OutsideLabel", phi: float, geom: _OutsideGeometry,
                half_w: float, half_h: float):
    """Centre and leader of a ``'radial'`` or unled label on the ray ``phi``.

    The arithmetic of :func:`_moved_layout`, apart so that
    :func:`_spread_on_circle` can try a label on many rays from its box
    size, measured once, without re-rendering it for each.
    """
    record, mid = label.record, label.mid
    cx, cy = record.patch.center
    rim = record.r_outer
    d = (math.cos(phi), math.sin(phi))
    if geom.leader == "radial":
        rim_point = (cx + rim * math.cos(mid), cy + rim * math.sin(mid))
        end = rim + geom.offset
        end_point = (cx + end * d[0], cy + end * d[1])
        centre = end + _centre_distance(half_w, half_h, d, geom.leader_gap)
        centre = _clear_leader(rim_point, end_point, (cx, cy), d, centre,
                               half_w, half_h, geom.leader_gap)
        vertices = ([rim_point[0], end_point[0]],
                    [rim_point[1], end_point[1]])
    else:
        centre = rim + _centre_distance(half_w, half_h, d, geom.offset)
        vertices = None
    centre += label.slide
    return (cx + centre * d[0], cy + centre * d[1]), vertices


def _slide_clear(label: _OutsideLabel, geom: _OutsideGeometry,
                 half_w: float, half_h: float, wedges, leaders) -> None:
    """Slide a moved label out along its ray until it touches nothing.

    ``wedges`` are the drawn wedges' paths and ``leaders`` the other
    labels' leader segments, all in data units. On a chart whose wedges
    all share its centre, a label moved round the circle is placed as
    the label of a wedge centred on its new ray would be, so it never
    reaches a wedge (see :func:`_spread_on_circle`) and this changes
    nothing. A wedge pulled out by a dict ``pull`` breaks that: it, and
    the leader off its rim, reach past its neighbours' rims into the
    ring their labels move round. Sets ``label.slide`` to the least
    distance, found by bisection, that clears them all by
    ``label_spacing``, leaving it 0 when nothing is that near.
    """
    from matplotlib.transforms import Bbox

    # Clear by `label_spacing`, as from another label, and never by less
    # than the measuring tolerance, so the rendered box -- whose edges
    # round a little differently from these -- does not touch either.
    margin = max(geom.label_spacing, geom.tolerance)

    def touches(slide: float) -> bool:
        label.slide = slide
        (x, y), _ = _ray_layout(label, label.phi, geom, half_w, half_h)
        box = (x - half_w - margin, y - half_h - margin,
               x + half_w + margin, y + half_h + margin)
        bbox = Bbox.from_extents(*box)
        return (any(path.intersects_bbox(bbox, filled=True)
                    for path in wedges)
                or any(_segment_hits_box(a, b, box) for a, b in leaders))

    if not touches(0.0):
        label.slide = 0.0
        return
    reach = label.record.r_outer
    lo, hi = 0.0, max(half_w, half_h)
    while touches(hi):
        if hi > 4.0 * reach:
            # Nothing to clear it out to: leave it where it was placed.
            label.slide = 0.0
            return
        lo, hi = hi, 2.0 * hi
    for _ in range(_BISECT_STEPS):
        midpoint = 0.5 * (lo + hi)
        if touches(midpoint):
            lo = midpoint
        else:
            hi = midpoint
    label.slide = hi


def _point_box_distance(p, box) -> float:
    """Straight-line distance from ``p`` to the box ``(x0, y0, x1, y1)``."""
    dx = max(box[0] - p[0], 0.0, p[0] - box[2])
    dy = max(box[1] - p[1], 0.0, p[1] - box[3])
    return math.hypot(dx, dy)


def _point_segment_distance(p, a, b) -> float:
    """Straight-line distance from ``p`` to the segment ``a``-``b``."""
    vx, vy = b[0] - a[0], b[1] - a[1]
    length2 = vx * vx + vy * vy
    if length2 == 0.0:
        return math.dist(p, a)
    t = ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / length2
    t = min(max(t, 0.0), 1.0)
    return math.dist(p, (a[0] + t * vx, a[1] + t * vy))


def _segment_hits_box(a, b, box) -> bool:
    """Does the segment ``a``-``b`` touch the box? Liang-Barsky clipping."""
    t0, t1 = 0.0, 1.0
    dx, dy = b[0] - a[0], b[1] - a[1]
    for p, q in ((-dx, a[0] - box[0]), (dx, box[2] - a[0]),
                 (-dy, a[1] - box[1]), (dy, box[3] - a[1])):
        if p == 0.0:
            if q < 0.0:
                return False
            continue
        r = q / p
        if p < 0.0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
        if t0 > t1:
            return False
    return True


def _segment_box_distance(a, b, box) -> float:
    """Straight-line distance between the segment ``a``-``b`` and a box.

    Zero when they touch. Otherwise the nearest pair of points between
    two disjoint convex shapes includes a vertex of one of them, so it
    is the least of the segment's ends to the box and the box's corners
    to the segment.
    """
    if _segment_hits_box(a, b, box):
        return 0.0
    corners = ((box[0], box[1]), (box[2], box[1]), (box[0], box[3]),
               (box[2], box[3]))
    return min([_point_box_distance(a, box), _point_box_distance(b, box)]
               + [_point_segment_distance(c, a, b) for c in corners])


# Bisection steps for `_clear_leader` and `_pair_requirement`. Each halves
# the bracket, so 60 take a bracket of a whole radius or a whole turn to
# well under 1e-15 of it -- far past anything a renderer resolves.
_BISECT_STEPS = 60


def _clear_leader(start, end_point, centre_xy, d, centre: float,
                  half_w: float, half_h: float, gap: float) -> float:
    """Slide a box out along ``d`` until the whole leader keeps ``gap``.

    The box is centred ``centre`` from ``centre_xy`` along ``d``, its
    nearest point ``gap`` from ``end_point``. On the mid-ray that is
    also its distance from every point of the leader; once the leader
    leans, a point short of its end can come nearer, most often the
    near corner of a box beside the line. Returns the least
    ``centre >= centre`` at which the segment ``start``-``end_point`` is
    at least ``gap`` from the box -- unchanged when it already is.
    Moving the box away along its own ray only takes it further from a
    line that ends behind it, so the bracket is grown and then bisected.
    """
    def clearance(c: float) -> float:
        x = centre_xy[0] + c * d[0]
        y = centre_xy[1] + c * d[1]
        return _segment_box_distance(
            start, end_point, (x - half_w, y - half_h, x + half_w, y + half_h))

    # A margin far below anything visible: the box's own placement puts
    # `end_point` at `gap` only to rounding.
    slack = 1e-9 * max(gap, half_w, half_h, 1e-12)
    if clearance(centre) >= gap - slack:
        return centre
    lo, step = centre, max(gap, half_w, half_h, 1e-12)
    hi = centre + step
    while clearance(hi) < gap:
        lo, step = hi, step * 2.0
        hi = hi + step
    for _ in range(_BISECT_STEPS):
        midpoint = 0.5 * (lo + hi)
        if clearance(midpoint) >= gap:
            hi = midpoint
        else:
            lo = midpoint
    return hi


def _apply_outside(label: _OutsideLabel, geom: _OutsideGeometry) -> None:
    """Write :func:`_outside_layout`'s result onto the label and its leader.

    The leader must already exist when the style draws one; its vertices
    are replaced, so a later pass can call this again after moving the
    label's ray or height and the pair stays consistent. A label with
    neither ``phi`` nor ``label_y`` set is laid out by
    :func:`_outside_layout`, one with either by :func:`_moved_layout`.
    """
    if label.phi is None and label.label_y is None:
        position, ha, va, vertices = _outside_layout(
            label.text, label.record, label.mid, geom)
    else:
        position, ha, va, vertices = _moved_layout(label, geom)
    label.text.set_ha(ha)
    label.text.set_va(va)
    label.text.set_position(position)
    if label.leader is not None and vertices is not None:
        label.leader.set_data(*vertices)


def _drawn_inset(record) -> float:
    """The perpendicular inset of the drawn wedge's straight edges.

    ``gap / 2`` plus any even explode, in data units, as capped for a
    thin wedge so it keeps half its rim: read off the patch, so it is
    what was drawn. A full circle has no edges to inset, and a patch
    that is not a :class:`~publiplots.plot.wedge.draw.GappedWedge` has
    none either.
    """
    get_inset = getattr(record.patch, "get_drawn_inset", None)
    if get_inset is None:
        return 0.0
    return max(float(get_inset()), 0.0)


def _drawn_inner_radius(record, inset: float, half_span: float) -> float:
    """Where the drawn wedge begins along its mid-ray, from its centre.

    ``r_inner`` while the inset leaves an inner arc; once the offset
    edges meet first, the apex where they do, ``inset / sin(half_span)``
    out -- the same rule :class:`~publiplots.plot.wedge.draw.GappedWedge`
    draws by.
    """
    # Lazy import to avoid a cycle: importing the plot package pulls in
    # every plotter, and they import publiplots.annotate.
    from publiplots.plot.wedge.draw import arc_survives_inset

    if inset <= 0.0:
        return record.r_inner
    apex = inset / math.sin(half_span)
    if half_span < math.pi / 2.0:
        return max(record.r_inner, apex)
    if arc_survives_inset(record.r_inner, inset):
        return record.r_inner
    return apex


def _label_fits(text: Text, record, renderer, scale: float, mid: float) -> bool:
    """Does the rendered label fit inside its own wedge, as drawn?

    The label is an axis-aligned rectangle once rendered (``get_window_extent``
    already folds in any rotation), so its reach along a unit direction
    ``d`` is the rectangle's support function, ``|w/2 * d.x| + |h/2 * d.y|``.
    Two directions matter: tangential, where the wedge's two edge rays
    close in, and radial, where the ring's inner and outer radii do.

    The wedge is the one drawn, not the one laid out. Its radii are the
    record's, which already include a scalar ``pull``'s even explode, and
    each straight edge sits ``inset`` inside its ray -- ``gap / 2`` plus
    that explode, as capped for a thin wedge, read off the patch. At the
    label radius ``r`` the perpendicular distance from the mid-ray to
    either inset edge is therefore ``r * sin(span / 2) - inset``; and
    where the inset closes the wedge in an apex before it reaches
    ``r_inner``, the wedge begins at the apex, so that is where the
    radial room starts. With no gap
    and no ``pull`` both reduce to the un-inset wedge. Past a half-span
    of 90 degrees the wedge stops being convex about its mid-ray and the
    edges no longer bound the label at all, so the disc itself is the
    limit.

    The tangential bound is **approximate, and its error is not small**.
    It resolves the label's reach along the mid-ray's tangent, whereas
    the half-planes that actually contain the wedge have normals
    perpendicular to its two *edge* rays, which the tangent only matches
    as the span goes to zero. In most directions that makes it
    optimistic: it accepts a label with a corner outside its wedge, by an
    amount that grows with the label's width and shrinks as the chart
    gets bigger. Swept over every span it accepts (0.5-degree steps) at
    every whole-degree direction, for a pie with no hole at the default
    rotation, the worst overhang is 0.63mm for ``"50.0%"`` on a 40x40mm
    chart, 1.35mm for ``"1,234,567"`` on that chart, and 1.98mm for
    ``"1,234,567"`` on a 25x25mm one, at a 64-degree span on a mid-ray
    22 degrees off horizontal (22, 158, 202 or 338 degrees, measured
    anticlockwise from three o'clock). That last is the largest over
    ``"5.0%"``, ``"50.0%"``, ``"1,234.57"`` and ``"1,234,567"`` on 25,
    40 and 80mm charts. At three o'clock exactly, the same label on the same chart
    is accepted from a 23.0-degree span and only fits from 136.8; and
    ``"50.0%"`` on 40mm is accepted from 14.3 against a geometric 22.3.
    Only on mid-rays within about 30 degrees of vertical -- 59 to 121
    degrees and 239 to 301 for that label on 25mm, so around twelve and
    six o'clock -- does a long label err the other way: ``"1,234,567"``
    on 25mm is rejected at both twelve and six o'clock below 134.2
    degrees although it fits from 98.0. So an ``auto`` label left inside
    can visibly cross its wedge's edge -- by up to about two millimetres
    with long labels on a small chart -- and ``anchor="outside"`` or a
    shorter ``fmt`` is the remedy there. Those figures are for a wedge
    with no gap and no ``pull``; the inset moves the tangential bound in
    by exactly the inset, so a gap or an even explode adds nothing to
    the overhang past the drawn edges.

    The span a label needs is *not* a single number. Against
    this test at the themed 7pt font, ``"50.0%"`` on a 40x40mm chart
    needs 43.2 degrees at twelve o'clock and 14.3 at three o'clock, where
    the label lies along the radius rather than across it; the same
    label at twelve o'clock needs 72.2 degrees on a 25mm chart and 21.2
    on an 80mm one; and ``"1,234,567"`` at twelve o'clock on that 25mm
    chart needs 134.2.
    That is the range a single constant would have had to stand for,
    which is why this measures instead.
    """
    box = text.get_window_extent(renderer)
    half_w = box.width * scale / 2.0
    half_h = box.height * scale / 2.0

    tangential = (-math.sin(mid), math.cos(mid))
    radial = (math.cos(mid), math.sin(mid))
    reach_t = abs(half_w * tangential[0]) + abs(half_h * tangential[1])
    reach_r = abs(half_w * radial[0]) + abs(half_h * radial[1])

    r_label = (record.r_inner + record.r_outer) / 2.0
    half_span = math.radians(record.theta1 - record.theta0) / 2.0
    inset = _drawn_inset(record)
    if half_span >= math.pi / 2.0:
        room_t = record.r_outer
    else:
        room_t = r_label * math.sin(half_span) - inset
    if reach_t > room_t:
        return False
    if 2.0 * reach_r > record.r_outer - record.r_inner:
        return False
    inner = _drawn_inner_radius(record, inset, half_span)
    # Only an apex past r_inner tightens the radial room; otherwise the
    # check above already is the whole band.
    return inner <= record.r_inner or reach_r <= r_label - inner


def _block_value(total: float, count: int, lower: float,
                 upper: float) -> float:
    """The value a pooled block takes: its mean, held inside its bounds.

    That is the minimiser of ``sum((z - w_i)**2)`` over ``lower <= z <=
    upper``. Bounds that cross leave no feasible value; the block then
    splits the difference, which keeps the solver total and
    deterministic on a chart too crowded to satisfy.
    """
    if lower > upper:
        return 0.5 * (lower + upper)
    return min(max(total / count, lower), upper)


def _pav_with_gaps(desired, gaps, lower=None, upper=None):
    """The nearest ordered positions that keep at least ``gaps`` apart.

    Solves, exactly,

        minimise    ``sum((y_i - desired_i)**2)``
        subject to  ``y_{i+1} - y_i >= gaps_i``    for each ``i``,
                    ``lower_i <= y_i <= upper_i``

    for positions already in order along a line -- labels' heights in a
    column, or their angles round a circle cut open at one point.

    **Formulation.** With ``C_i = gaps_0 + ... + gaps_{i-1}`` (``C_0 =
    0``) and ``z_i = y_i - C_i``, each gap constraint becomes ``z_{i+1}
    >= z_i`` and the objective ``sum((z_i - w_i)**2)`` with ``w_i =
    desired_i - C_i``: an isotonic regression of ``w``, with each bound
    shifted by ``C_i`` too.

    **Why pool-adjacent-violators gives the exact optimum.** The
    objective is a sum of one convex function of each ``z_i`` -- a
    square, plus the indicator of its interval -- and the constraints
    are a single chain. For that problem PAV is exact (Best, Chakravarti
    and Ubhaya, 2000): scanning left to right, each new element starts
    as its own block at its own minimiser; while the previous block's
    value exceeds the new one's, the two are pooled into a block at the
    minimiser of their combined objective, here the mean of their ``w``
    held inside the tightest of their bounds (:func:`_block_value`).
    The minimiser of a sum of convex functions lies between the
    minimisers of its terms, so pooling never needs undoing, and what is
    left is non-decreasing and satisfies the KKT conditions: inside a
    block, where every chain constraint is tight, each prefix of the
    block has values at or above the block's -- the residual it hands
    on, the multiplier of the constraint after it, is non-negative --
    and between blocks no constraint binds. No iteration and no step
    size are involved, so the answer is the optimum, not an
    approximation of it.

    **Nothing that does not conflict moves.** When ``desired`` already
    satisfies every constraint, ``w`` is already non-decreasing, no
    block is ever pooled, and every block is a single element. A
    single, unclipped element returns its ``desired`` value itself
    rather than ``w_i + C_i``, which could differ from it in the last
    bit; so the answer is ``desired``, exactly. Pooling waits for a
    violation larger than ``1e-12`` of the values' scale, so a pair
    exactly at its gap is never pooled by rounding.

    Parameters
    ----------
    desired : sequence of float
        Where each position would be with nothing in the way, in order:
        ``desired_i <= desired_{i+1}`` is not required, but the order of
        the answer is the order of the input.
    gaps : sequence of float
        ``len(desired) - 1`` least separations, each ``>= 0``.
    lower, upper : sequence of float, optional
        Per-position bounds; ``None`` for none. A bound must admit its
        own ``desired`` value for "nothing that does not conflict
        moves" to hold.

    Returns
    -------
    positions : list of float
        The optimal ``y``.
    moved : list of bool
        Whether each position differs from its ``desired`` value.
    """
    n = len(desired)
    if len(gaps) != max(n - 1, 0):
        raise ValueError(
            f"_pav_with_gaps: {n} positions need {max(n - 1, 0)} gaps; got "
            f"{len(gaps)}"
        )
    offsets = [0.0] * n
    for k in range(1, n):
        offsets[k] = offsets[k - 1] + gaps[k - 1]
    w = [desired[i] - offsets[i] for i in range(n)]
    lo = [(-math.inf if lower is None else lower[i] - offsets[i])
          for i in range(n)]
    hi = [(math.inf if upper is None else upper[i] - offsets[i])
          for i in range(n)]
    eps = 1e-12 * max([abs(value) for value in w] + [1.0])

    # Each block: [first, count, total, lower, upper, value].
    blocks: List[list] = []
    for i in range(n):
        block = [i, 1, w[i], lo[i], hi[i], 0.0]
        block[5] = _block_value(w[i], 1, lo[i], hi[i])
        while blocks and blocks[-1][5] > block[5] + eps:
            prev = blocks.pop()
            total = prev[2] + block[2]
            count = prev[1] + block[1]
            low, high = max(prev[3], block[3]), min(prev[4], block[4])
            block = [prev[0], count, total, low, high,
                     _block_value(total, count, low, high)]
        blocks.append(block)

    positions: List[float] = []
    moved: List[bool] = []
    for first, count, _, _, _, value in blocks:
        for i in range(first, first + count):
            if count == 1 and value == w[i]:
                positions.append(desired[i])
                moved.append(False)
            else:
                position = value + offsets[i]
                positions.append(position)
                moved.append(position != desired[i])
    return positions, moved


# Most rounds of solve-then-measure a de-collision pass makes. The column
# pass's separations are exact, so it needs a second round only when two
# labels that are not neighbours in their column overlap or a leader runs
# through a label; the circle pass's are estimated at each pair's middle,
# so it re-measures and re-solves -- but only pairs that were given their
# full separation, so most charts without the room stop at once.
# Measured over the 3,870-chart review sweep: the circle pass needed six
# to eight rounds on 34 charts and reached this cap on 4 (the 13-wedge
# tail at 25mm with no leader), each of which then fell back to elbow
# labels cleanly; the column pass reached it on 11, every one beside a
# dict pull, where a pulled wedge's leg can be in a label's way however
# often the column is re-solved. The cap bounds that cost.
_MAX_SPREAD_ROUNDS = 8


def _data_box(text: Text, renderer, ax: Axes):
    """The label's rendered box ``(x0, y0, x1, y1)``, in data units."""
    box = text.get_window_extent(renderer)
    (x0, y0), (x1, y1) = ax.transData.inverted().transform(
        [(box.x0, box.y0), (box.x1, box.y1)])
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def _box_gap(a, b) -> float:
    """Straight-line gap between two boxes; minus the overlap depth if
    they overlap."""
    gx = max(a[0] - b[2], b[0] - a[2])
    gy = max(a[1] - b[3], b[1] - a[3])
    if gx < 0.0 and gy < 0.0:
        return max(gx, gy)
    return math.hypot(max(gx, 0.0), max(gy, 0.0))


def _vertical_need(a, b, spacing: float) -> float:
    """Least centre-to-centre height between boxes ``a`` and ``b`` that
    keeps them ``spacing`` apart, given how far apart they already are
    across.

    Boxes whose sides are ``spacing`` or more apart across never conflict
    and need no height between them at all; boxes that overlap across
    need their half-heights plus ``spacing``; between the two, the
    corner-to-corner gap is ``spacing`` once the height is
    ``sqrt(spacing**2 - gx**2)`` more than the half-heights.
    """
    gx = max(a[0] - b[2], b[0] - a[2])
    if gx >= spacing:
        return 0.0
    extra = spacing if gx <= 0.0 else math.sqrt(spacing * spacing - gx * gx)
    return 0.5 * ((a[3] - a[1]) + (b[3] - b[1])) + extra


def _disc_polygons(records) -> List:
    """Every drawn wedge's outline, as closed polylines in data units.

    Curves are flattened by matplotlib's own path-to-polygon conversion,
    as finely as it renders them.
    """
    polygons = []
    for record in records:
        patch = record.patch
        path = patch.get_patch_transform().transform_path(patch.get_path())
        polygons.extend(path.to_polygons(closed_only=False))
    return polygons


def _strip_extent(polyline, x0: float, x1: float):
    """Lowest and highest point of ``polyline`` between ``x0`` and ``x1``.

    For a wedge's outline, the span of its area inside that vertical
    strip: the highest point of the area inside a strip is on its
    outline inside the strip -- where the strip's own sides cross the
    area, the highest point of the crossing is on the outline too -- so
    this clips each edge to the strip and reads its ends. That is exact
    for a convex wedge and covers a reflex one's notch, which the rest
    of the disc fills anyway. ``None`` when nothing of it is in the
    strip.
    """
    top, bottom = -math.inf, math.inf
    for (px, py), (qx, qy) in zip(polyline[:-1], polyline[1:]):
        lo, hi = (px, qx) if px <= qx else (qx, px)
        if hi < x0 or lo > x1:
            continue
        if qx == px:
            ys = (py, qy)
        else:
            ys = tuple(py + (qy - py) * (x - px) / (qx - px)
                       for x in (max(lo, x0), min(hi, x1)))
        top = max(top, *ys)
        bottom = min(bottom, *ys)
    if top == -math.inf:
        return None
    return bottom, top


def _column_bound(obstacles, x0: float, x1: float, y0: float, y1: float,
                  desired: float, spacing: float,
                  escape: bool = False) -> Tuple[float, float]:
    """Floor and ceiling on the height of one box in a column.

    The box spans ``x0``-``x1`` and ``y0``-``y1`` unmoved, centred on
    ``desired``. Widened by ``spacing`` on every side it spans a strip;
    each obstacle's span of heights in that strip (:func:`_strip_extent`)
    is what the widened box, moved straight down or up, would run into,
    exactly for a convex obstacle: a wedge of up to half the disc, or a
    straight leg. Moving down, the first contact is with the highest
    span wholly below the widened box, so the floor is that span's top
    plus the box's half-height plus ``spacing``; the ceiling is the
    mirror image. A span the widened box already reaches into holds it
    where it is on the side the span mostly lies. Each bound is relaxed
    to ``desired`` when that is nearer, so it never moves a box.

    With ``escape``, for the wedges themselves, a box that already
    *overlaps* one -- not merely within ``spacing`` of it -- is not held
    there but sent clear of it, away from the side the wedge mostly
    lies. Unmoved, an elbow label's box is centred level with its elbow,
    ``offset`` beyond the rim, so only a box taller than about twice
    that does: a label rotated across its tail near twelve or six
    o'clock.
    """
    half = 0.5 * (y1 - y0)
    floor, ceiling = -math.inf, math.inf
    for polyline in obstacles:
        extent = _strip_extent(polyline, x0 - spacing, x1 + spacing)
        if extent is None:
            continue
        bottom, top = extent
        inside = _strip_extent(polyline, x0, x1) if escape else None
        if inside is not None and inside[0] < y1 and inside[1] > y0:
            if bottom + top <= 2.0 * desired:
                floor = max(floor, top + half + spacing)
            else:
                ceiling = min(ceiling, bottom - half - spacing)
            continue
        if top <= y0 - spacing:
            floor = max(floor, min(desired, top + half + spacing))
        elif bottom >= y1 + spacing:
            ceiling = min(ceiling, max(desired, bottom - half - spacing))
        elif bottom + top <= 2.0 * desired:
            floor = max(floor, desired)
        else:
            ceiling = min(ceiling, desired)
    return floor, ceiling


def _column_bounds(label: _OutsideLabel, box, desired: float, obstacles,
                   spacing: float):
    """How far an elbow label may move down and up its column.

    ``obstacles`` are two lists of polylines in data units: every
    wedge's outline, and every other outside label's radial leg, rim to
    elbow, as unmoved. A box that overlaps a wedge is sent clear of it
    (``escape`` in :func:`_column_bound`).
    The label's box and its tail's end -- the point the tail reaches,
    ``leader_gap`` beside the box and nearer the chart -- each get a
    floor and a ceiling (:func:`_column_bound`), so neither is moved to
    within ``spacing`` of the disc or of another label's leg; and the
    label is not moved toward the chart past where its own leg, leaning
    to follow it, would point into its wedge (see
    :func:`_spread_in_columns`).

    Returns
    -------
    floor, ceiling : float
        Every bound.
    facing : tuple of float
        Only the bound on the side facing the chart -- the floor for a
        label on the upper half of its wedge's circle, the ceiling for
        one on the lower -- with the other side unbounded. A column
        bounded on both sides can have no solution at all: a wedge
        pulled out by a dict ``pull``, or its leg, crossing a column
        above a crowded run of labels boxes them in. Bounded on the side
        facing the chart only, it always has one, since every label can
        still move away from the chart.
    """
    x0, y0, x1, y1 = box
    xs, ys = label.leader.get_xdata(), label.leader.get_ydata()
    tail_x = xs[-1]
    sin, cos = math.sin(label.mid), math.cos(label.mid)
    upward = sin >= 0.0
    wedges, legs = obstacles
    of_wedges = _column_bound(wedges, x0, x1, y0, y1, desired, spacing,
                              escape=True)
    of_legs = _column_bound(legs, x0, x1, y0, y1, desired, spacing)
    of_tail = _column_bound(wedges + legs, tail_x, tail_x, desired, desired,
                            desired, spacing)
    floor = max(of_wedges[0], of_legs[0], of_tail[0])
    ceiling = min(of_wedges[1], of_legs[1], of_tail[1])
    # The leg from the rim to the moved elbow must not point into the
    # wedge: its direction (offset * cos, y - y_rim) against the mid-ray.
    offset = math.hypot(xs[1] - xs[0], ys[1] - ys[0])
    if abs(sin) > _VERTICAL_TOLERANCE:
        leg = ys[0] - offset * cos * cos / sin
        if upward:
            floor = max(floor, min(desired, leg))
        else:
            ceiling = min(ceiling, max(desired, leg))
    facing = (floor, math.inf) if upward else (-math.inf, ceiling)
    return floor, ceiling, facing


def _leaders_through_boxes(column: List[_OutsideLabel], boxes,
                           spacing: float, tol: float):
    """Leaders that come within ``spacing`` of another label in the column.

    Every moved label's leader against every other box, and every
    leader against every moved box. A box widened by ``spacing`` spans a
    vertical strip, and a leader point in that strip is inside the
    widened box exactly when its height is, so the span of heights the
    leader covers in the strip (:func:`_strip_extent`) says whether it
    is in the way and by how much.

    Returns
    -------
    list of tuple
        ``(owner, j, lacking, fixed)``: the leader of ``column[owner]``
        is ``lacking`` short of clearing ``column[j]`` by ``spacing``,
        and ``fixed`` is whether the part in the way is its rim point.
    """
    found = []
    for owner, label in enumerate(column):
        points = list(zip(label.leader.get_xdata(), label.leader.get_ydata()))
        rim_x, rim_y = points[0]
        for j, box in enumerate(boxes):
            if j == owner or (label.label_y is None
                              and column[j].label_y is None):
                continue
            x0, y0, x1, y1 = box
            extent = _strip_extent(points, x0 - spacing, x1 + spacing)
            if extent is None:
                continue
            low, high = extent
            if low >= y1 + spacing - tol or high <= y0 - spacing + tol:
                continue
            rim_in = x0 - spacing <= rim_x <= x1 + spacing
            if j > owner:
                found.append((owner, j, high + spacing - y0,
                              rim_in and abs(high - rim_y) <= tol))
            else:
                found.append((owner, j, y1 + spacing - low,
                              rim_in and abs(low - rim_y) <= tol))
    return found


def _spread_in_columns(labels: List[_OutsideLabel], geom: _OutsideGeometry,
                       ax: Axes, records) -> int:
    """Move crowded ``'elbow'`` labels apart, up and down their column.

    Each side of the chart -- the side its elbow turns to -- is solved on
    its own: a label on the right starts ``elbow_length + leader_gap``
    right of the chart's centre and one on the left as far left of it,
    so no label on one side can reach one on the other.

    On a side, the labels are taken in order of the height they would
    have unmoved, ``desired_i``, and moved to the heights ``y_i`` that
    minimise ``sum((y_i - desired_i)**2)`` subject to keeping that order
    and keeping neighbours apart -- solved exactly by
    :func:`_pav_with_gaps`. A label's ``x`` never changes, so how far
    apart two boxes are across is fixed, and the height two neighbours
    need between their centres is exact (:func:`_vertical_need`): their
    half-heights plus ``label_spacing`` when they overlap across, less
    when they are close across, nothing when they are ``label_spacing``
    or more apart across. That last case is why boxes are not simply
    stacked: two labels at the same height but a column apart do not
    conflict, and must not be moved.

    A label is also kept off the disc, and off the other labels' radial
    legs, by a floor and a ceiling on its height
    (:func:`_column_bounds`), which the solver honours exactly. Without
    them a column of labels near twelve or six o'clock, pushed apart,
    would push its outer labels into the chart; and a wedge pulled out
    by a dict ``pull``, and its leg, reach past their neighbours'
    elbows into their column. If the side's labels cannot all fit
    between their floors and ceilings, the side is solved again with
    only the bound facing the chart, which always has a solution; a
    label may then be moved onto the pulled-out wedge or leg beyond it.

    **Why the elbow takes the move, not the tail.** The leader of a
    moved label runs from the rim at the mid-ray to an elbow at the
    unmoved elbow's ``x`` and the label's new height, then level along
    the tail to the label. The alternative keeps the elbow on the
    mid-ray and slopes the tail up or down to the label instead. Both
    keep the leg's start on the wedge, and over the 1,152 charts in the
    sweep below without a dict ``pull`` neither crosses one leader with
    another; but a sloping tail is only ``elbow_length`` long, so a
    label moved several millimetres gets a steep, nearly vertical
    stroke right beside its neighbours' labels, and 56 of those tails
    ran through a neighbour's box, against none with the elbow moved.
    A level tail also keeps the labels reading as one aligned stack,
    each struck through at mid-height by its own tail, which is what an
    elbow leader is for. The sweep: the 40/30/20/5/3/1/0.5/0.5 and
    52.2/41.8/6/4 pies on 25, 40 and 80mm cells, every start angle in
    15-degree steps, both directions, as a pie, a ``donut=0.55``, with
    ``pull=0.05``, with ``gap=1``, and with a dict ``pull``.

    A leaning leg must not lean into its own wedge, so a label is not
    moved toward the chart past where the leg would point inward from
    the rim (``(y - y_rim) * sin(mid) >= -offset * cos(mid)**2``, from
    the leg's direction against the mid-ray); near three and nine
    o'clock that allows any move, near twelve and six almost none, where
    the floor above already holds the label anyway.

    The separations of neighbours are exact, but they do not bound two
    labels that are not neighbours in the column; when such a pair
    still conflicts after a solve, the separations between them are
    widened by what it lacks, shared out, and the side re-solved, at
    most :data:`_MAX_SPREAD_ROUNDS` rounds.

    **No leader through another label.** A moved label's leg leans from
    its rim point to its new height, and can then pass through a label
    beside it in the column that its unmoved leg cleared; and a moved
    box can land on an unmoved label's leader. After each solve every
    such crossing is found on the rendered boxes
    (:func:`_leaders_through_boxes`) and turned into a constraint: where
    the part of the leader in the way is its rim point, which no move
    shifts, the label in the way gets a floor or a ceiling just clear of
    it; where it is the elbow or tail, which move with their own label,
    the two labels are held further apart in the column. A constraint
    that would leave the column with no solution is not added, and a
    crossing between two labels neither of which moved is left as it
    was drawn: on rotated labels, and beside a dict ``pull``, some
    leaders still run through another label, never more than unmoved.

    Returns
    -------
    rounds : int
        The most rounds any side needed; 0 when no side had two labels.
    complete : bool
        Whether every side was separated and cleared in full.
    """
    spacing, tol = geom.label_spacing, geom.tolerance
    polygons = _disc_polygons(records)
    legs = {id(label): list(zip(label.leader.get_xdata()[:2],
                                label.leader.get_ydata()[:2]))
            for label in labels}
    sides = {1.0: [], -1.0: []}
    for label in labels:
        side = -1.0 if math.cos(label.mid) < -_VERTICAL_TOLERANCE else 1.0
        sides[side].append(label)

    rounds_needed = 0
    complete = True
    for side_labels in sides.values():
        if len(side_labels) < 2:
            continue
        boxes = {id(label): _data_box(label.text, geom.renderer, ax)
                 for label in side_labels}
        column = sorted(side_labels,
                        key=lambda label: label.text.get_position()[1])
        n = len(column)
        desired = [label.text.get_position()[1] for label in column]
        unmoved = [boxes[id(label)] for label in column]
        gaps = [_vertical_need(unmoved[k], unmoved[k + 1], spacing)
                for k in range(n - 1)]
        lower: List[float] = []
        upper: List[float] = []
        facing: List[Tuple[float, float]] = []
        for k, label in enumerate(column):
            obstacles = (polygons, [legs[id(other)] for other in labels
                                    if other is not label])
            low, high, toward = _column_bounds(label, unmoved[k], desired[k],
                                               obstacles, spacing)
            lower.append(low)
            upper.append(high)
            facing.append(toward)
        if _chain_window(gaps, lower, upper, tol) is not None:
            lower = [bound[0] for bound in facing]
            upper = [bound[1] for bound in facing]
            complete = False

        for rounds in range(1, _MAX_SPREAD_ROUNDS + 1):
            heights, moved = _pav_with_gaps(desired, gaps, lower, upper)
            for label, y, was_moved in zip(column, heights, moved):
                # Only a label whose move changed is laid out again: one
                # left where it was keeps the very floats it was placed
                # with, since re-measuring its box at another position
                # can round differently in the last bit.
                y = y if was_moved else None
                if y != label.label_y:
                    label.label_y = y
                    _apply_outside(label, geom)
            placed = [_data_box(label.text, geom.renderer, ax)
                      for label in column]
            conflicts = [(a, b, _box_gap(placed[a], placed[b]))
                         for a in range(n) for b in range(a + 1, n)
                         if _box_gap(placed[a], placed[b]) < spacing - tol]
            crossings = _leaders_through_boxes(column, placed, spacing, tol)
            if not (conflicts or crossings) or rounds == _MAX_SPREAD_ROUNDS:
                break
            for a, b, _ in conflicts:
                lacking = (_vertical_need(placed[a], placed[b], spacing)
                           - (heights[b] - heights[a]))
                share = max(lacking, tol) / (b - a)
                for k in range(a, b):
                    gaps[k] = max(gaps[k], heights[k + 1] - heights[k]) + share
            changed = bool(conflicts)
            for owner, j, lacking, fixed in crossings:
                # A leader's fixed end, its rim point, cannot be moved
                # out of the way: the label in its way gets a floor or a
                # ceiling. Any other part moves with its own label, so
                # the two are held further apart instead. Either is
                # kept only if the column can still be solved.
                trial_gaps, trial_lower = list(gaps), list(lower)
                trial_upper = list(upper)
                if fixed and j > owner:
                    trial_lower[j] = max(trial_lower[j], heights[j] + lacking)
                elif fixed:
                    trial_upper[j] = min(trial_upper[j], heights[j] - lacking)
                else:
                    a, b = min(owner, j), max(owner, j)
                    for k in range(a, b):
                        trial_gaps[k] = (max(trial_gaps[k],
                                             heights[k + 1] - heights[k])
                                         + lacking / (b - a))
                if _chain_window(trial_gaps, trial_lower, trial_upper) is None:
                    gaps, lower, upper = trial_gaps, trial_lower, trial_upper
                    changed = True
            if not changed:
                # Nothing more can be added without leaving the column
                # unsolvable: what is left stays.
                break
        complete = complete and not conflicts and not crossings
        rounds_needed = max(rounds_needed, rounds)
    return rounds_needed, complete


def _chain_window(gaps, lower, upper, tol: float = 0.0):
    """Is the chain feasible, and if not, which stretch of it is not?

    Gaps and bounds on positions in order: ``y_{k+1} - y_k >= gaps_k``
    and ``lower_k <= y_k <= upper_k``. Pushing each position as low as
    its bound and its predecessor allow gives the least feasible value
    of every one, so the chain is feasible exactly when none of those
    exceeds its upper bound. When one does, at ``j``, the stretch that
    cannot fit runs from the last position before it that sat on its own
    lower bound, ``i``: ``lower_i`` plus the gaps from ``i`` to ``j`` is
    more than ``upper_j``.

    Returns
    -------
    None when feasible, else ``(i, j)``.
    """
    y = lower[0]
    start = 0
    if y > upper[0] + tol:
        return 0, 0
    for k in range(1, len(lower)):
        pushed = y + gaps[k - 1]
        if lower[k] >= pushed:
            y, start = lower[k], k
        else:
            y = pushed
        if y > upper[k] + tol:
            return start, k
    return None


def _relax_to_fit(base, extra, lower_of, upper_of, n: int):
    """The gaps nearest to ``base + extra`` that the bounds allow.

    Pair ``k`` gets ``base_k + lam_k * extra_k`` with ``0 <= lam_k <=
    1``: ``base_k`` is what it already has, so ``lam = 0`` is always
    feasible, and ``extra_k`` what it lacks. ``lower_of(lam)`` and
    ``upper_of(lam)`` give the bounds, which may depend on ``lam`` too
    (the cut pair of the circle is held apart through them). Every
    ``lam_k`` is raised together as far as the chain stays feasible
    (bisection, :func:`_chain_window`); the stretch that then cannot fit
    has its ``lam`` fixed there, and the rest are raised again, until
    every ``lam`` is fixed or 1. So each crowded stretch gets as large a
    share of what it lacks as it has room for, and one stretch's lack of
    room costs no other stretch anything: the max-min fair relaxation.

    Returns
    -------
    lam : list of float
        One per pair, ``n`` of them.
    """
    lam = [1.0 if extra[k] > 0.0 else 0.0 for k in range(n)]
    free = {k for k in range(n) if extra[k] > 0.0}

    def window(level: float):
        trial = [level if k in free else lam[k] for k in range(n)]
        gaps = [base[k] + trial[k] * extra[k] for k in range(n)]
        return _chain_window(gaps, lower_of(trial), upper_of(trial))

    level = 0.0
    for k in free:
        lam[k] = 0.0
    while free:
        if window(1.0) is None:
            for k in free:
                lam[k] = 1.0
            break
        lo, hi = level, 1.0
        for _ in range(_BISECT_STEPS):
            midpoint = 0.5 * (lo + hi)
            if window(midpoint) is None:
                lo = midpoint
            else:
                hi = midpoint
        level = lo
        i, j = window(hi)
        # The pairs inside the stretch, and the cut pair (index n - 1)
        # when the stretch reaches either end of the chain.
        stuck = {k for k in range(i, j) if k in free}
        if (i == 0 or j == n - 1) and n - 1 in free:
            stuck.add(n - 1)
        if not stuck:
            stuck = set(free)
        for k in stuck:
            lam[k] = level
        free -= stuck
        for k in free:
            lam[k] = level
    return lam


def _spread_on_circle(labels: List[_OutsideLabel], geom: _OutsideGeometry,
                      ax: Axes, records) -> Tuple[int, bool]:
    """Move crowded ``'radial'`` and unled labels apart, round the chart.

    Each label is centred on a ray. The pass lets label ``i`` leave its
    wedge's mid-ray ``mid_i`` for a ray ``phi_i``, keeping the labels'
    order round the circle, and chooses the ``phi`` that minimise
    ``sum((phi_i - mid_i)**2)`` subject to neighbouring labels' boxes
    keeping ``label_spacing`` apart and to **every label staying within
    ``beta_i`` of its own mid-ray**. On its new ray a label is placed
    exactly as an unmoved one is on its mid-ray (:func:`_moved_layout`).

    **The bound.** A radial leader runs from the rim point on ``mid``, at
    radius ``R`` about the wedge's own centre, to the point at radius
    ``R + offset`` on ``phi``. That segment stays outside the wedge's
    circle exactly when it leaves the rim point outward or along the
    tangent, which is when ``cos(phi - mid) >= R / (R + offset)``; past
    that it cuts back across the rim and lies over the chart. So

        ``beta_i = acos(R_i / (R_i + offset))``,

    with ``R_i`` the wedge's drawn outer radius, an even explode
    included, about its own (dict-``pull``ed) centre. It is about 25
    degrees on a 40mm chart at the default 2mm ``offset``, 31 on a
    25mm one (``R`` = 11.90mm) and 18 on an 80mm one, and 0 for ``offset=0``, where
    nothing can move. An unled label takes the same bound with ``offset``
    as its gap from the rim: the straight line from its wedge's rim
    point to the point ``offset`` beyond the rim on its ray then does not
    cross the chart either, so the label still sits beside the wedge it
    names rather than over another one.

    **Cutting the circle.** The labels are taken in order of ``mid``
    round the circle and the circle is cut at its largest free angular
    gap -- the largest angle between neighbours' mid-rays once each
    label's own angular half-width is taken off both sides. That leaves
    a chain, solved by :func:`_pav_with_gaps` with each neighbouring
    pair's least angular separation as its gap and ``mid_i +- beta_i``
    as bounds. The pair the circle is cut between is held apart through
    those bounds: the slack it has beyond its own least separation is
    split between its two labels' bounds, half each, so the chain can
    never close up across the cut. That split is the one approximation
    in the solve: a cyclic solver could share it unevenly.

    **The separations.** A pair's least separation depends on where on
    the circle it sits, so it is found by bisection on the boxes
    themselves, the pair's two rays an equal angle either side of their
    current middle (:func:`_pair_requirement`). A pair that does not
    conflict where it stands is given no more than its present
    separation, so nothing that does not conflict moves.

    **When there is not room.** With every label held within its bound a
    crowded stretch may have no placement that separates it. The gaps
    are then relaxed toward what the pairs already have, each stretch as
    little as its room requires (:func:`_relax_to_fit`), and the solve
    is exact for those gaps: the labels move as far apart as their
    bounds allow and some overlap remains -- and the pass says so,
    returning ``complete=False``. The strategy then lays the whole pie's
    outside labels out as elbow labels (:func:`_as_elbows`), whose
    columns have the chart's height to spread into, so that layout is
    never what is drawn. ``complete=False`` means this solve did not
    clear every conflict, not that none could be cleared: the
    separations are estimated at each pair's middle and relaxed stretch
    by stretch, so on some pies a tight packing within the bounds
    exists -- always with leaders near the rim's tangent -- that the
    solve does not find. On the 2,430 radial and unled charts a review
    checked against a separate greedy packer, that was 80 of 496
    fallbacks; elbow labels read better there anyway.

    **Verification.** The separations are right for each pair's middle,
    not for wherever the solve puts it, so after each solve every label
    is re-placed on its ray and every neighbouring pair's *rendered* box
    measured, the cut pair included. A pair that was given its full
    separation and still conflicts has it re-found at its new middle
    (or widened by what the gap lacks, over the labels' radius), and the
    chain is solved again, for at most :data:`_MAX_SPREAD_ROUNDS`
    rounds. A pair that was relaxed for lack of room is not widened:
    there is no room to widen it into, and widening it anyway is what
    would let the labels run round the circle.

    **The disc.** A label of a wedge centred on the chart's centre that
    is moved to ``phi`` sits exactly where the label of a wedge with its
    mid-ray on ``phi`` would, so it is as clear of the disc as any
    unmoved label, and clearing a leaning leader
    (:func:`_clear_leader`) only slides it further out along its ray. A
    wedge pulled out by a dict ``pull`` breaks that for its neighbours:
    it, and the leader off its rim, reach into the ring their labels
    move round. So after every solve a moved label that touches any
    wedge or any other label's leader is slid further out along its own
    ray until it touches none (:func:`_slide_clear`). Two things a dict
    ``pull`` can still cause: the leader of a label leaning toward a
    pulled-out wedge can cross that wedge's leader, which starts further
    out than its own; and a slid label can land on a label the chain
    does not join it to. For the second there is a safety net: if the
    closest pair of labels ends up closer than with nothing moved, the
    pass puts every label back where it was.

    Returns
    -------
    rounds : int
        Rounds of solve-then-measure used; 0 with fewer than two labels.
    complete : bool
        Whether every crowded pair got its full separation within the
        bounds: ``False`` where this solve left a conflict, which is
        usually but not always for want of room (see "When there is not
        room").
    """
    n = len(labels)
    if n < 2:
        return 0, True
    spacing, tol = geom.label_spacing, geom.tolerance
    two_pi = 2.0 * math.pi
    halves = [_half_extents(label.text, geom.renderer, geom.scale)
              for label in labels]

    def box_on(i: int, phi: float):
        (x, y), _ = _ray_layout(labels[i], phi, geom, *halves[i])
        half_w, half_h = halves[i]
        return x - half_w, y - half_h, x + half_w, y + half_h

    radii = []
    for label in labels:
        cx, cy = label.record.patch.center
        x, y = label.text.get_position()
        radii.append(math.hypot(x - cx, y - cy))
    beta = [math.acos(min(1.0, label.record.r_outer
                          / (label.record.r_outer + geom.offset)))
            if geom.offset > 0.0 else 0.0 for label in labels]

    theta = [label.mid % two_pi for label in labels]
    order = sorted(range(n), key=lambda i: (theta[i], i))
    half_angle = [
        (abs(halves[i][0] * math.sin(theta[i]))
         + abs(halves[i][1] * math.cos(theta[i]))) / radii[i]
        for i in range(n)
    ]
    free = []
    for k in range(n):
        a, b = order[k], order[(k + 1) % n]
        between = (theta[b] - theta[a]) % two_pi
        free.append(between - half_angle[a] - half_angle[b])
    cut = free.index(max(free))
    chain = order[cut + 1:] + order[:cut + 1]

    # Unwrapped angles along the chain; each is its label's own `mid`
    # plus whole turns, so an unmoved label keeps `mid` exactly.
    angles = [labels[chain[0]].mid]
    for k in range(1, n):
        step = (theta[chain[k]] - theta[chain[k - 1]]) % two_pi
        turns = round((angles[-1] + step - labels[chain[k]].mid) / two_pi)
        angles.append(labels[chain[k]].mid + turns * two_pi)
    bound = [beta[i] for i in chain]
    mean_radius = sum(radii) / n

    def requirement(i: int, j: int, middle: float, hint: float) -> float:
        return _pair_requirement(lambda delta: _box_gap(
            box_on(i, middle - 0.5 * delta), box_on(j, middle + 0.5 * delta)),
            spacing + tol, hint)

    # Pairs 0..n-2 along the chain, and n - 1: the cut pair, from the
    # chain's last label round to its first.
    def pair(k: int):
        return (k, k + 1) if k < n - 1 else (n - 1, 0)

    def separation(k: int, positions) -> float:
        a, b = pair(k)
        return positions[b] - positions[a] + (two_pi if k == n - 1 else 0.0)

    unmoved = [_data_box(labels[i].text, geom.renderer, ax) for i in chain]
    sep = [separation(k, angles) for k in range(n)]
    need = []
    for k in range(n):
        a, b = pair(k)
        middle = angles[a] + 0.5 * sep[k]
        # With two labels the cut pair is the same two labels, seen the
        # wide way round: their boxes overlap the short way, but the
        # wide way they need only what the boxes need there. Judging it
        # by the short way's overlap, or bracketing its bisection from
        # its present separation -- where, the wide way round, the boxes
        # meet again -- would ask for the whole separation, leave the cut
        # no slack, and pin both labels on their mid-rays.
        wide = n == 2 and k == 1
        estimate = requirement(chain[a], chain[b], middle,
                               0.0 if wide else sep[k])
        if not wide and _box_gap(unmoved[a], unmoved[b]) < spacing - tol:
            need.append(max(estimate, sep[k]))
        else:
            need.append(min(estimate, sep[k]))

    def bounds_for(lam, needs):
        lower = [angles[k] - bound[k] for k in range(n)]
        upper = [angles[k] + bound[k] for k in range(n)]
        base, extra = min(needs[-1], sep[-1]), max(needs[-1] - sep[-1], 0.0)
        slack = 0.5 * (sep[-1] - (base + lam[-1] * extra))
        upper[-1] = min(upper[-1], angles[-1] + slack)
        lower[0] = max(lower[0], angles[0] - slack)
        return lower, upper

    wedges = [record.patch.get_patch_transform().transform_path(
                  record.patch.get_path()) for record in records]
    complete = True
    for rounds in range(1, _MAX_SPREAD_ROUNDS + 1):
        base = [min(need[k], sep[k]) for k in range(n)]
        extra = [max(need[k] - sep[k], 0.0) for k in range(n)]
        lam = _relax_to_fit(base, extra,
                            lambda trial: bounds_for(trial, need)[0],
                            lambda trial: bounds_for(trial, need)[1], n)
        complete = all(lam[k] >= 1.0 for k in range(n) if extra[k] > 0.0)
        gaps = [base[k] + lam[k] * extra[k] for k in range(n - 1)]
        lower, upper = bounds_for(lam, need)
        phis, moved = _pav_with_gaps(angles, gaps, lower, upper)
        for k, i in enumerate(chain):
            # As in `_spread_in_columns`: a label that stays unmoved is
            # not laid out again, so it keeps its floats to the bit.
            phi = (min(max(phis[k], angles[k] - bound[k]),
                       angles[k] + bound[k]) if moved[k] else None)
            if phi is not None or labels[i].phi is not None:
                labels[i].phi = phi
                labels[i].slide = 0.0
                _apply_outside(labels[i], geom)
        for i, label in enumerate(labels):
            if label.phi is None:
                continue
            leaders = [tuple(zip(other.leader.get_xdata(),
                                 other.leader.get_ydata()))
                       for other in labels
                       if other is not label and other.leader is not None]
            _slide_clear(label, geom, *halves[i], wedges,
                         [(line[0], line[1]) for line in leaders])
            _apply_outside(label, geom)
        placed = [_data_box(labels[i].text, geom.renderer, ax) for i in chain]
        widen = []
        for k in range(n if n > 2 else 1):
            a, b = pair(k)
            gap = _box_gap(placed[a], placed[b])
            if gap < spacing - tol and (extra[k] == 0.0 or lam[k] >= 1.0):
                widen.append((k, gap))
        if not widen or rounds == _MAX_SPREAD_ROUNDS:
            break
        positions = [labels[i].phi if labels[i].phi is not None else angles[k]
                     for k, i in enumerate(chain)]
        for k, gap in widen:
            a, b = pair(k)
            current = separation(k, positions)
            estimate = requirement(chain[a], chain[b],
                                   positions[a] + 0.5 * current, current)
            need[k] = max(need[k], estimate if estimate > current + 1e-12
                          else current + max(spacing - gap, tol) / mean_radius)
    # The safety net: see "The disc" above. Should the closest pair end
    # up closer than with nothing moved, nothing is moved.
    if _closest(placed) < _closest(unmoved) - tol:
        for label in labels:
            if label.phi is not None:
                label.phi, label.slide = None, 0.0
                _apply_outside(label, geom)
        complete = False
    return rounds, complete


def _closest(boxes) -> float:
    """The least gap between any two of ``boxes``."""
    return min((_box_gap(a, b) for k, a in enumerate(boxes)
                for b in boxes[k + 1:]), default=math.inf)


def _pair_requirement(gap_at, spacing: float, hint: float) -> float:
    """Least angle ``delta`` at which ``gap_at(delta) >= spacing``.

    ``gap_at`` places a pair of labels ``delta`` apart and returns the
    gap between their boxes. The bracket starts at ``hint`` -- the
    pair's present separation -- and doubles until it holds, then is
    bisected. A pair that a whole turn does not separate gets a whole
    turn, which leaves the solver to do what it can.
    """
    lo, hi = 0.0, max(hint, 1e-6)
    while gap_at(hi) < spacing:
        if hi >= 2.0 * math.pi:
            return hi
        lo, hi = hi, min(2.0 * hi, 2.0 * math.pi)
    for _ in range(_BISECT_STEPS):
        midpoint = 0.5 * (lo + hi)
        if gap_at(midpoint) >= spacing:
            hi = midpoint
        else:
            lo = midpoint
    return hi


def _wedge_values_strategy(
    ax: Axes,
    *,
    fmt: str,
    anchor,
    offset: Optional[float],
    color,
    pad: float,
    rotation=0.0,
    basis: Optional[str] = None,
    leader=_LEADER_DEFAULT,
    leader_gap: Optional[float] = None,
    elbow_length: Optional[float] = None,
    avoid_overlap: Optional[bool] = None,
    label_spacing: Optional[float] = None,
    **text_kws,
) -> List[Text]:
    """Label every wedge on ``ax`` with its share, or its formatted value.

    ``anchor="inside"`` centres each label in its wedge's radial band;
    ``"outside"`` puts it beyond the rim on the wedge's mid-ray, joined
    to the rim by a leader line in the wedge's own edge colour at
    ``edgewidth``; ``"auto"`` measures each label inside
    (:func:`_label_fits`) and moves only the ones that do not fit, which
    are then placed exactly as ``"outside"`` places them.

    ``leader`` picks how an outside label hangs off its wedge --
    ``"radial"``, ``"elbow"`` or ``None`` (``False`` too) for no line --
    and :func:`_outside_layout` gives the geometry of each. ``offset``
    runs from the rim to the leader's end (to the label itself, with no
    leader), ``leader_gap`` from the leader's end to the label, and
    ``elbow_length`` is an elbow's horizontal tail, all in millimetres.
    Each defaults to its ``pie.*`` rcParam: ``pie.leader``,
    ``pie.label_offset``, ``pie.leader_gap`` and ``pie.elbow_length``.

    ``avoid_overlap`` (``pie.avoid_overlap``, ``True``) then moves
    outside labels that crowd one another apart, keeping
    ``label_spacing`` mm (``pie.label_spacing``, ``0.4``) between any
    two boxes: elbow labels up and down their column
    (:func:`_spread_in_columns`), the others round the circle
    (:func:`_spread_on_circle`). Where that bounded pass does not clear
    every conflict, every outside label is laid out as an elbow label
    instead (:func:`_as_elbows`) and spread up and down its
    column. A label that crowds nothing is not moved, and with
    ``avoid_overlap=False`` neither pass runs.

    The axis limits are never changed here -- they are whatever
    :func:`publiplots.pieplot` framed, room for ``pull`` included -- so
    ``inside``, ``outside`` and ``auto`` all draw the same disc. Outside
    labels and leaders are unclipped and sit beyond the axes frame, and
    the layout grows the figure around them the way it does for any
    other decoration while the axes keeps its ``axes_size``. Because the
    axes' millimetre size is fixed, every length is converted to data
    units once and renders as exactly that many millimetres.

    ``pad`` is accepted for signature parity with the other strategies
    and has **no effect** here: it is a margin between labels and the
    axes edge, and these labels are not kept inside that edge.
    """
    meta = _meta(ax)
    anchor = _DEFAULT_ANCHOR if anchor is None else anchor
    if anchor not in _VALID_ANCHORS:
        raise ValueError(
            f"wedge_values anchor must be one of {sorted(_VALID_ANCHORS)}; "
            f"got {anchor!r}"
        )
    basis = _DEFAULT_BASIS if basis is None else basis
    if basis not in _VALID_BASES:
        raise ValueError(
            f"basis must be one of {sorted(_VALID_BASES)}; got {basis!r}"
        )
    if isinstance(rotation, str) and rotation not in _VALID_ROTATIONS:
        raise ValueError(
            f"rotation must be a finite number of degrees or one of "
            f"{sorted(_VALID_ROTATIONS)}; got {rotation!r}"
        )
    leader = _resolve_leader(leader)
    offset = _resolve_length("offset", "pie.label_offset", offset)
    leader_gap = _resolve_length("leader_gap", "pie.leader_gap", leader_gap)
    elbow_length = _resolve_length("elbow_length", "pie.elbow_length",
                                   elbow_length)
    avoid_overlap = _resolve_avoid_overlap(avoid_overlap)
    label_spacing = _resolve_length("label_spacing", "pie.label_spacing",
                                    label_spacing)
    # Defensive: the dispatcher always passes `rotation=` explicitly, so
    # text_kws cannot currently carry one and this pop never fires. It
    # stays because the top-level parameter must win if a future caller
    # reaches this function directly -- `ax.text` would raise on the
    # duplicate keyword rather than pick one.
    text_kws.pop("rotation", None)

    if not meta.wedges:
        return []

    renderer = _ensure_renderer(ax)

    # Every conversion is made once, for the whole call. Nothing here
    # touches the axis limits and the layout fixes the axes box in
    # millimetres, so a data unit is worth the same number of millimetres
    # now as when the figure is drawn: a length converted here renders
    # as exactly that many millimetres, with no framing to settle first.
    # `apply_aspect` first, so that under `aspect="equal"` the box these
    # read is the square matplotlib will draw rather than the full cell
    # -- see `_data_per_pixel`.
    ax.apply_aspect()
    geom = _OutsideGeometry(
        leader=leader,
        offset=mm_to_data(offset, ax, "y"),
        leader_gap=mm_to_data(leader_gap, ax, "y"),
        elbow_length=mm_to_data(elbow_length, ax, "y"),
        scale=_data_per_pixel(ax),
        renderer=renderer,
        label_spacing=mm_to_data(label_spacing, ax, "y"),
        tolerance=mm_to_data(_SPACING_TOLERANCE_MM, ax, "y"),
    )
    # A leader outlines the gap between rim and label rather than being
    # data itself, so it takes `edgewidth`, not `lines.linewidth`.
    edgewidth = resolve_param("edgewidth", None)
    # Outside labels and their leaders are decorations: they reach past
    # the axes frame, and the layout grows the figure to hold them the
    # way it does for a title or a tick label. A clipped label would be
    # cut at the frame instead, so neither may clip. `ax.text` already
    # defaults to unclipped; saying so keeps that from being an accident.
    text_kws.setdefault("clip_on", False)

    texts: List[Text] = []
    outside: List[_OutsideLabel] = []
    for record in meta.wedges:
        share = (record.fraction_of_total if basis == "total"
                 else record.fraction_of_parent)
        # Percentage is the default; an explicit spec formats the value.
        if fmt == _WEDGE_DEFAULT_FMT:
            label = format(share * 100.0, ".1f") + "%"
        else:
            label = _format_value(record.value, fmt)

        mid = math.radians((record.theta0 + record.theta1) / 2.0)
        place_outside = anchor == "outside"
        # Every label is born at its inside point. One that ends outside is
        # laid out from its rendered box, so the Text has to exist first.
        x, y, ha = _inside_point(record, mid)

        text = ax.text(
            x, y, label,
            ha=ha, va="center",
            rotation=_rotation_for(rotation, math.degrees(mid)),
            # resolve_color reads `.patch` and `.hue_color` off the record
            # and branches on the anchor, so a WedgeRecord works unchanged.
            # Pass the anchor this label actually uses, not the one that
            # was requested: under `auto` they differ per wedge, and the
            # contrast calculation is only correct for a label drawn over
            # a fill.
            color=resolve_color(
                record,
                color,
                "outside" if place_outside else "inside",
                ax,
                hue_active=meta.hue_active,
            ),
            **text_kws,
        )
        texts.append(text)

        # Under `auto` the label is measured where it was born, inside its
        # wedge, and moved out only if it does not fit. One pass is
        # enough: the disc no longer shrinks when a label leaves it, so a
        # label that fitted cannot be unseated by another one moving out.
        if anchor == "auto" and not _label_fits(text, record, renderer,
                                                 geom.scale, mid):
            place_outside = True
            text.set_color(resolve_color(
                record, color, "outside", ax, hue_active=meta.hue_active,
            ))

        if not place_outside:
            continue
        line = (_leader_line(ax, record, edgewidth) if leader is not None
                else None)
        placed = _OutsideLabel(text=text, leader=line, record=record, mid=mid)
        _apply_outside(placed, geom)
        outside.append(placed)

    # `outside` holds every label that left its wedge with its leader;
    # the pass moves crowded ones apart and re-applies each pair through
    # `_apply_outside`. Inside labels are never moved.
    if avoid_overlap and len(outside) > 1:
        if leader == "elbow":
            _spread_in_columns(outside, geom, ax, meta.wedges)
        else:
            _, cleared = _spread_on_circle(outside, geom, ax, meta.wedges)
            if not cleared:
                elbow = _as_elbows(outside, geom, ax, edgewidth)
                _spread_in_columns(outside, elbow, ax, meta.wedges)

    # Outside labels and leaders -- where the pass left them, elbow
    # leaders the fallback drew included -- sit past the disc, where a
    # per-axes legend and the axes title would otherwise be drawn over
    # them. Registering them makes both step past them. They are placed
    # in data coordinates
    # on an axes whose millimetre size the layout keeps fixed, so neither
    # the legend nor the figure's growth moves them -- the condition
    # `register_decoration` asks for.
    for placed in outside:
        register_decoration(ax, placed.text)
        if placed.leader is not None:
            register_decoration(ax, placed.leader)
    return texts


def _leader_line(ax: Axes, record: WedgeRecord, edgewidth: float) -> Line2D:
    """An empty leader line for ``record``'s label, styled as every one is.

    A leader outlines the gap between rim and label rather than being
    data itself, so it takes ``edgewidth``, not ``lines.linewidth``.
    """
    (line,) = ax.plot(
        [], [],
        # The wedge's own edge, not the themed `edgecolor`: that rcParam
        # is None by default, which means "match the fill" for a wedge
        # outline and would mean "take the next colour off the property
        # cycle" here -- leaders in colours from no wedge at all. Reading
        # the drawn patch honours an explicit edgecolor too, since the
        # patch already carries it.
        color=record.patch.get_edgecolor(),
        linewidth=edgewidth,
        solid_capstyle="butt",
        # An elbow is one polyline, so its corner is a join; a round one
        # reads as a single bent stroke.
        solid_joinstyle="round",
        zorder=record.patch.get_zorder() + 1,
        # Line2D clips to the axes by default, which would cut a leader
        # off at the frame on its way to a label beyond it.
        clip_on=False,
    )
    return line


def _as_elbows(labels: List[_OutsideLabel], geom: _OutsideGeometry,
               ax: Axes, edgewidth: float) -> _OutsideGeometry:
    """Lay every outside label of the axes out as an elbow label.

    The fallback for a pie crowded on the circle: when
    :func:`_spread_on_circle` does not clear every conflict within its
    bounds (see its "When there is not room"), the radial or unled
    labels it was asked for would be left overlapping, with leaders
    leaning as far as they can. The whole
    pie's outside labels are then laid out as ``leader="elbow"`` labels
    instead, so that :func:`_spread_in_columns` can spread them up and
    down the chart's height, which always has room. Every label is
    re-laid out, not only the crowded ones, so the chart keeps one
    leader style. The configured ``offset``, ``leader_gap`` and
    ``elbow_length`` are used as they are; with no leader asked for,
    elbow leaders are drawn, since an elbow label beside a column has
    no other way to say which wedge it names. Each label keeps its own
    :class:`~matplotlib.text.Text`, so what ``annotate`` returns is
    unchanged.

    Returns
    -------
    _OutsideGeometry
        ``geom`` with ``leader='elbow'``, for the column pass.
    """
    elbow = replace(geom, leader="elbow")
    for label in labels:
        label.phi, label.slide, label.label_y = None, 0.0, None
        if label.leader is None:
            label.leader = _leader_line(ax, label.record, edgewidth)
        _apply_outside(label, elbow)
    return elbow
