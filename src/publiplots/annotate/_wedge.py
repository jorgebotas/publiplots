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
:class:`_OutsideLabel`, so that a later pass moving labels apart can move
a label and re-route its leader in one step.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

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
            f"leader must be 'radial', 'elbow' or None; got {leader!r}"
        )
    return leader


def _resolve_length(name: str, key: str, value: Optional[float]) -> float:
    """A non-negative finite length in mm, defaulting to ``rcParams[key]``."""
    value = resolve_param(key, value)
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number of mm; got {value!r}")
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite number >= 0; got {value!r}")
    return value


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
    """

    text: Text
    leader: Optional[Line2D]
    record: WedgeRecord
    mid: float


def _support(text: Text, renderer, scale: float,
             direction: Tuple[float, float]) -> float:
    """Reach of the rendered label's box from its centre along ``direction``.

    The support function of an axis-aligned rectangle of half-sizes
    ``w/2``, ``h/2``: ``|w/2 * dx| + |h/2 * dy|``. ``get_window_extent``
    already folds in any rotation, so this is the rotated label's box.
    """
    box = text.get_window_extent(renderer)
    return (abs(box.width * scale / 2.0 * direction[0])
            + abs(box.height * scale / 2.0 * direction[1]))


def _outside_layout(text: Text, record: WedgeRecord, mid: float,
                    geom: _OutsideGeometry):
    """Where an outside label goes, and the vertices of its leader.

    With ``d`` the unit mid-ray and ``c`` the wedge's own centre -- the
    patch centre, which already carries any ``pull``, so a pulled wedge
    keeps its label -- the rim point is ``c + r_outer * d``, and:

    - ``'radial'``: the leader runs from the rim to ``offset`` beyond it
      along ``d``. The label's box is centred on the mid-ray with its
      nearest point ``leader_gap`` past the leader's end, measured along
      ``d``: its centre sits at ``r_outer + offset + leader_gap + s``,
      where ``s`` is the box's support along ``d`` (:func:`_support`).
      Every point of the box then projects at least that far along
      ``d``, so the box clears the whole leader by ``leader_gap`` and its
      own wedge's disc by ``offset + leader_gap``. Off the axes the
      straight-line gap from the leader's end to the box is larger than
      ``leader_gap``, because the box's nearest point is a corner beside
      the mid-ray rather than on it.
    - ``None``: no leader, and the same centred box with its nearest
      point ``offset`` beyond the rim.
    - ``'elbow'``: a radial segment from the rim to an elbow ``offset``
      beyond it, then a horizontal tail of ``elbow_length`` toward the
      label's side -- right where ``d`` points right or straight up or
      down, left otherwise. The label is ``ha='left'``/``'right'``,
      ``va='center'``, ``leader_gap`` past the tail's end.

    A rotated label is laid out by its rotated box in every style. For
    ``'radial'`` and ``None`` that is the support above; for ``'elbow'``
    it is matplotlib's default ``rotation_mode``, which applies ``ha``
    and ``va`` to the rotated box, so its near side still sits
    ``leader_gap`` past the tail and its middle level with the tail. A
    caller-supplied ``rotation_mode="anchor"`` aligns the unrotated text
    instead and gives up that guarantee.

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

    reach = _support(text, geom.renderer, geom.scale, d)
    if geom.leader == "radial":
        centre = end + geom.leader_gap + reach
        vertices = ([rim_point[0], end_point[0]],
                    [rim_point[1], end_point[1]])
    else:
        centre = end + reach
        vertices = None
    position = (cx + centre * d[0], cy + centre * d[1])
    return position, "center", "center", vertices


def _apply_outside(label: _OutsideLabel, geom: _OutsideGeometry) -> None:
    """Write :func:`_outside_layout`'s result onto the label and its leader.

    The leader must already exist when the style draws one; its vertices
    are replaced, so a later pass can call this again after moving the
    label's mid-ray or geometry and the pair stays consistent.
    """
    position, ha, va, vertices = _outside_layout(
        label.text, label.record, label.mid, geom)
    label.text.set_ha(ha)
    label.text.set_va(va)
    label.text.set_position(position)
    if label.leader is not None and vertices is not None:
        label.leader.set_data(*vertices)


def _label_fits(text: Text, record, renderer, scale: float, mid: float) -> bool:
    """Does the rendered label fit inside its own wedge?

    The label is an axis-aligned rectangle once rendered (``get_window_extent``
    already folds in any rotation), so its reach along a unit direction
    ``d`` is the rectangle's support function, ``|w/2 * d.x| + |h/2 * d.y|``.
    Two directions matter: tangential, where the wedge's two edge rays
    close in, and radial, where the ring's inner and outer radii do.

    At the label radius ``r`` the perpendicular distance from the mid-ray
    to either edge ray is ``r * sin(span / 2)``. Past a half-span of 90
    degrees the wedge stops being convex about its mid-ray and the edges
    no longer bound the label at all, so the disc itself is the limit.

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
    shorter ``fmt`` is the remedy there.

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
    if half_span >= math.pi / 2.0:
        room_t = record.r_outer
    else:
        room_t = r_label * math.sin(half_span)
    if reach_t > room_t:
        return False
    return 2.0 * reach_r <= (record.r_outer - record.r_inner)


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
        line = None
        if leader is not None:
            (line,) = ax.plot(
                [], [],
                # The wedge's own edge, not the themed `edgecolor`: that
                # rcParam is None by default, which means "match the
                # fill" for a wedge outline and would mean "take the next
                # colour off the property cycle" here -- leaders in
                # colours from no wedge at all. Reading the drawn patch
                # honours an explicit edgecolor too, since the patch
                # already carries it.
                color=record.patch.get_edgecolor(),
                linewidth=edgewidth,
                solid_capstyle="butt",
                # An elbow is one polyline, so its corner is a join; a
                # round one reads as a single bent stroke.
                solid_joinstyle="round",
                zorder=record.patch.get_zorder() + 1,
                # Line2D clips to the axes by default, which would cut a
                # leader off at the frame on its way to a label beyond it.
                clip_on=False,
            )
        placed = _OutsideLabel(text=text, leader=line, record=record, mid=mid)
        _apply_outside(placed, geom)
        outside.append(placed)

    # `outside` holds every label that left its wedge with its leader. A
    # pass that moves crowded labels apart belongs here: it adjusts the
    # pairs together and re-applies them through `_apply_outside`.
    return texts
