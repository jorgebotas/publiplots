Configuration
=============

rcParams
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   publiplots.rcParams
   publiplots.resolve_param

Pie charts
----------

Seven ``publiplots.rcParams`` keys set the defaults of
:func:`publiplots.pieplot` and of its labels,
``pp.annotate(ax, "wedge_values")``. Every length is in millimetres,
and each is overridden per call by the argument of the same name
(``offset`` for ``pie.label_offset``).

``pie.gap`` : float, default ``0.0``
    Gap between every pair of neighbouring wedges: each wedge's straight
    edges are inset by half of it, so two neighbours sit exactly this
    far apart along the whole edge they share. ``0`` draws no gap.

``pie.leader`` : {``"radial"``, ``"elbow"``, ``None``}, default ``"radial"``
    How an outside label is joined to its wedge: a straight line along
    the mid-ray, a line that bends into a horizontal tail, or no line
    (``False`` is accepted for ``None``).

``pie.label_offset`` : float, default ``2.0``
    From the rim to the leader's end -- to its elbow, for ``"elbow"`` --
    or, with no leader, to the nearest point of the label.

``pie.leader_gap`` : float, default ``0.8``
    From the leader's end to the nearest point of the label. Unused with
    no leader.

``pie.elbow_length`` : float, default ``2.0``
    Length of an ``"elbow"`` leader's horizontal tail. Unused by the
    other styles.

``pie.avoid_overlap`` : bool, default ``True``
    Whether outside labels that would overlap are moved apart. Elbow
    labels move up or down their column, and radial and unled labels
    round the circle -- never further from their wedge's mid-ray than
    a leader can lean without crossing the chart -- each by as little
    as keeps every pair ``pie.label_spacing`` apart; a label that
    crowds nothing does not move. Where that bounded solve does not
    clear every conflict -- a crowded run of slivers or a full circle,
    and also some pies where a tight radial packing with leaders close
    to the rim's tangent exists but is not reached for -- every outside
    label of the axes is laid out as an ``"elbow"`` label instead -- leader lines drawn even for ``pie.leader = None``, with
    the ``pie.label_offset``, ``pie.leader_gap`` and
    ``pie.elbow_length`` in effect -- and spread up and down its
    column. ``False`` places every label on its own wedge's mid-ray,
    in the style asked for, overlapping or not.

``pie.label_spacing`` : float, default ``0.4``
    The least gap kept between two outside labels' boxes, and between a
    moved elbow label and the disc. Unused with ``pie.avoid_overlap``
    off.

Sunburst colour shading
-----------------------

Two ``publiplots.rcParams`` keys set how :func:`publiplots.sunburstplot`
shades the wedges outside its root ring from their root's colour. Both
are a **total span across the whole tree**, not a per-ring step, so a
value needs no retuning when a ring is added, and each is overridden
per call by the argument of the same name. ``|depth_range| +
|sibling_range|`` must not exceed 1, or the call raises ``ValueError``,
whether the values are passed or read from here. Neither has any effect
on :func:`publiplots.pieplot`, which draws one ring, nor under a numeric
``hue``.

``sunburst.depth_range`` : float, default ``0.45``
    The total blend toward white (negative: toward black) at the
    outermost ring of a :func:`publiplots.sunburstplot`; rings between
    interpolate, and the root ring keeps the pure palette colour. ``0``
    shades no ring.

``sunburst.sibling_range`` : float, default ``0.14``
    The spread across siblings under one parent: the first sibling, in
    draw order, takes no extra blend and the last the full amount, the
    rest interpolating between; negative darkens. It adds to the ring's
    ``depth_range`` blend. ``0`` gives siblings their ring's shade.
