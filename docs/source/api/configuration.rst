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
    crowds nothing does not move. On a pie too crowded for that room,
    every outside label of the axes is laid out as an ``"elbow"`` label
    instead -- leader lines drawn even for ``pie.leader = None``, with
    the ``pie.label_offset``, ``pie.leader_gap`` and
    ``pie.elbow_length`` in effect -- and spread up and down its
    column. ``False`` places every label on its own wedge's mid-ray,
    in the style asked for, overlapping or not.

``pie.label_spacing`` : float, default ``0.4``
    The least gap kept between two outside labels' boxes, and between a
    moved elbow label and the disc. Unused with ``pie.avoid_overlap``
    off.

Sunburst colour derivation (reserved)
-------------------------------------

.. warning::

   These two keys are **reserved and have no effect yet**. Nothing
   publiplots currently draws reads them: the only part-of-whole chart
   is :func:`publiplots.pieplot`, which draws one ring and passes
   ``0.0`` for both, because a single ring has no descendant wedges to
   shade. They are documented here so the names and their meaning are
   settled ahead of a multi-ring chart; setting either changes nothing
   today.

Two ``publiplots.rcParams`` keys are reserved for the colour shading a
multi-ring part-of-whole chart would derive for descendant wedges. Both
are expressed as a **total span across the whole tree**, not a per-ring
step, so the value would need no retuning when a level is added.

``sunburst.depth_range`` : float, default ``0.45``
    How far the outermost ring would blend toward white, interpolated
    across intermediate rings. Negative would blend toward black
    instead. No effect today.

``sunburst.sibling_range`` : float, default ``0.14``
    The spread across wedges sharing a parent, interpolated over
    however many siblings there are. No effect today.
