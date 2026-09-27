Configuration
=============

rcParams
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   publiplots.rcParams
   publiplots.resolve_param

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
