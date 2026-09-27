Configuration
=============

rcParams
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   publiplots.rcParams
   publiplots.resolve_param

Sunburst colour derivation
--------------------------

Two ``publiplots.rcParams`` keys control the colour shading
:func:`publiplots.sunburstplot` (and, inertly, :func:`publiplots.pieplot`,
a one-level sunburst) derives for descendant wedges. Both are expressed
as a **total span across the whole tree**, not a per-ring step, so the
value needs no retuning when a level is added.

``sunburst.depth_range`` : float, default ``0.45``
    How far the outermost ring blends toward white, interpolated across
    intermediate rings. Negative blends toward black instead. Inert for
    a single-ring chart, so ``pieplot`` pays nothing for it.

``sunburst.sibling_range`` : float, default ``0.14``
    The spread across wedges sharing a parent, interpolated over
    however many siblings there are. Inert for a single-ring chart.
