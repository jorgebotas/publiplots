"""
Sunburst Charts
===============

A part-of-whole hierarchy, one ring per level. The root ring splits the
whole, as a pie would, and each ring outside it splits its parent. ``x``
lists one column per ring, root first, ``y`` the value (omitted, rows
are counted), and ``hue`` what determines colour -- the root column by
default. An outer wedge takes its root's colour, shaded by ring and by
its place among its siblings, so a branch reads as one colour from the
centre to the rim.
"""

import pandas as pd
import publiplots as pp

visits = pd.DataFrame(
    [
        ("symptom", "pain", "acute", 13.0),
        ("symptom", "pain", "chronic", 9.0),
        ("symptom", "fever", "child", 9.0),
        ("symptom", "fever", "adult", 7.0),
        ("symptom", "rash", "new", 7.0),
        ("symptom", "rash", "recurring", 5.0),
        ("admin", "billing", "invoice", 10.0),
        ("admin", "billing", "insurance", 6.0),
        ("admin", "records", "request", 7.0),
        ("admin", "records", "transfer", 4.0),
        ("admin", "referral", "specialist", 4.0),
        ("admin", "referral", "imaging", 3.0),
        ("lay_dx", "online", "search", 4.0),
        ("lay_dx", "online", "app", 2.0),
        ("lay_dx", "family", "parent", 2.5),
        ("lay_dx", "family", "partner", 1.5),
        ("none", "follow_up", "booked", 2.5),
        ("none", "follow_up", "unbooked", 1.5),
        ("none", "other", "walk_in", 2.0),
    ],
    columns=["kind", "topic", "detail", "n"],
)

# The same colours as the pie and donut page, keyed on the root column.
palette = {"symptom": "#8e8ec1", "admin": "#eeaa58",
           "lay_dx": "#e67e7e", "none": "#75b375"}

# %%
# A sunburst beside the pie of its root ring
# ------------------------------------------
# The same frame, drawn twice. The pie splits the whole by ``kind``; the
# sunburst draws that split as its root ring and then splits each wedge
# again by ``topic`` and by ``detail``, outwards. Each ring is a third of
# the radius. Both panels take the same ``palette`` dict, so a category
# has one colour across the figure, and the sunburst's key lists only
# the root categories: the paler shades of the outer rings are the same
# colour, not new ones. The two legends merge into one band, by name.

fig, axes = pp.subplots(ncols=2, axes_size=(50, 50))
pp.legend(side="right")
pp.pieplot(data=visits, x="kind", y="n", palette=palette, ax=axes[0],
           title="pieplot")
pp.sunburstplot(data=visits, x=["kind", "topic", "detail"], y="n",
                palette=palette, ax=axes[1], title="sunburstplot")
pp.show()

# %%
# Colour across rings
# -------------------
# Two knobs shade the outer rings from the root colour, both a **total
# span** rather than a per-ring step, so adding a ring never washes the
# rim out to white. ``depth_range`` is how far the outermost ring blends
# toward white; rings between interpolate, and the root ring keeps the
# pure palette colour. ``sibling_range`` spreads the wedges under one
# parent from none to that much more, in draw order, so neighbours stay
# apart. Both at ``0`` paint each branch one flat colour (left); the
# defaults are ``0.45`` and ``0.14`` (``pp.rcParams["sunburst.depth_range"]``
# and ``pp.rcParams["sunburst.sibling_range"]``, middle). A negative
# ``depth_range`` blends toward black instead, for a dark rim (right).
# ``|depth_range| + |sibling_range|`` may not exceed 1.

fig, axes = pp.subplots(ncols=3, axes_size=(42, 42))
pp.legend(side="right")
for ax, (title, kws) in zip(axes, [
        ("flat", {"depth_range": 0, "sibling_range": 0}),
        ("default", {}),
        ("depth_range=-0.45", {"depth_range": -0.45})]):
    pp.sunburstplot(data=visits, x=["kind", "topic", "detail"], y="n",
                    palette=palette, ax=ax, title=title, **kws)
pp.show()

# %%
# Branches that end early
# -----------------------
# Hierarchies need not be the same depth everywhere. A row whose deeper
# columns are ``None`` (or ``NaN``) ends at its last named level: here
# ``fever`` and ``records`` have no ``detail``, so the ring outside them
# stays empty over their span. A wedge may also have children *and*
# rows of its own. ``symptom`` has 12 visits with no ``topic`` beside
# its ``pain`` and ``fever`` rows, and ``billing`` 4 with no ``detail``:
# the children are laid out first, from the wedge's leading edge, and
# the wedge's own share is left as an empty stretch at its trailing end,
# in every ring outside it. A parent is always the sum of what it
# holds -- its children and its own rows -- so the empty stretch is
# exactly the share that goes no deeper.

ragged = pd.DataFrame(
    [
        ("symptom", "pain", "acute", 13.0),
        ("symptom", "pain", "chronic", 9.0),
        ("symptom", "fever", None, 16.0),
        ("symptom", None, None, 12.0),
        ("admin", "billing", "invoice", 10.0),
        ("admin", "billing", "insurance", 6.0),
        ("admin", "billing", None, 4.0),
        ("admin", "records", None, 11.0),
        ("lay_dx", "online", "search", 4.0),
        ("lay_dx", "online", "app", 2.0),
        ("none", "follow_up", "booked", 2.5),
        ("none", "follow_up", "unbooked", 1.5),
    ],
    columns=["kind", "topic", "detail", "n"],
)
fig, ax = pp.subplots(axes_size=(55, 55))
pp.sunburstplot(data=ragged, x=["kind", "topic", "detail"], y="n",
                palette=palette, ax=ax)
pp.show()

# %%
# Gaps and explode
# ----------------
# ``gap`` separates every pair of neighbouring wedges by a width in
# millimetres, siblings and rings alike, without moving anything out:
# a seam between siblings runs straight through every ring outside it,
# and each boundary between rings opens by the same width. A float
# ``pull`` explodes the whole chart evenly instead, as a fraction of the
# radius: every ring moves out, keeping its width, so every seam and
# every ring boundary opens to the same ``2 * pull``, and a small hole
# opens at the centre. With both, the two add up.

fig, axes = pp.subplots(ncols=2, axes_size=(50, 50))
pp.legend(side="right")
for ax, (title, kws) in zip(axes, [("gap=1.0", {"gap": 1.0}),
                                   ("pull=0.05", {"pull": 0.05})]):
    pp.sunburstplot(data=visits, x=["kind", "topic", "detail"], y="n",
                    palette=palette, ax=ax, title=title, **kws)
pp.show()

# %%
# Pulling one branch out
# ----------------------
# A dict ``pull`` pops out just the wedges it names, each along its own
# mid-ray, and each takes its whole subtree with it, so the chart never
# tears between rings. A key is a root category (``"admin"``), or a path
# from the root (``("admin", "billing")``) for a wedge further out; a
# key that names no wedge raises. Pulling ``admin`` moves the branch
# from the root ring to the rim; pulling ``("admin", "billing")`` moves
# ``billing`` and its two ``detail`` wedges, and leaves the rest of
# ``admin`` where it was.

fig, axes = pp.subplots(ncols=2, axes_size=(50, 50))
pp.legend(side="right")
for ax, key in zip(axes, ["admin", ("admin", "billing")]):
    pp.sunburstplot(data=visits, x=["kind", "topic", "detail"], y="n",
                    palette=palette, pull={key: 0.12}, ax=ax,
                    title=f"pull={{{key!r}: 0.12}}")
pp.show()

# %%
# Colouring by a number
# ---------------------
# A numeric ``hue`` maps each root wedge through a colormap, and the
# legend becomes a colorbar. The value must be constant within each
# root -- here the median wait in days for each kind of visit -- and the
# wedges further out take their root's colour unshaded, since a lighter
# shade would read as a different number on the colorbar. With every
# ring of a branch the same colour, the wedges are filled solid
# (``alpha=1``) to read against the colorbar, and outlined in white to
# keep the seams.

days = {"symptom": 3.0, "admin": 11.0, "lay_dx": 26.0, "none": 40.0}
waits = visits.assign(days=visits["kind"].map(days))
fig, ax = pp.subplots(axes_size=(50, 50))
pp.sunburstplot(data=waits, x=["kind", "topic", "detail"], y="n",
                hue="days", palette="viridis", alpha=1,
                edgecolor="white", ax=ax)
pp.show()

# %%
# Patterns on one ring
# --------------------
# ``hatch`` naming one of the ``x`` columns gives that ring's wedges
# their patterns, and every wedge outside them inherits its parent's;
# the rings inside stay plain. Here each kind of visit splits into
# ``in_person`` and ``remote`` visits, and ``hatch="setting"`` marks the
# remote ones -- in the setting ring and, below them, in every
# ``detail`` wedge too. The patterns get their own legend entry, beside
# the colours.

settings = pd.DataFrame(
    [
        ("symptom", "in_person", "walk_in", 18.0),
        ("symptom", "in_person", "booked", 14.0),
        ("symptom", "remote", "phone", 10.0),
        ("symptom", "remote", "video", 8.0),
        ("admin", "in_person", "desk", 12.0),
        ("admin", "remote", "phone", 14.0),
        ("admin", "remote", "portal", 8.0),
        ("lay_dx", "in_person", "walk_in", 6.0),
        ("lay_dx", "remote", "portal", 4.0),
        ("none", "in_person", "walk_in", 6.0),
    ],
    columns=["kind", "setting", "channel", "n"],
)
fig, ax = pp.subplots(axes_size=(55, 55))
pp.sunburstplot(data=settings, x=["kind", "setting", "channel"], y="n",
                palette=palette, hatch="setting",
                hatch_map={"in_person": "", "remote": "////"}, ax=ax)
pp.show()

# %%
# Labels
# ------
# Wedge labels for sunburst charts, through ``annotate=``, arrive in the
# next release; for now ``annotate=`` raises ``NotImplementedError``.
# Each chart above is labelled by its legend.
