"""
Pie and Donut Charts
=====================

Part-of-whole composition, in the same visual language as the rest of a
figure set: the publiplots palette, the theme edge, and a legend that can
join a shared band. ``pp.pieplot`` is a one-level ``pp.sunburstplot`` —
``x`` names the wedge, ``y`` the value (omitted, rows are counted), and
``hue`` means colour and nothing else.
"""

import pandas as pd
import publiplots as pp

df = pd.DataFrame({
    "kind": ["symptom", "admin", "lay_dx", "none"],
    "n": [52.2, 41.8, 6.0, 4.0],
})

# %%
# A pie beside a bar panel
# ------------------------
# The case this plotter exists for. Both panels take the same ``palette``
# dict keyed on the same categories, so the reader can map one to the
# other without a legend lookup. ``annotate=True`` labels each wedge with
# its share of the whole, as a percentage.

palette = {"symptom": "#75b375", "admin": "#8e8ec1",
           "lay_dx": "#eeaa58", "none": "#e67e7e"}

fig, axes = pp.subplots(ncols=2, axes_size=(50, 45))
pp.barplot(data=df, x="kind", y="n", palette=palette, legend=False,
           ax=axes[0])
pp.pieplot(data=df, x="kind", y="n", palette=palette, legend=False,
           ax=axes[1], annotate=True)
pp.show()

# %%
# A donut, with the total in the hole
# -----------------------------------
# ``center_label`` takes a string, or a callable receiving the total, and
# is only drawn when ``donut`` is non-zero.

fig, ax = pp.subplots(axes_size=(50, 50))
pp.pieplot(data=df, x="kind", y="n", donut=0.55, palette=palette,
           center_label="{:,.0f}".format, ax=ax, annotate=True)
pp.show()

# %%
# Pulling a wedge out
# --------------------
# ``pull`` is keyed by category, never positional.

fig, ax = pp.subplots(axes_size=(50, 50))
pp.pieplot(data=df, x="kind", y="n", palette=palette,
           pull={"lay_dx": 0.15}, ax=ax)
pp.show()

# %%
# Collapsing the long tail
# -------------------------
# ``min_share`` merges every category below a threshold into one
# ``"Other"`` wedge. It applies only when at least two categories fall
# below it, so a single small category is never relabelled ``"Other"``
# on its own. ``min_share`` cannot be combined with ``order``: the
# ``"Other"`` bucket is not a category of the data, so it could not be
# named in an explicit order.

fig, ax = pp.subplots(axes_size=(50, 50))
pp.pieplot(data=df, x="kind", y="n", min_share=0.1, ax=ax, annotate=True)
pp.show()

# %%
# Labels inside, outside, or chosen per wedge
# ---------------------------------------------
# ``anchor="auto"`` (the default) measures each label against the room
# its own wedge has, and moves it outside with a leader line when it
# does not fit — the thin slivers below. ``"inside"`` and ``"outside"``
# force every label to one placement regardless of fit.

thin = pd.DataFrame({"kind": list("ABCD"), "n": [80.0, 12.0, 6.0, 2.0]})
fig, axes = pp.subplots(ncols=3, axes_size=(38, 38))
for ax, anchor in zip(axes, ["inside", "outside", "auto"]):
    pp.pieplot(data=thin, x="kind", y="n", legend=False, ax=ax)
    pp.annotate(ax, "wedge_values", anchor=anchor)
    ax.set_title(anchor)
pp.show()

# %%
# Colouring by something other than the category
# -------------------------------------------------
# ``hue`` means colour and nothing else — unlike ``barplot``, where
# ``hue`` also controls how marks group, a wedge chart keeps its
# structure in ``x``. ``hue`` may name a column outside the one that
# defines the wedges, as long as it is constant within each wedge.

staff = pd.DataFrame({
    "dept": ["cardio", "neuro", "billing", "records"],
    "wing": ["clinical", "clinical", "admin", "admin"],
    "n": [30.0, 24.0, 18.0, 12.0],
})
fig, ax = pp.subplots(axes_size=(50, 50))
pp.pieplot(data=staff, x="dept", y="n", hue="wing",
           palette={"clinical": "#75b375", "admin": "#8e8ec1"}, ax=ax)
pp.show()
