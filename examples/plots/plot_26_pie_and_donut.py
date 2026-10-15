"""
Pie and Donut Charts
=====================

Part-of-whole composition, in the same visual language as the rest of a
figure set: the publiplots palette, the theme edge, and a legend that can
join a shared band. A pie is a one-level sunburst — ``x`` names the
wedge, ``y`` the value (omitted, rows are counted), and ``hue`` means
colour and nothing else.
"""

import pandas as pd
import publiplots as pp

df = pd.DataFrame({
    "kind": ["symptom", "admin", "lay_dx", "none"],
    "n": [52.2, 41.8, 6.0, 4.0],
})

palette = {"symptom": "#75b375", "admin": "#8e8ec1",
           "lay_dx": "#eeaa58", "none": "#e67e7e"}

# %%
# A pie beside a bar panel
# ------------------------
# The case this plotter exists for. Both panels take the same ``palette``
# dict keyed on the same categories, so each category has one colour
# across the figure. ``barplot`` only maps ``palette`` through ``hue``,
# so the bar panel passes ``hue="kind"`` as well; the pie colours by its
# ``x`` column without being asked. ``annotate=True`` labels each wedge
# with its share of the whole, as a percentage.

fig, axes = pp.subplots(ncols=2, axes_size=(50, 45))
pp.barplot(data=df, x="kind", y="n", hue="kind", palette=palette,
           legend=False, ax=axes[0])
pp.pieplot(data=df, x="kind", y="n", palette=palette, legend=False,
           ax=axes[1], annotate=True)
pp.show()

# %%
# One legend for both panels
# --------------------------
# A pie stashes its legend entry the same way every other plotter does,
# so a figure-level ``pp.legend`` band collects it alongside the bar
# panel's. The two entries share a name (``kind``) and the same
# colours, so the band shows them once, for both panels.

fig, axes = pp.subplots(ncols=2, axes_size=(50, 45))
pp.legend(side="right")
pp.barplot(data=df, x="kind", y="n", hue="kind", palette=palette,
           ax=axes[0])
pp.pieplot(data=df, x="kind", y="n", palette=palette, ax=axes[1])
pp.show()

# %%
# Wedge order
# -----------
# By default wedges follow the category order of the data — the same
# order a bar panel drawn from the same frame uses — running clockwise
# from twelve o'clock. ``sort=True`` orders them by descending value
# instead, and ``order=`` names the order outright — here the reverse
# of the data order. ``order=`` should
# list every category: one it leaves out is dropped and the rest
# renormalise to a full circle, with a warning.

severity = pd.DataFrame({
    "stage": ["mild", "moderate", "severe", "critical"],
    "n": [20.0, 45.0, 25.0, 10.0],
})
stage_palette = dict(zip(severity["stage"], pp.color_palette("pastel", 4)))

fig, axes = pp.subplots(ncols=4, axes_size=(36, 36))
pp.legend(side="right")
pp.barplot(data=severity, x="stage", y="n", hue="stage",
           palette=stage_palette, ax=axes[0])
axes[0].tick_params(axis="x", labelrotation=45)
pp.pieplot(data=severity, x="stage", y="n", palette=stage_palette,
           ax=axes[1], title="default")
pp.pieplot(data=severity, x="stage", y="n", palette=stage_palette,
           sort=True, ax=axes[2], title="sort=True")
pp.pieplot(data=severity, x="stage", y="n", palette=stage_palette,
           order=["critical", "severe", "moderate", "mild"], ax=axes[3],
           title="order=[...]")
pp.show()

# %%
# Where the first wedge starts, and which way they run
# ------------------------------------------------------
# ``start_angle`` is the angle of the first wedge's leading edge, in
# degrees: 90 (the default) is twelve o'clock, 0 is three o'clock.
# ``direction`` is ``"clockwise"`` (the default) or
# ``"counterclockwise"``.

fig, axes = pp.subplots(ncols=3, axes_size=(36, 36))
pp.legend(side="right")
for ax, (start, direction) in zip(axes, [(90, "clockwise"),
                                         (0, "clockwise"),
                                         (90, "counterclockwise")]):
    pp.pieplot(data=severity, x="stage", y="n", palette=stage_palette,
               start_angle=start, direction=direction, ax=ax,
               title=f"{start}°, {direction}")
pp.show()

# %%
# A donut, with the total in the hole
# -----------------------------------
# ``donut`` is the inner radius as a fraction of the outer one.
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
# Formatting and rotating the labels
# ----------------------------------
# The default label is the wedge's share of the total, as a percentage.
# Passing ``fmt`` as a format spec labels the raw value instead — here
# ``",.0f"`` for counts with a thousands separator. ``rotation`` takes
# degrees, or ``"tangential"`` (along the arc) or ``"radial"`` (along
# the spoke); either is flipped where needed so no label reads upside
# down.

cohort = pd.DataFrame({
    "site": ["north", "east", "south", "west", "central"],
    "patients": [4210.0, 3380.0, 2950.0, 2160.0, 1870.0],
})
fig, axes = pp.subplots(ncols=3, axes_size=(42, 42))
pp.legend(side="right")
for ax, rotation in zip(axes, [0, "tangential", "radial"]):
    pp.pieplot(data=cohort, x="site", y="patients", ax=ax,
               annotate={"fmt": ",.0f", "anchor": "inside",
                         "rotation": rotation},
               title=f"rotation={rotation!r}")
pp.show()

# %%
# Patterns
# --------
# ``hatch`` adds a texture, keyed by a column the same way ``hue`` is.
# Naming the wedge column itself (``hatch="kind"``) gives every wedge
# its own pattern on top of its own colour — useful when the figure may
# be printed in greyscale — and the patterns fold into the colour
# swatches of a single legend entry. Patterns come from
# ``pp.HATCH_PATTERNS``, at the density ``pp.set_hatch_mode`` sets,
# unless ``hatch_map`` names them; the dense ones below stay readable in
# the small legend swatches.

hatches = {"symptom": "", "admin": "////", "lay_dx": "\\\\\\\\",
           "none": "...."}
fig, ax = pp.subplots(axes_size=(50, 50))
pp.pieplot(data=df, x="kind", y="n", palette=palette, hatch="kind",
           hatch_map=hatches, ax=ax)
pp.show()

# %%
# A pattern for a second column
# -----------------------------
# ``hatch`` may instead name a different column, as long as it is
# constant within each wedge. The pattern then carries a second piece of
# information, and gets a legend entry of its own with neutral swatches,
# since colour already means the wedge.

sourced = df.assign(source=["registry", "registry", "survey", "survey"])
fig, ax = pp.subplots(axes_size=(50, 50))
pp.pieplot(data=sourced, x="kind", y="n", palette=palette,
           hatch="source", hatch_map={"registry": "", "survey": "////"},
           ax=ax)
pp.show()

# %%
# Colouring by something other than the category
# -------------------------------------------------
# ``hue`` means colour and nothing else — unlike ``barplot``, where
# ``hue`` also controls how marks group, a wedge chart keeps its
# structure in ``x``. ``hue`` may name a column outside the one that
# defines the wedges, as long as it is constant within each wedge. Here
# four departments are coloured by the wing they belong to, so two
# wedges share each colour; ``hatch="dept"`` gives each department its
# own pattern, and the patterns their own legend entry, so the four stay
# apart.

staff = pd.DataFrame({
    "dept": ["cardio", "neuro", "billing", "records"],
    "wing": ["clinical", "clinical", "admin", "admin"],
    "n": [30.0, 24.0, 18.0, 12.0],
})
fig, ax = pp.subplots(axes_size=(50, 50))
pp.pieplot(data=staff, x="dept", y="n", hue="wing", hatch="dept",
           hatch_map={"cardio": "", "neuro": "////", "billing": "\\\\\\\\",
                      "records": "...."},
           palette={"clinical": "#75b375", "admin": "#8e8ec1"}, ax=ax)
pp.show()

# %%
# Colouring by a number
# ---------------------
# A numeric ``hue`` maps through a colormap instead of a palette, and
# the legend becomes a colorbar. As with a categorical ``hue``, the
# value must be constant within each wedge. The colormap runs from
# near-white up to the theme colour, so this example fills the wedges
# solid (``alpha=1``) to read against the colorbar, and outlines them in
# grey so the lightest wedge keeps its edge.

waits = df.assign(days=[3.0, 11.0, 26.0, 40.0])
fig, ax = pp.subplots(axes_size=(50, 50))
pp.pieplot(data=waits, x="kind", y="n", hue="days", alpha=1,
           edgecolor="0.3", ax=ax)
pp.show()

# %%
# Fill, outline and stroke width
# ------------------------------
# Wedges take the house style by default: a tinted fill behind a
# full-strength outline in the wedge's own colour. ``alpha`` sets the
# fill alone (``0`` leaves outlines only, ``1`` a solid fill);
# ``edgecolor`` outlines every wedge the same way regardless of fill;
# ``linewidth`` sets the outline width.

fig, axes = pp.subplots(ncols=3, axes_size=(36, 36))
# The three panels draw their keys in different styles, so only the
# first panel's is collected; the band still sits past the last panel.
pp.legend(axes=[axes[0]], anchor=axes[-1], side="right")
pp.pieplot(data=df, x="kind", y="n", palette=palette, ax=axes[0],
           title="default")
pp.pieplot(data=df, x="kind", y="n", palette=palette, alpha=0,
           linewidth=1.5, legend=False, ax=axes[1],
           title="alpha=0, linewidth=1.5")
pp.pieplot(data=df, x="kind", y="n", palette=palette, alpha=1,
           edgecolor="white", linewidth=1.5, legend=False, ax=axes[2],
           title='alpha=1, edgecolor="white"')
pp.show()
