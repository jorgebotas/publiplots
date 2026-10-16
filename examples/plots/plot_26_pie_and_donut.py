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

# Not the default assignment: each category gets a colour the default
# palette would have given another one.
palette = {"symptom": "#8e8ec1", "admin": "#eeaa58",
           "lay_dx": "#e67e7e", "none": "#75b375"}

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
# Here the bars break each category down by year, and the pie shows the
# two years pooled. Both panels key colour on ``kind``, and each stashes
# a ``kind`` colour entry; a figure-level ``pp.legend`` band collects
# both and merges them by name, so ``kind`` is listed once, for both
# panels. Merging is by name, not by colour: where two panels draw the
# same label differently, the band keeps the first panel's swatch and,
# for a different fill or stroke width, warns. The shared ``palette``
# is what makes the merged key true of both panels.
#
# (A bar panel whose ``x`` is ``kind`` itself, as above, labels its
# categories on the axis and stashes no colour entry, so it has nothing
# to merge.)

yearly = pd.DataFrame({
    "year": ["2023"] * 4 + ["2024"] * 4,
    "kind": ["symptom", "admin", "lay_dx", "none"] * 2,
    "n": [24.1, 20.3, 3.2, 1.9, 28.1, 21.5, 2.8, 2.1],
})
fig, axes = pp.subplots(ncols=2, axes_size=(50, 45))
pp.legend(side="right")
pp.barplot(data=yearly, x="year", y="n", hue="kind", palette=palette,
           ax=axes[0])
pp.pieplot(data=yearly, x="kind", y="n", palette=palette, ax=axes[1])
pp.show()

# %%
# Counting rows
# -------------
# With ``y`` omitted, each wedge is the number of rows in its category,
# so a pie can be drawn straight from one-row-per-observation data.
# ``fmt=",.0f"`` labels the counts rather than the shares.

visits = pd.DataFrame({
    "kind": ["symptom"] * 12 + ["admin"] * 9 + ["lay_dx"] * 5
            + ["none"] * 4,
})
fig, ax = pp.subplots(axes_size=(50, 50))
pp.pieplot(data=visits, x="kind", palette=palette, ax=ax,
           annotate={"fmt": ",.0f"})
pp.show()

# %%
# A palette by name
# -----------------
# ``palette`` also takes a seaborn or publiplots palette name, assigned
# in category order.

fig, ax = pp.subplots(axes_size=(50, 50))
pp.pieplot(data=df, x="kind", y="n", palette="Set2", ax=ax)
pp.show()

# %%
# Wedge order
# -----------
# By default wedges follow the category order of the data — the same
# order a bar panel drawn from the same frame uses — running clockwise
# from twelve o'clock. ``sort=True`` orders them by descending value
# instead, and ``order=`` names the order outright. ``order=`` should
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
           palette=stage_palette, ax=axes[0], title="bars")
for label in axes[0].get_xticklabels():
    label.set(rotation=45, ha="right", rotation_mode="anchor")
pp.pieplot(data=severity, x="stage", y="n", palette=stage_palette,
           ax=axes[1], title="default")
pp.pieplot(data=severity, x="stage", y="n", palette=stage_palette,
           sort=True, ax=axes[2], title="sort=True")
pp.pieplot(data=severity, x="stage", y="n", palette=stage_palette,
           order=["severe", "mild", "critical", "moderate"], ax=axes[3],
           title="order=[...]")
pp.show()

# %%
# Legend order
# ------------
# ``hue_order`` reorders the legend's entries and nothing else: the
# wedges stay where the data order puts them, and every wedge keeps its
# colour.

fig, axes = pp.subplots(ncols=2, axes_size=(36, 36))
pp.pieplot(data=severity, x="stage", y="n", palette=stage_palette,
           ax=axes[0], title="default")
pp.pieplot(data=severity, x="stage", y="n", palette=stage_palette,
           hue_order=["critical", "severe", "moderate", "mild"],
           ax=axes[1], title="hue_order=[...]")
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
# Pulling wedges out
# ------------------
# ``pull`` moves a wedge out along its own mid-ray, by a fraction of the
# radius. A dict keyed by category pulls just those wedges — it is never
# positional — which is what ``pull`` is for: popping one wedge out.
# A single float pulls every wedge the same distance, but the sideways
# gap that opens between two neighbours is that distance times the sine
# of their half-spans, so wide wedges part widely and thin ones barely
# at all. To separate every wedge evenly, use ``gap``: a width in
# millimetres that every pair of neighbours keeps along the whole of
# the edge they share, thin wedges included, in a pie or a donut.

fig, axes = pp.subplots(ncols=4, axes_size=(36, 36))
pp.legend(side="right")
pp.pieplot(data=df, x="kind", y="n", palette=palette,
           pull={"lay_dx": 0.15}, ax=axes[0],
           title='pull={"lay_dx": 0.15}')
pp.pieplot(data=df, x="kind", y="n", palette=palette, pull=0.05,
           ax=axes[1], title="pull=0.05")
pp.pieplot(data=df, x="kind", y="n", palette=palette, gap=1.0,
           ax=axes[2], title="gap=1.0")
pp.pieplot(data=df, x="kind", y="n", palette=palette, gap=1.0,
           donut=0.55, ax=axes[3], title="gap=1.0, donut=0.55")
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
pp.pieplot(data=df, x="kind", y="n", palette=palette, min_share=0.1,
           ax=ax, annotate=True)
pp.show()

# %%
# Labels inside, outside, or chosen per wedge
# ---------------------------------------------
# ``anchor="auto"`` (the default) measures each label against the room
# its own wedge has, and moves it outside with a leader line when it
# does not fit — the thin slivers below. ``"inside"`` and ``"outside"``
# force every label to one placement regardless of fit. The disc is the
# same size under all three: a label that leaves it extends past the
# axes rather than shrinking the pie. Outside labels are not moved apart
# from one another, so those of adjacent thin wedges — ``"6.0%"`` and
# ``"2.0%"`` here — can overlap.

thin = pd.DataFrame({"kind": list("ABCD"), "n": [80.0, 12.0, 6.0, 2.0]})
fig, axes = pp.subplots(ncols=3, axes_size=(38, 38))
for ax, anchor in zip(axes, ["inside", "outside", "auto"]):
    pp.pieplot(data=thin, x="kind", y="n", legend=False, ax=ax)
    pp.annotate(ax, "wedge_values", anchor=anchor)
    ax.set_title(anchor)
pp.show()

# %%
# How far outside labels sit
# --------------------------
# ``offset`` is the gap from the rim to an outside label, in
# millimetres (``1`` by default), with the leader line spanning it.
# Like tick labels, outside labels sit beyond the axes, so a larger
# offset leaves the disc as it is and grows the figure instead.

arms = pd.DataFrame({"arm": ["placebo", "low", "high"],
                     "n": [35.0, 40.0, 25.0]})
fig, axes = pp.subplots(ncols=2, axes_size=(38, 38))
for ax, offset in zip(axes, [1, 4]):
    pp.pieplot(data=arms, x="arm", y="n", legend=False, ax=ax,
               annotate={"anchor": "outside", "offset": offset},
               title=f"offset={offset}")
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
# the small legend swatches. Pattern strokes are drawn at the outline
# width, ``linewidth`` (see the last section).

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
# A numeric ``hue`` maps through a colormap, and the legend becomes a
# colorbar. As with a categorical ``hue``, the value must be constant
# within each wedge. ``palette`` names the colormap (``"viridis"``, or a
# matplotlib ``Colormap``); a dict or list raises, since a number has no
# discrete levels to assign colours to. Left unset, the colormap is a
# light ramp running from near-white up to ``color``. The colorbar shows
# the colormap at full strength, so both panels fill their wedges solid
# (``alpha=1``) to read against it. The ramp's lightest wedge would all
# but vanish into the page, so the left panel outlines its wedges in
# grey; viridis starts dark and needs no outline.

waits = df.assign(days=[3.0, 11.0, 26.0, 40.0])
fig, axes = pp.subplots(ncols=2, axes_size=(42, 42))
pp.pieplot(data=waits, x="kind", y="n", hue="days", color="#e67e7e",
           alpha=1, edgecolor="0.3", ax=axes[0], title='color="#e67e7e"')
pp.pieplot(data=waits, x="kind", y="n", hue="days", palette="viridis",
           alpha=1, ax=axes[1], title='palette="viridis"')
pp.show()

# %%
# Fill, outline and stroke width
# ------------------------------
# Wedges take the house style by default: a tinted fill behind a
# full-strength outline in the wedge's own colour. ``alpha`` sets the
# fill alone (``0`` leaves outlines only, ``1`` a solid fill);
# ``edgecolor`` outlines every wedge the same way regardless of fill;
# ``linewidth`` sets the outline width and the width of the pattern
# strokes too, in the wedges and their legend swatches alike: the key
# beside the hatched panel draws its patterns at 1.5.

fig, axes = pp.subplots(ncols=4, axes_size=(36, 36))
# The key is the hatched panel's, beside it, with its patterns at its
# stroke width. The colours are the same in every panel; only the
# hatched panel's key is collected because the panels draw their
# swatches in different styles.
pp.legend(axes=[axes[3]], anchor=axes[-1], side="right")
pp.pieplot(data=df, x="kind", y="n", palette=palette, legend=False,
           ax=axes[0], title="default")
pp.pieplot(data=df, x="kind", y="n", palette=palette, alpha=0,
           linewidth=1.5, legend=False, ax=axes[1],
           title="alpha=0, linewidth=1.5")
pp.pieplot(data=df, x="kind", y="n", palette=palette, alpha=1,
           edgecolor="white", linewidth=1.5, legend=False, ax=axes[2],
           title='alpha=1, linewidth=1.5\nedgecolor="white"')
pp.pieplot(data=df, x="kind", y="n", palette=palette, hatch="kind",
           hatch_map=hatches, linewidth=1.5, ax=axes[3],
           title="hatch, linewidth=1.5")
pp.show()
