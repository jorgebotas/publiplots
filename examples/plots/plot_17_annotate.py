"""
Value Label Annotations
=======================

This example showcases ``pp.annotate`` across every plot type that
supports it. Each ``pp.*plot`` function takes an ``annotate`` kwarg —
pass ``True`` for defaults or a dict to override format, anchor, color,
text kwargs, and (for boxplots/violins) which statistics to label.

Anchor vocabularies are strategy-specific:

- ``pp.barplot``: ``outside`` / ``inside`` / ``base`` / ``center``
- ``pp.pointplot``, ``pp.boxplot``, ``pp.violinplot``:
  ``top`` / ``bottom`` / ``left`` / ``right`` / ``center``
"""

import publiplots as pp
import pandas as pd
import numpy as np

rng = np.random.default_rng(0)

# %%
# Simple: labels outside the bars (default)
# -----------------------------------------
df = pd.DataFrame({
    "category": pd.Categorical(["A", "B", "C", "D"]),
    "value": [1.2, 2.8, 2.1, 3.6],
})

ax = pp.barplot(data=df, x="category", y="value", annotate=True,
                title="annotate=True")
pp.show()

# %%
# Custom format string
# --------------------
ax = pp.barplot(
    data=df, x="category", y="value",
    annotate={"fmt": "{:.1f}%"},
    title="annotate={'fmt': '{:.1f}%'}",
)
pp.show()

# %%
# Anchor positions
# ----------------
fig, axes = pp.subplots(1, 4, axes_size=(40, 40))
for ax, anchor in zip(axes, ("outside", "inside", "base", "center")):
    pp.barplot(data=df, x="category", y="value", ax=ax,
               annotate={"anchor": anchor},
               title=f"anchor='{anchor}'")
pp.show()

# %%
# Hue-colored labels
# ------------------
rows = []
for group in ("A", "B", "C"):
    for cond in ("ctrl", "trt"):
        base = 2 if group == "A" else (3 if group == "B" else 4)
        bump = 0 if cond == "ctrl" else 0.8
        for v in rng.normal(loc=base + bump, scale=0.2, size=10):
            rows.append({"group": group, "cond": cond, "y": float(v)})
grouped = pd.DataFrame(rows)
grouped["group"] = grouped["group"].astype("category")
grouped["cond"] = grouped["cond"].astype("category")

ax = pp.barplot(
    data=grouped, x="group", y="y", hue="cond", errorbar="se",
    annotate={"color": "hue"},
    title='annotate={"color": "hue"}',
)
pp.show()

# %%
# Horizontal orientation
# ----------------------
df_h = df.rename(columns={"category": "c", "value": "v"})
ax = pp.barplot(data=df_h, x="v", y="c", annotate=True, title="horizontal")
pp.show()

# %%
# Horizontal + inside with mixed bar sizes
# ----------------------------------------
# When ``anchor="inside"`` is requested but a bar is too short to fit the
# label, that single label flips to ``"outside"`` automatically. Other bars
# keep their inside anchoring.
df_mixed = pd.DataFrame({
    "c": pd.Categorical(["A", "B", "C", "D", "E"]),
    "v": [0.2, 8.4, 0.5, 12.0, 3.1],
})
ax = pp.barplot(
    data=df_mixed, x="v", y="c",
    annotate={"anchor": "inside", "fmt": ".1f"},
    title='horizontal + anchor="inside" (short bars flip out)',
)
pp.show()

# %%
# Pointplot: mean ± errorbar labels
# ---------------------------------
# pp.pointplot(..., annotate=True) labels each mean at the top of its
# errorbar cap. Anchor vocabulary differs from barplots: point labels are
# directional — ``top`` / ``bottom`` / ``left`` / ``right`` / ``center``.
rows = []
for grp in ("A", "B"):
    for t in ("t1", "t2", "t3"):
        base = {"t1": 1.0, "t2": 2.5, "t3": 3.2}[t] + (0 if grp == "A" else 0.5)
        for v in rng.normal(loc=base, scale=0.3, size=10):
            rows.append({"group": grp, "time": t, "v": float(v)})
point_df = pd.DataFrame(rows)
point_df["time"] = point_df["time"].astype("category")
point_df["group"] = point_df["group"].astype("category")

ax = pp.pointplot(
    data=point_df, x="time", y="v", hue="group",
    errorbar="se", annotate={"fmt": ".2f"},
    title="pp.pointplot(annotate=True)",
)
pp.show()

# %%
# Point anchor variants
# ---------------------
fig, axes = pp.subplots(1, 5, axes_size=(40, 40))
for ax, anchor in zip(axes, ("top", "bottom", "left", "right", "center")):
    pp.pointplot(data=point_df, x="time", y="v", ax=ax,
                 errorbar="se",
                 annotate={"anchor": anchor, "fmt": ".1f"},
                 title=f"anchor='{anchor}'")
pp.show()

# %%
# Boxplot: stats labels
# ---------------------
# pp.boxplot(..., annotate=True) labels the median by default. Pass a
# `stats=[...]` list to label multiple stats per box: any of
# ``median``, ``q1``, ``q3``, ``whisker_low``, ``whisker_high``, ``mean``.
box_rows = []
for g, base in zip(("A", "B", "C"), (1.0, 2.0, 3.0)):
    for v in rng.normal(base, 0.5, 40):
        box_rows.append({"g": g, "y": float(v)})
box_df = pd.DataFrame(box_rows)
box_df["g"] = box_df["g"].astype("category")

ax = pp.boxplot(
    data=box_df, x="g", y="y",
    annotate={"stats": ["median", "q1", "q3"], "fmt": ".2f"},
    title="pp.boxplot(annotate={'stats': [...]})",
)
pp.show()

# %%
# Violinplot: same stats API
# --------------------------
# pp.violinplot shares the box_stats strategy — violins are a different
# visual but the same stats (median, quartiles) apply.
ax = pp.violinplot(
    data=box_df, x="g", y="y",
    annotate={"stats": ["median"], "fmt": ".2f"},
    title="pp.violinplot(annotate=True)",
)
pp.show()
