"""
Value Label Annotations
=======================

This example demonstrates pp.annotate, the helper for labeling plot
marks with their aggregated values. v1 supports barplots; the module is
designed so point_values, box_medians, and other strategies can slot in.

Two entry points:

- ``pp.annotate(ax, kind="bar_values", ...)`` — post-hoc on any axes
- ``pp.barplot(..., annotate=True | dict)`` — the sugar call
"""

import publiplots as pp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# %%
# Simple: labels outside the bars (default)
# -----------------------------------------
df = pd.DataFrame({
    "category": pd.Categorical(["A", "B", "C", "D"]),
    "value": [1.2, 2.8, 2.1, 3.6],
})

fig, ax = pp.barplot(data=df, x="category", y="value", annotate=True)
ax.set_title("annotate=True")
plt.show()

# %%
# Custom format string
# --------------------
fig, ax = pp.barplot(
    data=df, x="category", y="value",
    annotate={"fmt": "{:.1f}%"},
)
ax.set_title("annotate={'fmt': '{:.1f}%'}")
plt.show()

# %%
# Anchor positions
# ----------------
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for ax, anchor in zip(axes, ("outside", "inside", "base", "center")):
    pp.barplot(data=df, x="category", y="value", ax=ax,
               annotate={"anchor": anchor})
    ax.set_title(f"anchor='{anchor}'")
plt.show()

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

fig, ax = pp.barplot(
    data=grouped, x="group", y="y", hue="cond", errorbar="se",
    annotate={"color": "hue"},
)
ax.set_title('annotate={"color": "hue"}')
plt.show()

# %%
# Horizontal orientation
# ----------------------
df_h = df.rename(columns={"category": "c", "value": "v"})
fig, ax = pp.barplot(data=df_h, x="v", y="c", annotate=True)
ax.set_title("horizontal")
plt.show()

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
fig, ax = pp.barplot(
    data=df_mixed, x="v", y="c",
    annotate={"anchor": "inside", "fmt": ".1f"},
)
ax.set_title('horizontal + anchor="inside" (short bars flip out)')
plt.show()

# %%
# Post-hoc on a foreign axes
# --------------------------
# pp.annotate works on any Axes with bars on it, not just pp.barplot output.
fig, ax = plt.subplots()
ax.bar([0, 1, 2], [1.0, 2.4, 0.7])
pp.annotate(ax, kind="bar_values", fmt=".1f")
ax.set_title("foreign ax + pp.annotate")
plt.show()

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

fig, ax = pp.pointplot(
    data=point_df, x="time", y="v", hue="group",
    errorbar="se", annotate={"fmt": ".2f"},
)
ax.set_title("pp.pointplot(annotate=True)")
plt.show()

# %%
# Point anchor variants
# ---------------------
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for ax, anchor in zip(axes, ("top", "bottom", "left", "right", "center")):
    pp.pointplot(data=point_df, x="time", y="v", ax=ax,
                 errorbar="se",
                 annotate={"anchor": anchor, "fmt": ".1f"})
    ax.set_title(f"anchor='{anchor}'")
plt.show()
