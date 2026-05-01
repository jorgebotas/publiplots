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
# Post-hoc on a foreign axes
# --------------------------
# pp.annotate works on any Axes with bars on it, not just pp.barplot output.
fig, ax = plt.subplots()
ax.bar([0, 1, 2], [1.0, 2.4, 0.7])
pp.annotate(ax, kind="bar_values", fmt=".1f")
ax.set_title("foreign ax + pp.annotate")
plt.show()
