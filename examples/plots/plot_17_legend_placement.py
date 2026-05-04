"""
Legend Placement
================

publiplots offers three complementary legend-placement knobs:

1. **Per-axis inside**: ``legend_kws={'inside': True, 'loc': 'upper right'}``
   drops the legend inside the axes using matplotlib's corner-based
   placement. No reactor, no layout reservation — just a local legend.

2. **Figure-anchored group**: ``pp.legend_group(side=...)`` (no anchor)
   spans the full subplot grid on the chosen side. The figure grows on
   that side to accommodate the legend; panel sizes stay inviolate.

3. **Axes-anchored group**: ``pp.legend_group(anchor=axes[r,c], side=...)``
   pins the band to a single cell. The corresponding per-cell reservation
   (``right[c]`` for ``side='right'``, ``xlabel_space[r]`` for
   ``side='bottom'``, ...) absorbs the band width, pushing just that
   axes' column/row larger.

``pp.legend_group`` also chooses its **orientation** and **alignment**
per side by default:

- ``side='top'`` / ``'bottom'`` → ``orientation='horizontal'`` (entries
  run along the edge, ``ncol`` defaults to ``len(handles)``) and
  ``align='center'`` (centered along the anchor edge).
- ``side='left'`` / ``'right'`` → ``orientation='vertical'`` and
  ``align='start'`` (legend begins at the anchor's top corner).

Override either with ``orientation='vertical'|'horizontal'`` or
``align='start'|'center'|'end'`` when the default isn't what you want.

This gallery walks through each mode.
"""

import publiplots as pp
import pandas as pd
import numpy as np

# Shared fixture --------------------------------------------------------------

np.random.seed(42)
_df = pd.DataFrame({
    "x": np.random.randn(240),
    "y": np.random.randn(240),
    "group": np.tile(["Control", "Low", "High"], 80),
    "panel": np.repeat(["A", "B", "C", "D"], 60),
})

# %%
# 1. Default outside-right (no legend_group)
# ------------------------------------------
# A single ``pp.scatterplot`` with a categorical ``hue`` renders its legend
# just past the axes' right edge — the publication-ready default.

pp.scatterplot(
    data=_df, x="x", y="y", hue="group", palette="pastel",
    title="Default outside-right",
)
pp.show()

# %%
# 2. Per-axis inside legend
# -------------------------
# ``legend_kws={'inside': True, 'loc': ...}`` keeps the legend local to
# the axes. Useful when vertical real estate is precious or the data
# leaves a natural empty corner.

pp.scatterplot(
    data=_df, x="x", y="y", hue="group", palette="pastel",
    legend_kws={"inside": True, "loc": "upper right"},
    title='legend_kws={"inside": True, "loc": "upper right"}',
)
pp.show()

# %%
# 3. Figure-anchored ``side='right'`` on a 2×2 grid
# -------------------------------------------------
# ``pp.legend_group()`` without an ``anchor=`` spans the full figure
# vertically, tucked past the rightmost column. Auto-collects every
# stashed entry from every panel; dedupes by name.

fig, axes = pp.subplots(2, 2, axes_size=(35, 30))
pp.legend_group(side="right")
for (r, c), panel in zip([(0, 0), (0, 1), (1, 0), (1, 1)], "ABCD"):
    pp.scatterplot(
        data=_df[_df["panel"] == panel], x="x", y="y",
        hue="group", palette="pastel",
        title=f"Panel {panel}", ax=axes[r, c],
    )
pp.show()

# %%
# 4. Figure-anchored ``side='bottom'`` on a 2×2 grid
# --------------------------------------------------
# When panels leave spare vertical headroom, a bottom-anchored legend
# uses that space instead of the figure's right column. Same call, just
# ``side='bottom'``. The default orientation flips to ``horizontal``
# (entries run along the edge) and ``align='center'`` keeps the block
# balanced under the grid.

fig, axes = pp.subplots(2, 2, axes_size=(35, 30))
pp.legend_group(side="bottom")
for (r, c), panel in zip([(0, 0), (0, 1), (1, 0), (1, 1)], "ABCD"):
    pp.scatterplot(
        data=_df[_df["panel"] == panel], x="x", y="y",
        hue="group", palette="pastel",
        title=f"Panel {panel}", ax=axes[r, c],
    )
pp.show()

# %%
# 4b. Bottom with ``align='end'``
# -------------------------------
# Override the default center-alignment to pin the legend to the right
# edge of the grid — useful when the legend should align with a
# particular panel column rather than the figure midline.

fig, axes = pp.subplots(2, 2, axes_size=(35, 30))
pp.legend_group(side="bottom", align="end")
for (r, c), panel in zip([(0, 0), (0, 1), (1, 0), (1, 1)], "ABCD"):
    pp.scatterplot(
        data=_df[_df["panel"] == panel], x="x", y="y",
        hue="group", palette="pastel",
        title=f"Panel {panel}", ax=axes[r, c],
    )
pp.show()

# %%
# 5. Figure-anchored ``side='top'`` + ``side='left'``
# ---------------------------------------------------
# All four sides are supported. Top/left are less common but useful for
# figures with a strong vertical hierarchy or right-to-left reading order.

fig, axes = pp.subplots(2, 2, axes_size=(35, 30))
pp.legend_group(side="top")
for (r, c), panel in zip([(0, 0), (0, 1), (1, 0), (1, 1)], "ABCD"):
    pp.scatterplot(
        data=_df[_df["panel"] == panel], x="x", y="y",
        hue="group", palette="pastel",
        title=f"Panel {panel}", ax=axes[r, c],
    )
pp.show()

fig, axes = pp.subplots(2, 2, axes_size=(35, 30))
pp.legend_group(side="left")
for (r, c), panel in zip([(0, 0), (0, 1), (1, 0), (1, 1)], "ABCD"):
    pp.scatterplot(
        data=_df[_df["panel"] == panel], x="x", y="y",
        hue="group", palette="pastel",
        title=f"Panel {panel}", ax=axes[r, c],
    )
pp.show()

# %%
# 6. Axes-anchored: pin the band to a single cell
# -----------------------------------------------
# Pass ``anchor=axes[r, c]`` to pin the band to one cell. The
# corresponding per-cell reservation (the column's ``right`` for
# ``side='right'``, the row's ``xlabel_space`` for ``side='bottom'``,
# etc.) absorbs the band width. Useful when one panel deserves its own
# annotation band without growing the rest of the figure.

fig, axes = pp.subplots(2, 2, axes_size=(35, 30))
# Anchor the band to the top-right panel only.
pp.legend_group(anchor=axes[0, 1], side="right")
for (r, c), panel in zip([(0, 0), (0, 1), (1, 0), (1, 1)], "ABCD"):
    pp.scatterplot(
        data=_df[_df["panel"] == panel], x="x", y="y",
        hue="group", palette="pastel",
        title=f"Panel {panel}", ax=axes[r, c],
    )
pp.show()

# %%
# 7. Combining inside + figure-anchored group
# -------------------------------------------
# The two modes compose: collect one shared dimension into the
# figure-level group (here ``group``) and render other per-panel
# legends inside each axes. ``collect=['group']`` filters the group's
# auto-collect pass; ``legend_kws={'inside': True}`` on each scatter
# renders the non-collected style (``replicate`` below) as a local
# legend in each panel.

rng = np.random.default_rng(7)
split_df = pd.DataFrame({
    "x": rng.normal(size=240),
    "y": rng.normal(size=240),
    "group": np.tile(["Control", "Low", "High"], 80),
    "replicate": np.tile(["R1", "R2"], 120),
    "panel": np.repeat(["A", "B", "C", "D"], 60),
})

fig, axes = pp.subplots(2, 2, axes_size=(35, 30))
pp.legend_group(side="bottom", collect=["group"])
for (r, c), panel in zip([(0, 0), (0, 1), (1, 0), (1, 1)], "ABCD"):
    pp.scatterplot(
        data=split_df[split_df["panel"] == panel], x="x", y="y",
        hue="group", style="replicate", palette="pastel",
        title=f"Panel {panel}", ax=axes[r, c],
        legend_kws={"inside": True, "loc": "upper right"},
    )
pp.show()
