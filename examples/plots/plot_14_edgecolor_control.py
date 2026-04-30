"""
Edge Color Control
==================

This example demonstrates how to control edge color across all plot types,
either globally through ``pp.rcParams['edgecolor']`` or per-call via the
``edgecolor=`` argument.

By default, ``pp.rcParams['edgecolor']`` is ``None``, which means each plot
picks its own edge automatically (typically the face/palette color). Setting
it once applies uniform edges across every plot in the figure.
"""

import publiplots as pp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set style
pp.set_notebook_style()

# %%
# Default Behavior: Automatic Edges
# ---------------------------------
# With the default rcParam (``None``), edges match the face color. Each bar's
# outline follows its palette entry.

np.random.seed(7)
bar_data = pd.DataFrame({
    'group': np.repeat(['A', 'B', 'C', 'D'], 12),
    'value': np.concatenate([
        np.random.normal(45, 7, 12),
        np.random.normal(60, 8, 12),
        np.random.normal(75, 9, 12),
        np.random.normal(90, 10, 12),
    ])
})

fig, ax = plt.subplots(figsize=(5, 3))
pp.barplot(
    data=bar_data,
    x='group',
    y='value',
    hue='group',
    palette='pastel',
    errorbar='se',
    title='Default: edges match palette',
    ax=ax,
)
plt.tight_layout()
plt.show()

# %%
# Global Override via rcParams
# ----------------------------
# Set ``pp.rcParams['edgecolor']`` to apply a single edge color across every
# plot in the script. This is the recommended way to enforce a consistent
# publication style without repeating ``edgecolor=`` on every call.

pp.rcParams['edgecolor'] = 'black'

fig, ax = plt.subplots(figsize=(5, 3))
pp.barplot(
    data=bar_data,
    x='group',
    y='value',
    hue='group',
    palette='pastel',
    errorbar='se',
    title="rcParams['edgecolor'] = 'black'",
    ax=ax,
)
plt.tight_layout()
plt.show()

# %%
# Uniform Edges Across Plot Types
# -------------------------------
# The same rcParam applies to every publiplots function: bars, boxes, violins,
# scatter, strip/swarm, point, raincloud, and dot-heatmaps.

np.random.seed(11)
mixed_data = pd.DataFrame({
    'group': np.repeat(['A', 'B', 'C'], 30),
    'value': np.concatenate([
        np.random.normal(50, 8, 30),
        np.random.normal(65, 9, 30),
        np.random.normal(80, 10, 30),
    ]),
})

fig, axes = plt.subplots(1, 3, figsize=(12, 3))
pp.barplot(
    data=mixed_data, x='group', y='value', hue='group',
    palette='pastel', errorbar='se', title='Bar', ax=axes[0],
)
pp.boxplot(
    data=mixed_data, x='group', y='value', hue='group',
    palette='pastel', title='Box', ax=axes[1],
)
pp.scatterplot(
    data=mixed_data, x='group', y='value', hue='group',
    palette='pastel', title='Scatter', ax=axes[2],
)
plt.tight_layout()
plt.show()

# %%
# Per-Call Override
# -----------------
# A per-call ``edgecolor=`` argument always wins over the rcParam. Use this
# to tweak a single plot without resetting the global default.

fig, axes = plt.subplots(1, 2, figsize=(9, 3))
pp.barplot(
    data=bar_data,
    x='group',
    y='value',
    hue='group',
    palette='pastel',
    errorbar='se',
    title="rcParam (black) inherited",
    ax=axes[0],
)
pp.barplot(
    data=bar_data,
    x='group',
    y='value',
    hue='group',
    palette='pastel',
    errorbar='se',
    edgecolor='#c0392b',
    title="edgecolor='#c0392b' (per-call)",
    ax=axes[1],
)
plt.tight_layout()
plt.show()

# %%
# Restoring the Default
# ---------------------
# Set the rcParam back to ``None`` to return to per-plot automatic edges.

pp.rcParams['edgecolor'] = None

fig, ax = plt.subplots(figsize=(5, 3))
pp.barplot(
    data=bar_data,
    x='group',
    y='value',
    hue='group',
    palette='pastel',
    errorbar='se',
    title="Back to default (None = auto)",
    ax=ax,
)
plt.tight_layout()
plt.show()
