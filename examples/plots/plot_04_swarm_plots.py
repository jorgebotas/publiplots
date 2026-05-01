"""
Swarm Plot Examples
===================

This example demonstrates swarm plot functionality in PubliPlots,
which shows individual data points with minimal overlap.
"""

import publiplots as pp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
# Simple Swarm Plot
# -----------------
# Basic swarm plot showing individual data points.

# Create sample data
np.random.seed(42)
n = 120
swarm_data = pd.DataFrame({
    'category': np.repeat(['A', 'B', 'C'], n // 3),
    'value': np.concatenate([
        np.random.normal(10, 2, n // 3),
        np.random.normal(15, 3, n // 3),
        np.random.normal(12, 2.5, n // 3)
    ])
})

# Create simple swarm plot
fig, ax = pp.swarmplot(
    data=swarm_data,
    x='category',
    y='value',
    title='Simple Swarm Plot',
    xlabel='Category',
    ylabel='Value',
)
plt.show()

# %%
# Swarm Plot with Hue Grouping
# ----------------------------
# Use the hue parameter to color points by group.

# Add group variable
swarm_data['group'] = np.tile(['Group 1', 'Group 2'], n // 2)

# Create swarm plot with hue
fig, ax = pp.swarmplot(
    data=swarm_data,
    x='category',
    y='value',
    hue='group',
    title='Swarm Plot with Hue',
    xlabel='Category',
    ylabel='Value',
    palette={'Group 1': '#8E8EC1', 'Group 2': '#75B375'},
)
plt.show()

# %%
# Dodged Swarm Plot
# -----------------
# Separate points by hue along the categorical axis.

# Create dodged swarm plot
fig, ax = pp.swarmplot(
    data=swarm_data,
    x='category',
    y='value',
    hue='group',
    dodge=True,
    title='Dodged Swarm Plot',
    xlabel='Category',
    ylabel='Value',
)
plt.show()

# %%
# Swarm Plot with Custom Size
# ---------------------------
# Adjust marker size for different data densities.

# Create swarm plot with custom size
fig, ax = pp.swarmplot(
    data=swarm_data,
    x='category',
    y='value',
    size=8,
    title='Swarm Plot with Larger Markers',
    xlabel='Category',
    ylabel='Value',
)
plt.show()

# %%
# Horizontal Swarm Plot
# ---------------------
# Create horizontal swarm plot by swapping x and y.

# Create horizontal swarm plot
fig, ax = pp.swarmplot(
    data=swarm_data,
    x='value',
    y='category',
    title='Horizontal Swarm Plot',
    xlabel='Value',
    ylabel='Category',
)
plt.show()
