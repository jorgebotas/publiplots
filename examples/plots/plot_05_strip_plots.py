"""
Strip Plot Examples
===================

This example demonstrates strip plot functionality in PubliPlots,
which shows individual data points with optional jitter.
"""

import publiplots as pp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
# Simple Strip Plot
# -----------------
# Basic strip plot showing individual data points.

# Create sample data
np.random.seed(42)
n = 120
strip_data = pd.DataFrame({
    'category': np.repeat(['A', 'B', 'C'], n // 3),
    'value': np.concatenate([
        np.random.normal(10, 2, n // 3),
        np.random.normal(15, 3, n // 3),
        np.random.normal(12, 2.5, n // 3)
    ])
})

# Create simple strip plot
fig, ax = pp.stripplot(
    data=strip_data,
    x='category',
    y='value',
    title='Simple Strip Plot',
    xlabel='Category',
    ylabel='Value',
)
plt.show()

# %%
# Strip Plot with Jitter
# ----------------------
# Use jitter to spread points horizontally.

# Create strip plot with jitter
fig, ax = pp.stripplot(
    data=strip_data,
    x='category',
    y='value',
    jitter=0.2,
    title='Strip Plot with Jitter',
    xlabel='Category',
    ylabel='Value',
)
plt.show()

# %%
# Strip Plot with Hue Grouping
# ----------------------------
# Use the hue parameter to color points by group.

# Add group variable
strip_data['group'] = np.tile(['Group 1', 'Group 2'], n // 2)

# Create strip plot with hue
fig, ax = pp.stripplot(
    data=strip_data,
    x='category',
    y='value',
    hue='group',
    title='Strip Plot with Hue',
    xlabel='Category',
    ylabel='Value',
    palette={'Group 1': '#8E8EC1', 'Group 2': '#75B375'},
)
plt.show()

# %%
# Dodged Strip Plot
# -----------------
# Separate points by hue along the categorical axis.

# Create dodged strip plot
fig, ax = pp.stripplot(
    data=strip_data,
    x='category',
    y='value',
    hue='group',
    dodge=True,
    title='Dodged Strip Plot',
    xlabel='Category',
    ylabel='Value',
)
plt.show()

# %%
# Horizontal Strip Plot
# ---------------------
# Create horizontal strip plot by swapping x and y.

# Create horizontal strip plot
fig, ax = pp.stripplot(
    data=strip_data,
    x='value',
    y='category',
    title='Horizontal Strip Plot',
    xlabel='Value',
    ylabel='Category',
)
plt.show()
