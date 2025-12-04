"""
Bar Plot Examples
=================

This example demonstrates the various bar plot capabilities in PubliPlots,
including simple bars, grouped bars, error bars, and hatch patterns.
"""

import publiplots as pp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set style
pp.set_notebook_style()

# %%
# Simple Bar Plot
# ---------------
# Create a basic bar plot with categorical data.

# Create sample data
simple_data = pd.DataFrame({
    'category': ['A', 'B', 'C', 'D', 'E'],
    'value': [23, 45, 38, 52, 41]
})

# Create simple bar plot
fig, ax = pp.barplot(
    data=simple_data,
    x='category',
    y='value',
    title='Simple Bar Plot',
    xlabel='Category',
    ylabel='Value',
    palette='pastel',
)
plt.show()

# %%
# Bar Plot with Error Bars
# -------------------------
# Show mean ± standard error with multiple measurements per category.

# Create data with multiple measurements per category
np.random.seed(42)
error_data = pd.DataFrame({
    'treatment': np.repeat(['Control', 'Drug A', 'Drug B', 'Drug C'], 12),
    'response': np.concatenate([
        np.random.normal(100, 15, 12),  # Control
        np.random.normal(120, 12, 12),  # Drug A
        np.random.normal(135, 18, 12),  # Drug B
        np.random.normal(110, 14, 12),  # Drug C
    ])
})

# Create bar plot with error bars
fig, ax = pp.barplot(
    data=error_data,
    x='treatment',
    y='response',
    title='Drug Response with Standard Error',
    xlabel='Treatment',
    ylabel='Response (a.u.)',
    errorbar='se',  # standard error
    capsize=0.1,
    palette='pastel',
)
plt.show()

# %%
# Grouped Bar Plot with Hue
# --------------------------
# Use the hue parameter to create grouped bars split by a categorical variable.

# Create grouped data
np.random.seed(123)
hue_data = pd.DataFrame({
    'time': np.repeat(['Day 1', 'Day 2', 'Day 3', 'Day 4'], 20),
    'group': np.tile(np.repeat(['Control', 'Treated'], 10), 4),
    'measurement': np.concatenate([
        # Day 1
        np.random.normal(50, 8, 10),   # Control
        np.random.normal(52, 8, 10),   # Treated
        # Day 2
        np.random.normal(52, 9, 10),   # Control
        np.random.normal(65, 10, 10),  # Treated
        # Day 3
        np.random.normal(54, 9, 10),   # Control
        np.random.normal(78, 12, 10),  # Treated
        # Day 4
        np.random.normal(55, 10, 10),  # Control
        np.random.normal(85, 14, 10),  # Treated
    ])
})

# Create grouped bar plot
fig, ax = pp.barplot(
    data=hue_data,
    x='time',
    y='measurement',
    hue='group',
    title='Time Course: Control vs Treated',
    xlabel='Time Point',
    ylabel='Measurement',
    errorbar='se',
    palette="RdGyBu_r",
)
plt.show()

# %%
# Bar Plot with Hatch Patterns Only
# ----------------------------------
# Use hatch patterns without color grouping for black-and-white publications.

# Create data for hatch-only plot
np.random.seed(456)
hatch_only_data = pd.DataFrame({
    'condition': np.repeat(['Low', 'Medium', 'High'], 15),
    'intensity': np.concatenate([
        np.random.normal(30, 5, 15),
        np.random.normal(60, 8, 15),
        np.random.normal(90, 10, 15),
    ])
})

# Create bar plot with hatch patterns (no hue)
fig, ax = pp.barplot(
    data=hatch_only_data,
    x='condition',
    y='intensity',
    hatch='condition',
    title='Intensity by Condition (Hatch Patterns Only)',
    xlabel='Condition',
    ylabel='Intensity',
    errorbar='se',
    color='#5D83C3',
    hatch_map={'Low': '', 'Medium': '//', 'High': 'xx'},
    alpha=0.0,
)
plt.show()

# %%
# Bar Plot with Hue and Hatch (Double Split)
# -------------------------------------------
# Combine color grouping (hue) and pattern differentiation (hatch) for
# visualizing data with two categorical grouping variables.

# Create data with both hue and hatch
np.random.seed(789)
double_split_data = pd.DataFrame({
    "cell_type": np.repeat(["TypeA", "TypeB", "TypeC"], 40),
    "treatment": np.tile(np.repeat(["Vehicle", "Drug"], 20), 3),
    "time": np.tile(np.repeat(["24h", "48h"], 10), 6),
    "viability": np.concatenate([
        # TypeA
        np.random.normal(95, 5, 10),   # Vehicle, 24h
        np.random.normal(93, 5, 10),   # Vehicle, 48h
        np.random.normal(75, 8, 10),   # Drug, 24h
        np.random.normal(60, 10, 10),  # Drug, 48h
        # TypeB
        np.random.normal(94, 5, 10),   # Vehicle, 24h
        np.random.normal(92, 5, 10),   # Vehicle, 48h
        np.random.normal(80, 8, 10),   # Drug, 24h
        np.random.normal(70, 9, 10),   # Drug, 48h
        # TypeC
        np.random.normal(96, 4, 10),   # Vehicle, 24h
        np.random.normal(95, 4, 10),   # Vehicle, 48h
        np.random.normal(85, 7, 10),   # Drug, 24h
        np.random.normal(78, 8, 10),   # Drug, 48h
    ])
})

# Create bar plot with both hue and hatch
fig, ax = pp.barplot(
    data=double_split_data,
    x="cell_type",
    y="viability",
    hue="treatment",
    hatch="time",
    title="Cell Viability: Treatment × Time × Cell Type",
    xlabel="Cell Type",
    ylabel="Viability (%)",
    errorbar="se",
    palette={"Vehicle": "#8E8EC1", "Drug": "#60a8a8"},
    hatch_map={"24h": "", "48h": "///"},
    figsize=(8, 5)
)
plt.show()

# %%
# Horizontal Bar Plot
# -------------------
# Create horizontal bars by swapping x and y axes.

# Create data for horizontal bar plot
np.random.seed(111)
horizontal_data = pd.DataFrame({
    'gene': ['Gene A', 'Gene B', 'Gene C', 'Gene D', 'Gene E', 'Gene F'],
    'expression': np.random.uniform(50, 200, 6),
    'group': ['Upregulated', 'Upregulated', 'Downregulated',
              'Upregulated', 'Downregulated', 'Upregulated']
})

# Create horizontal bar plot
fig, ax = pp.barplot(
    data=horizontal_data,
    x='expression',
    y='gene',
    hue='group',
    title='Gene Expression Levels (Horizontal)',
    xlabel='Expression Level',
    ylabel='Gene',
    palette="RdGyBu",
    errorbar=None,
    alpha=0.3,
    order=horizontal_data['gene'].tolist()
)
plt.show()
