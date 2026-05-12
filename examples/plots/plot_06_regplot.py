"""
Regplot Examples
================

:func:`publiplots.regplot` wraps :func:`seaborn.regplot` with publiplots
styling and adds a crucial missing feature: categorical ``hue=`` support.
Seaborn's native ``regplot`` does not accept ``hue``; users historically
had to reach for ``sns.lmplot`` (which owns its figure) to get per-group
fits. ``pp.regplot`` loops over hue levels and calls ``sns.regplot`` per
group onto a shared axes, preserving compatibility with
:func:`publiplots.subplots` and the figure/axes layout pipeline.
"""

import numpy as np
import pandas as pd

import publiplots as pp
from publiplots.plot.regplot import regplot

# %%
# Basic Linear Fit
# ----------------
# The un-hued case mirrors :func:`seaborn.regplot` but routes through the
# publiplots styling pipeline: linewidth, alpha, and edgecolor defaults
# come from ``pp.rcParams``, and the figure is sized from
# ``pp.subplots(axes_size=...)`` rather than a ``figsize=`` kwarg.

rng = np.random.default_rng(0)
n = 120
x = rng.normal(size=n)
y = 1.2 * x + 0.25 * rng.normal(size=n)
linear_df = pd.DataFrame({"x": x, "y": y})

ax = regplot(
    data=linear_df,
    x="x",
    y="y",
    title="Linear Fit",
    xlabel="x",
    ylabel="y",
)
pp.show()

# %%
# Polynomial Fit
# --------------
# Pass ``order=`` to fit an order-N polynomial. The CI band is bootstrapped
# around the curved fit the same way as for the linear case.

n = 120
x = rng.uniform(-2.0, 2.0, size=n)
y = x ** 2 + 0.3 * rng.normal(size=n)
quad_df = pd.DataFrame({"x": x, "y": y})

ax = regplot(
    data=quad_df,
    x="x",
    y="y",
    order=2,
    title="Quadratic Fit (order=2)",
    xlabel="x",
    ylabel="y",
)
pp.show()

# %%
# Hue'd Fits (the novelty over ``sns.regplot``)
# ---------------------------------------------
# With ``hue=``, ``pp.regplot`` loops over hue levels and draws one
# regression per group on a shared axes. Each fit carries its own CI band
# and scatter swatch; the palette comes from the standard publiplots
# palette resolver. A per-group legend is stashed automatically.

per_group = 50
rows = []
for i, g in enumerate(["A", "B", "C"]):
    gx = rng.normal(size=per_group)
    slope = 0.4 + 0.6 * i
    gy = slope * gx + 0.3 * rng.normal(size=per_group)
    rows.extend({"x": xi, "y": yi, "group": g} for xi, yi in zip(gx, gy))
hue_df = pd.DataFrame(rows)

ax = regplot(
    data=hue_df,
    x="x",
    y="y",
    hue="group",
    palette="pastel",
    title="Per-group Fits (hue='group')",
    xlabel="x",
    ylabel="y",
)
pp.show()

# %%
# Lowess Smoothing
# ----------------
# ``lowess=True`` fits a locally-weighted regression curve instead of an
# ordinary-least-squares line. Handy for data with non-parametric trends.
# Requires ``statsmodels``.

n = 150
x = np.linspace(-3.0, 3.0, n)
y = np.sin(x) + 0.2 * rng.normal(size=n)
lowess_df = pd.DataFrame({"x": x, "y": y})

ax = regplot(
    data=lowess_df,
    x="x",
    y="y",
    lowess=True,
    title="Lowess Smoothing",
    xlabel="x",
    ylabel="y",
)
pp.show()

# %%
# Binned Aggregation with Error Bars
# ----------------------------------
# ``x_bins=`` discretizes the x-axis and ``x_estimator=`` aggregates each
# bin's y values. ``x_ci="ci"`` draws bootstrap error bars on each bin.
# Useful when the raw scatter is too dense to read but the marginal
# trend matters.

n = 200
x = rng.uniform(-3.0, 3.0, size=n)
y = 0.8 * x + 0.5 * rng.normal(size=n)
binned_df = pd.DataFrame({"x": x, "y": y})

ax = regplot(
    data=binned_df,
    x="x",
    y="y",
    x_bins=10,
    x_estimator=np.mean,
    title="Binned Aggregation (x_bins=10, x_estimator=mean)",
    xlabel="x",
    ylabel="y",
)
pp.show()
