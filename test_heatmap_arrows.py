#!/usr/bin/env python
"""
Test script matching user's heatmap example.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'src')

import publiplots as pp

# Create expression matrix
np.random.seed(42)
genes = ['BRCA1', 'KRAS', 'EGFR', 'TP53', 'MYC', 'PIK3CA']
samples = ['S1', 'S2', 'S3', 'S4', 'S5']
expr_matrix = pd.DataFrame(
    np.random.randn(len(genes), len(samples)),
    index=genes,
    columns=samples
)

# Convert to long format
df_long = (
    expr_matrix
    .reset_index()
    .melt(id_vars="index", var_name="sample", value_name="expression")
    .rename(columns={"index": "gene"})
)

# Add detection threshold
detect_thresh = 0

# % Samples expressing gene
gene_stats = (
    df_long
    .assign(detected = lambda d: d["expression"] > detect_thresh)
    .groupby("gene")
    .agg(
        pct_exp = ("detected", lambda x: 100 * x.mean()),
        n_detected = ("detected", "sum"),
        mean_exp_detected = (
            "expression",
            lambda x: x[x > detect_thresh].mean() if any(x > detect_thresh) else 0
        )
    )
)

# Merge stats back
df_long = df_long.merge(gene_stats, on="gene")

# Optional: magnitude feature for dot size
df_long["abs_expr"] = df_long["expression"].abs()
df_long["condition"] = df_long["sample"].isin(["S3", "S4", "S5"]).map({True: "Disease", False: "Control"})

print("Testing heatmap with label annotations...")

fig, axes = (
    pp.complex_heatmap(
        data=df_long,
        x="sample",
        y="gene",
        value="expression",
        size="pct_exp",
        figsize=(4, 4),
        row_cluster=False,
        col_cluster=False,
    )
    .add_top(pp.label, x="condition", arrow="down", hue="condition")  # Colored by condition
    .add_top(pp.block, x="condition", height=5)
    .add_left(pp.label, y="gene", arrow="right")  # No coloring
    .add_left(pp.block, y="gene", width=5)
    .add_right(pp.legend)
    .build()
)

fig.savefig('/tmp/test_heatmap_full.png', dpi=100, bbox_inches='tight')
print("✓ Saved to /tmp/test_heatmap_full.png")

print("\n✓ All tests completed!")
