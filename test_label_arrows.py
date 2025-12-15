#!/usr/bin/env python
"""
Test script for label() function with arrows.
"""
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.insert(0, 'src')

from publiplots.plot.complex_heatmap.annotations import label

# Test data
gene_names = ['BRCA1', 'KRAS', 'EGFR', 'TP53', 'MYC', 'PIK3CA']

# Test 1: Horizontal labels with upward arrows
print("Test 1: Horizontal labels with upward arrows")
fig1, ax1 = label(
    gene_names,
    x=True,
    order=gene_names,
    arrow='up',
    arrow_kws={'arrow_height': 5, 'frac': 0.2, 'rad': 2},
    rotation=90,
    fontsize=10
)
fig1.savefig('/tmp/test_label_up.png', dpi=100, bbox_inches='tight')
print("✓ Saved to /tmp/test_label_up.png")

# Test 2: Vertical labels with right arrows
print("\nTest 2: Vertical labels with right arrows")
fig2, ax2 = label(
    gene_names,
    y=True,
    order=gene_names,
    arrow='right',
    arrow_kws={'arrow_width': 5, 'frac': 0.2, 'rad': 2},
    fontsize=10
)
fig2.savefig('/tmp/test_label_right.png', dpi=100, bbox_inches='tight')
print("✓ Saved to /tmp/test_label_right.png")

# Test 3: Horizontal labels with downward arrows
print("\nTest 3: Horizontal labels with downward arrows")
fig3, ax3 = label(
    gene_names,
    x=True,
    order=gene_names,
    arrow='down',
    arrow_kws={'arrow_height': 5, 'frac': 0.2, 'rad': 2},
    rotation=-90,
    fontsize=10
)
fig3.savefig('/tmp/test_label_down.png', dpi=100, bbox_inches='tight')
print("✓ Saved to /tmp/test_label_down.png")

# Test 4: Vertical labels with left arrows
print("\nTest 4: Vertical labels with left arrows")
fig4, ax4 = label(
    gene_names,
    y=True,
    order=gene_names,
    arrow='left',
    arrow_kws={'arrow_width': 5, 'frac': 0.2, 'rad': 2},
    fontsize=10
)
fig4.savefig('/tmp/test_label_left.png', dpi=100, bbox_inches='tight')
print("✓ Saved to /tmp/test_label_left.png")

print("\n✓ All tests completed successfully!")
