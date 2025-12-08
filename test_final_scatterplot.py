"""Final comprehensive test matching user's original example."""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/home/user/publiplots/src')

import publiplots as pp
import matplotlib.pyplot as plt

# Create test data
np.random.seed(42)
scatter_data = pd.DataFrame({
    'x': np.random.randn(100) * 20 + 40,
    'y': np.random.randn(100) * 30 + 100,
    'score': np.random.rand(100) * 100,
    'magnitude': np.random.choice([30, 60, 90], 100)
})

print("="*70)
print("TEST 1: Scatterplot with auto=True (should work)")
print("="*70)

fig, ax = pp.scatterplot(
    data=scatter_data,
    x='x',
    y='y',
    hue='score',
    size='magnitude',
    palette='viridis',
    hue_norm=(scatter_data['score'].min(), scatter_data['score'].max()),
    title='Scatter Plot with Continuous Color Scale (auto=True)',
    xlabel='X Variable',
    ylabel='Y Variable',
    alpha=0.6,
    figsize=(8, 6),
    legend=True  # Auto-create legend
)

plt.savefig('/tmp/final_test_auto_true.png', dpi=150)
print("✓ Saved to /tmp/final_test_auto_true.png\n")
plt.close()

print("="*70)
print("TEST 2: Scatterplot with legend=False + manual legend creation")
print("="*70)

fig, ax = pp.scatterplot(
    data=scatter_data,
    x='x',
    y='y',
    hue='score',
    size='magnitude',
    palette='viridis',
    hue_norm=(scatter_data['score'].min(), scatter_data['score'].max()),
    title='Scatter Plot with Continuous Color Scale (manual legend)',
    xlabel='X Variable',
    ylabel='Y Variable',
    alpha=0.6,
    figsize=(8, 6),
    legend=False  # Don't auto-create
)

# Manually create legend with auto mode
print("Creating legend with pp.legend(ax, auto=True)...")
builder = pp.legend(ax, auto=True)

print(f"  - Legend builder created with {len(builder.elements)} elements")
for i, (elem_type, _) in enumerate(builder.elements):
    print(f"    {i+1}. {elem_type}")

plt.savefig('/tmp/final_test_manual.png', dpi=150)
print("✓ Saved to /tmp/final_test_manual.png\n")
plt.close()

print("="*70)
print("TEST 3: Manual builder control (no auto)")
print("="*70)

fig, ax = pp.scatterplot(
    data=scatter_data,
    x='x',
    y='y',
    hue='score',
    size='magnitude',
    palette='viridis',
    hue_norm=(scatter_data['score'].min(), scatter_data['score'].max()),
    title='Scatter Plot with Manual Legend Control',
    xlabel='X Variable',
    ylabel='Y Variable',
    alpha=0.6,
    figsize=(8, 6),
    legend=False
)

# Manual control
builder = pp.legend(ax, auto=False, x_offset=2, gap=3, vpad=5)

# Add size legend first
from publiplots.utils.legend import _get_legend_data
legend_data = _get_legend_data(ax)

if 'size' in legend_data:
    print("Adding size legend...")
    builder.add_legend(**legend_data['size'])

if 'hue' in legend_data:
    print("Adding hue colorbar...")
    hue_data = legend_data['hue'].copy()
    hue_data.pop('type', None)
    builder.add_colorbar(**hue_data)

print(f"  - Total elements: {len(builder.elements)}")

plt.savefig('/tmp/final_test_manual_control.png', dpi=150)
print("✓ Saved to /tmp/final_test_manual_control.png\n")
plt.close()

print("="*70)
print("ALL TESTS COMPLETED SUCCESSFULLY!")
print("="*70)
print("\nPlease check these images:")
print("  1. /tmp/final_test_auto_true.png")
print("  2. /tmp/final_test_manual.png")
print("  3. /tmp/final_test_manual_control.png")
print("\nAll should show properly rendered colorbars at the top right!")
