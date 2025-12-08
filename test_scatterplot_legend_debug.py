"""Debug what legend metadata is stored by pp.scatterplot."""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/home/user/publiplots/src')

import publiplots as pp
from publiplots.utils.legend import _get_legend_data

# Create test data
np.random.seed(42)
scatter_data = pd.DataFrame({
    'x': np.random.randn(100) * 20 + 40,
    'y': np.random.randn(100) * 30 + 100,
    'score': np.random.rand(100) * 100,
    'magnitude': np.random.choice([30, 60, 90], 100)
})

print("Creating scatterplot with continuous hue...")
fig, ax = pp.scatterplot(
    data=scatter_data,
    x='x',
    y='y',
    hue='score',
    size='magnitude',
    palette='viridis',
    hue_norm=(scatter_data['score'].min(), scatter_data['score'].max()),
    alpha=0.6,
    legend=False  # Don't auto-create legend
)

print("\nRetrieving legend metadata...")
legend_data = _get_legend_data(ax)

print("\nLegend metadata keys:", legend_data.keys())

if 'hue' in legend_data:
    print("\n'hue' legend data:")
    hue_data = legend_data['hue']
    for key, value in hue_data.items():
        if key == 'mappable':
            print(f"  {key}: {type(value)} - {value}")
        else:
            print(f"  {key}: {value}")

if 'size' in legend_data:
    print("\n'size' legend data:")
    size_data = legend_data['size']
    for key, value in size_data.items():
        print(f"  {key}: {value if not isinstance(value, list) else f'list with {len(value)} items'}")

print("\n" + "="*60)
print("Now testing auto legend creation...")

# Try to create legend in auto mode
builder = pp.legend(ax, auto=True)

print(f"\nLegend builder has {len(builder.elements)} elements")
for i, (elem_type, elem) in enumerate(builder.elements):
    print(f"  {i+1}. {elem_type}")
    if elem_type == 'colorbar':
        print(f"     - axes bounds: {elem.ax.get_position().bounds}")
        print(f"     - vmin, vmax: {elem.vmin}, {elem.vmax}")
        print(f"     - mappable: {elem.mappable}")

import matplotlib.pyplot as plt
plt.savefig('/tmp/test_scatterplot_auto_legend.png', dpi=150)
print("\n✓ Saved to /tmp/test_scatterplot_auto_legend.png")
