"""Test if bbox_inches='tight' is causing the colorbar to disappear."""
import matplotlib.pyplot as plt
import numpy as np
from publiplots.utils.legend import LegendBuilder

fig, ax = plt.subplots(figsize=(6, 5))
x = np.random.rand(50)
y = np.random.rand(50) * 100
c = np.random.rand(50) * 100

scatter = ax.scatter(x, y, c=c, cmap='viridis', s=80, edgecolors='black', linewidths=1)
ax.set_xlabel('X Variable')
ax.set_ylabel('Y Variable')
ax.set_title('Testing bbox_inches effect')

builder = LegendBuilder(ax, x_offset=2, gap=3, vpad=5)

# Add colorbar
cbar = builder.add_colorbar(
    mappable=scatter,
    label="Color Scale",
    height=25,
    width=5,
    title_position='top'
)

print(f"Colorbar axes position: {cbar.ax.get_position().bounds}")
print(f"Main axes position: {ax.get_position().bounds}")

# Save WITHOUT bbox_inches='tight'
plt.savefig('/tmp/test_no_tight.png', dpi=150)
print("✓ Saved WITHOUT bbox_inches='tight' to /tmp/test_no_tight.png")

# Save WITH bbox_inches='tight'
plt.savefig('/tmp/test_with_tight.png', dpi=150, bbox_inches='tight')
print("✓ Saved WITH bbox_inches='tight' to /tmp/test_with_tight.png")

print("\nCompare the two images - colorbar should be visible in both if working correctly")
