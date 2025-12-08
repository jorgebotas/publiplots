"""Simple test to verify legend positioning at top right."""
import matplotlib.pyplot as plt
import numpy as np
from publiplots.utils.legend import LegendBuilder, create_legend_handles

fig, ax = plt.subplots(figsize=(5, 4))

# Simple scatter plot
np.random.seed(42)
x = np.random.rand(50) * 100
y = np.random.rand(50) * 100
ax.scatter(x, y, c='blue', alpha=0.5, s=50)

ax.set_xlabel('X Variable')
ax.set_ylabel('Y Variable')
ax.set_title('Legend Position Test')
ax.set_xlim(-10, 110)
ax.set_ylim(-10, 110)

# Add legend at top right
print("Creating LegendBuilder...")
builder = LegendBuilder(ax, x_offset=2, gap=3, vpad=5)
print(f"  - Axes height: {builder._get_axes_height():.2f} mm")
print(f"  - Initial current_y (remaining space): {builder.current_y:.2f} mm")

# Convert to see where it will be positioned
x_fig, y_fig = builder._mm_to_figure_coords(builder.current_x, builder.current_y)
print(f"  - Initial position in fig coords: ({x_fig:.3f}, {y_fig:.3f})")

handles = create_legend_handles(
    labels=['Group A', 'Group B', 'Group C'],
    colors=['red', 'green', 'blue'],
    style='circle'
)

print("\nAdding legend...")
legend = builder.add_legend(handles, label="Groups", frameon=False)

print(f"  - After legend, current_y: {builder.current_y:.2f} mm")

# Add a second legend
print("\nAdding second legend...")
handles2 = create_legend_handles(
    labels=['Type X', 'Type Y'],
    colors=['orange', 'purple'],
    style='rectangle'
)
legend2 = builder.add_legend(handles2, label="Types", frameon=False)
print(f"  - After 2nd legend, current_y: {builder.current_y:.2f} mm")

plt.savefig('/tmp/test_position_simple.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved to /tmp/test_position_simple.png")
print("\nLegends should be positioned at TOP RIGHT of the plot area.")
plt.close()
