"""Debug colorbar rendering to see what's happening."""
import matplotlib.pyplot as plt
import numpy as np
from publiplots.utils.legend import LegendBuilder, create_legend_handles

fig, ax = plt.subplots(figsize=(6, 5))

# Create data with continuous color scale
n_points = 100
x = np.random.randn(n_points) * 20 + 40
y = np.random.randn(n_points) * 30 + 100
scores = np.random.rand(n_points) * 100

# Plot with color mapping
scatter = ax.scatter(x, y, c=scores, cmap='viridis',
                     s=60, alpha=0.6, edgecolors='black', linewidths=1.5)

ax.set_xlabel('X Variable', fontsize=12)
ax.set_ylabel('Y Variable', fontsize=12)
ax.set_title('Colorbar Debug Test', fontsize=14)

# Add legends
print("Creating builder...")
builder = LegendBuilder(ax, x_offset=2, gap=3, vpad=5)
print(f"  Axes height: {builder._get_axes_height():.2f} mm")
print(f"  Initial current_y: {builder.current_y:.2f} mm")

# Add colorbar
print("\nAdding colorbar...")
print(f"  Before colorbar: current_y = {builder.current_y:.2f} mm")

cbar = builder.add_colorbar(
    mappable=scatter,
    label="Continuous Score",
    height=20,  # Try larger height
    width=4.5,
    title_position='top'
)

print(f"  After colorbar: current_y = {builder.current_y:.2f} mm")
print(f"  Colorbar ax position: {cbar.ax.get_position()}")
print(f"  Colorbar ax bounds: {cbar.ax.get_position().bounds}")

# Add size legend
print("\nAdding size legend...")
size_handles = create_legend_handles(
    labels=['30', '60', '90'],
    colors=['gray', 'gray', 'gray'],
    sizes=[30, 60, 90],
    style='circle'
)
builder.add_legend(size_handles, label="magnitude", frameon=False)

print(f"  After legend: current_y = {builder.current_y:.2f} mm")
print(f"\nFinal elements: {len(builder.elements)}")

# Mark the colorbar axes with a red box for debugging
cbar_ax_pos = cbar.ax.get_position()
print(f"\nColorbar axes in figure coords:")
print(f"  x0={cbar_ax_pos.x0:.3f}, y0={cbar_ax_pos.y0:.3f}")
print(f"  width={cbar_ax_pos.width:.3f}, height={cbar_ax_pos.height:.3f}")

plt.savefig('/tmp/test_colorbar_debug.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved to /tmp/test_colorbar_debug.png")
plt.close()
