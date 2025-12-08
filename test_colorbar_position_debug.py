"""Debug exact positioning of colorbar elements."""
import matplotlib.pyplot as plt
import numpy as np
from publiplots.utils.legend import LegendBuilder

fig, ax = plt.subplots(figsize=(6, 5))
x = np.random.rand(50)
y = np.random.rand(50) * 100
c = np.random.rand(50) * 100

scatter = ax.scatter(x, y, c=c, cmap='viridis', s=80)
ax.set_title('Colorbar Position Debug')

print("Creating LegendBuilder...")
builder = LegendBuilder(ax, x_offset=2, gap=3, vpad=5)

print(f"\nInitial state:")
print(f"  Axes height: {builder._get_axes_height():.2f} mm")
print(f"  current_y (remaining): {builder.current_y:.2f} mm")
print(f"  Axes position in fig coords: {ax.get_position().bounds}")

# Add colorbar with debugging
print(f"\nAdding colorbar...")
print(f"  Before: current_y = {builder.current_y:.2f} mm")

# Manually trace through add_colorbar logic
title_position = "top"
label = "Test Label"
height = 20  # mm
width = 5    # mm

if title_position == "top" and label:
    # Calculate where title will go
    x_fig_title, y_fig_title = builder._mm_to_figure_coords(builder.current_x, builder.current_y)
    print(f"\nTitle positioning:")
    print(f"  current_y (remaining): {builder.current_y:.2f} mm")
    print(f"  Title position in fig coords: ({x_fig_title:.3f}, {y_fig_title:.3f})")

cbar = builder.add_colorbar(
    mappable=scatter,
    label=label,
    height=height,
    width=width,
    title_position=title_position
)

print(f"\nAfter adding colorbar:")
print(f"  current_y (remaining): {builder.current_y:.2f} mm")
print(f"  Colorbar axes bounds: {cbar.ax.get_position().bounds}")
print(f"  Colorbar axes visible: {cbar.ax.get_visible()}")

# Check if colorbar has actual content
print(f"\nColorbar properties:")
print(f"  mappable: {cbar.mappable}")
print(f"  vmin, vmax: {cbar.vmin}, {cbar.vmax}")
print(f"  number of ticks: {len(cbar.get_ticks())}")

# Draw a red box around the colorbar axes for debugging
from matplotlib.patches import Rectangle
cbar_bounds = cbar.ax.get_position()
debug_rect = Rectangle(
    (cbar_bounds.x0, cbar_bounds.y0),
    cbar_bounds.width,
    cbar_bounds.height,
    transform=fig.transFigure,
    fill=False,
    edgecolor='red',
    linewidth=2,
    zorder=1000
)
fig.add_artist(debug_rect)
print("  Added RED DEBUG BOX around colorbar axes")

plt.savefig('/tmp/test_colorbar_debug_with_box.png', dpi=150)
print("\n✓ Saved to /tmp/test_colorbar_debug_with_box.png")
print("  Look for the RED BOX - that's where the colorbar axes are!")
