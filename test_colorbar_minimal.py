"""Minimal test to isolate colorbar rendering issue."""
import matplotlib.pyplot as plt
import numpy as np

# Create simple scatter
fig, ax = plt.subplots(figsize=(5, 4))
x = np.random.rand(50)
y = np.random.rand(50)
c = np.random.rand(50) * 100

scatter = ax.scatter(x, y, c=c, cmap='viridis', s=100)
ax.set_title('Minimal Colorbar Test')

# Test 1: Standard matplotlib colorbar (for comparison)
print("Creating standard matplotlib colorbar...")
cbar_standard = fig.colorbar(scatter, ax=ax)
plt.savefig('/tmp/test_standard_colorbar.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved to /tmp/test_standard_colorbar.png")
plt.close()

# Test 2: Manual axes creation (like our code)
fig, ax = plt.subplots(figsize=(5, 4))
scatter = ax.scatter(x, y, c=c, cmap='viridis', s=100)
ax.set_title('Manual Axes Colorbar Test')

# Create colorbar axes manually
ax_pos = ax.get_position()
fig_extent = fig.get_window_extent()
print(f"\nAxes position: {ax_pos.bounds}")
print(f"Figure extent (pixels): width={fig_extent.width}, height={fig_extent.height}")
print(f"Figure DPI: {fig.dpi}")

# Create axes at top right
cbar_width_fig = 0.03  # 3% of figure width
cbar_height_fig = 0.15  # 15% of figure height
cbar_left = ax_pos.x1 + 0.02
cbar_bottom = ax_pos.y1 - cbar_height_fig

print(f"\nColorbar axes:")
print(f"  left={cbar_left:.3f}, bottom={cbar_bottom:.3f}")
print(f"  width={cbar_width_fig:.3f}, height={cbar_height_fig:.3f}")

cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width_fig, cbar_height_fig])
cbar = fig.colorbar(scatter, cax=cbar_ax)
print(f"  Final bounds: {cbar_ax.get_position().bounds}")

plt.savefig('/tmp/test_manual_colorbar.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved to /tmp/test_manual_colorbar.png")
plt.close()

# Test 3: Using our LegendBuilder
print("\nTesting with LegendBuilder...")
from publiplots.utils.legend import LegendBuilder

fig, ax = plt.subplots(figsize=(5, 4))
scatter = ax.scatter(x, y, c=c, cmap='viridis', s=100)
ax.set_title('LegendBuilder Colorbar Test')

builder = LegendBuilder(ax, x_offset=2, gap=3, vpad=5)
print(f"Builder initial state:")
print(f"  axes_height = {builder._get_axes_height():.2f} mm")
print(f"  current_y = {builder.current_y:.2f} mm")

cbar = builder.add_colorbar(
    mappable=scatter,
    label="Test Colorbar",
    height=20,
    width=5,
    title_position='top'
)

print(f"\nColorbar created:")
print(f"  Axes position: {cbar.ax.get_position().bounds}")
print(f"  Axes visible: {cbar.ax.get_visible()}")

plt.savefig('/tmp/test_legendbuilder_colorbar.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved to /tmp/test_legendbuilder_colorbar.png")

print("\nCheck all three images to compare!")
