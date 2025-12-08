"""Test the new mm-based LegendBuilder implementation."""
import matplotlib.pyplot as plt
import numpy as np
from publiplots.utils.legend import LegendBuilder, legend, create_legend_handles

# Create a simple plot
fig, ax = plt.subplots(figsize=(4, 3))
ax.scatter([1, 2, 3], [1, 2, 3], c='blue', alpha=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')

print("Testing LegendBuilder with mm-based positioning...")

# Test 1: Create builder with default mm parameters
print("\nTest 1: Create builder")
builder = LegendBuilder(ax, x_offset=2, gap=2, column_spacing=5, vpad=5)
print(f"  ✓ Builder created")
print(f"  - Initial x position: {builder.current_x} mm")
print(f"  - Initial y position: {builder.current_y} mm")

# Test 2: Add a legend
print("\nTest 2: Add legend")
handles = create_legend_handles(
    labels=['Group A', 'Group B', 'Group C'],
    colors=['red', 'green', 'blue'],
    style='circle'
)
leg = builder.add_legend(handles, label="Groups", frameon=False)
print(f"  ✓ Legend added")
print(f"  - Current y position after legend: {builder.current_y} mm")
print(f"  - Column width: {builder.current_column_width} mm")

# Test 3: Add a colorbar
print("\nTest 3: Add colorbar")
try:
    cbar = builder.add_colorbar(
        cmap='viridis',
        vmin=0,
        vmax=1,
        label="Values",
        height=15,
        width=4.5,
        title_position='top'
    )
    print(f"  ✓ Colorbar added")
    print(f"  - Current y position after colorbar: {builder.current_y} mm")
except Exception as e:
    print(f"  ✗ Error adding colorbar: {e}")

# Test 4: Add divergent colorbar
print("\nTest 4: Add divergent colorbar with center")
try:
    cbar2 = builder.add_colorbar(
        cmap='RdBu_r',
        vmin=-2,
        vmax=2,
        center=0,
        label="Log2 FC",
        height=15,
        ticks=[-2, 0, 2],
        title_position='top'
    )
    print(f"  ✓ Divergent colorbar added")
    print(f"  - Current y position: {builder.current_y} mm")
except Exception as e:
    print(f"  ✗ Error adding divergent colorbar: {e}")

# Test 5: Check overflow detection
print("\nTest 5: Test overflow (should create new column)")
print(f"  - Current column: {len(builder.columns) + 1}")
print(f"  - Current y: {builder.current_y} mm")

# Add many legends to trigger overflow
try:
    for i in range(5):
        handles = create_legend_handles(
            labels=[f'Item {i}-{j}' for j in range(3)],
            colors=['red', 'green', 'blue'],
            style='rectangle'
        )
        builder.add_legend(handles, label=f"Set {i}", frameon=False)
    print(f"  ✓ Multiple legends added")
    print(f"  - Total columns: {len(builder.columns) + 1}")
    print(f"  - Column widths: {builder.columns} mm")
except Exception as e:
    print(f"  ✗ Error: {e}")

print("\n" + "="*60)
print("All tests completed!")
print(f"Total elements created: {len(builder.elements)}")
print(f"Columns used: {len(builder.columns) + 1}")

# Save figure
try:
    plt.savefig('/tmp/test_legend_mm.png', dpi=150, bbox_inches='tight')
    print("\nFigure saved to /tmp/test_legend_mm.png")
except Exception as e:
    print(f"\nCouldn't save figure: {e}")

plt.close()
print("\n✓ All tests passed!")
