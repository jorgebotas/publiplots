"""Test that multiple calls to pp.legend() reuse the same builder."""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/home/user/publiplots/src')

import publiplots as pp
import matplotlib.pyplot as plt

# Create test data
np.random.seed(42)
data = pd.DataFrame({
    'x': np.random.randn(50),
    'y': np.random.randn(50),
    'category': np.random.choice(['A', 'B', 'C'], 50),
})

print("="*70)
print("TEST: Multiple legend() calls should reuse the same builder")
print("="*70)

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(data['x'], data['y'], c='blue', alpha=0.6)
ax.set_title('Testing Builder Reuse')

# First call to legend
print("\n1. First call to pp.legend()...")
builder1 = pp.legend(ax, auto=False)
print(f"   Builder ID: {id(builder1)}")
print(f"   Elements: {len(builder1.elements)}")

# Add a legend manually
from publiplots.utils.legend import create_legend_handles
handles1 = create_legend_handles(
    labels=['Group 1', 'Group 2'],
    colors=['red', 'green'],
    style='circle'
)
builder1.add_legend(handles1, label="First Legend")
print(f"   After add_legend: {len(builder1.elements)} elements")

# Second call to legend (should return same builder)
print("\n2. Second call to pp.legend()...")
builder2 = pp.legend(ax, auto=False)
print(f"   Builder ID: {id(builder2)}")
print(f"   Elements: {len(builder2.elements)}")

# Check if they're the same object
if builder1 is builder2:
    print("   ✓ SUCCESS: Same builder instance reused!")
else:
    print("   ✗ FAIL: Different builder created!")

# Add another legend
handles2 = create_legend_handles(
    labels=['Type X', 'Type Y', 'Type Z'],
    colors=['orange', 'purple', 'cyan'],
    style='rectangle'
)
builder2.add_legend(handles2, label="Second Legend")
print(f"   After add_legend: {len(builder2.elements)} elements")

# Third call
print("\n3. Third call to pp.legend()...")
builder3 = pp.legend(ax, auto=False)
print(f"   Builder ID: {id(builder3)}")
print(f"   Elements: {len(builder3.elements)}")

if builder1 is builder3:
    print("   ✓ SUCCESS: Still same builder instance!")
else:
    print("   ✗ FAIL: Different builder created!")

# Verify that ax._legend_builder is set
print("\n4. Checking ax._legend_builder...")
if hasattr(ax, '_legend_builder'):
    print(f"   ✓ ax._legend_builder exists")
    print(f"   Builder ID: {id(ax._legend_builder)}")
    if ax._legend_builder is builder1:
        print("   ✓ SUCCESS: Stored builder matches returned builder!")
    else:
        print("   ✗ FAIL: Stored builder doesn't match!")
else:
    print("   ✗ FAIL: ax._legend_builder not set!")

print(f"\nFinal state:")
print(f"  Total elements: {len(builder1.elements)}")
print(f"  current_y: {builder1.current_y:.2f} mm")
print(f"  Columns: {len(builder1.columns) + 1}")

plt.savefig('/tmp/test_builder_reuse.png', dpi=150)
print("\n✓ Saved to /tmp/test_builder_reuse.png")
print("  (Should show both legends stacked vertically)")
plt.close()

print("\n" + "="*70)
print("TEST COMPLETE!")
print("="*70)
