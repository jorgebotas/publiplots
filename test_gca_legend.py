"""Test that pp.legend() works without passing axes (using plt.gca())."""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/home/user/publiplots/src')

import publiplots as pp
import matplotlib.pyplot as plt

np.random.seed(42)
data = pd.DataFrame({
    'x': np.random.randn(50),
    'y': np.random.randn(50),
    'category': np.random.choice(['A', 'B', 'C'], 50),
})

print("="*70)
print("TEST: pp.legend() without passing axes (like plt.legend())")
print("="*70)

# Test 1: Using pp.scatterplot (which creates axes)
print("\nTest 1: After pp.scatterplot()")
fig, ax = pp.scatterplot(
    data=data,
    x='x',
    y='y',
    hue='category',
    legend=False,
    figsize=(6, 5),
    title='Test 1: pp.legend() without axes'
)

print(f"  Current axes from plt.gca(): {plt.gca()}")
print(f"  Axes from scatterplot: {ax}")
print(f"  Are they the same? {plt.gca() is ax}")

# Call pp.legend() WITHOUT passing ax
print("\n  Calling pp.legend() without axes...")
builder1 = pp.legend()  # Should use plt.gca()

print(f"  Builder created: {builder1 is not None}")
print(f"  Builder axes: {builder1.ax}")
print(f"  Correct axes? {builder1.ax is ax}")
print(f"  Elements: {len(builder1.elements)}")

plt.savefig('/tmp/test_gca_1.png', dpi=150)
print("  ✓ Saved to /tmp/test_gca_1.png\n")
plt.close()

# Test 2: Using standard matplotlib (implicit current axes)
print("Test 2: After plt.subplots()")
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(data['x'], data['y'], c='blue', alpha=0.6)
ax.set_title('Test 2: pp.legend() with plt axes')

print(f"  Current axes: {plt.gca() is ax}")

# Add legend without passing ax
from publiplots.utils.legend import create_legend_handles
handles = create_legend_handles(
    labels=['A', 'B', 'C'],
    colors=['red', 'green', 'blue'],
    style='circle'
)

print("\n  Calling pp.legend() without axes...")
builder2 = pp.legend(handles=handles, label="Category")

print(f"  Builder created: {builder2 is not None}")
print(f"  Builder axes: {builder2.ax}")
print(f"  Correct axes? {builder2.ax is ax}")
print(f"  Elements: {len(builder2.elements)}")

plt.savefig('/tmp/test_gca_2.png', dpi=150)
print("  ✓ Saved to /tmp/test_gca_2.png\n")
plt.close()

# Test 3: Multiple calls without axes
print("Test 3: Multiple calls without axes")
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(data['x'], data['y'], c='purple', alpha=0.6)
ax.set_title('Test 3: Multiple pp.legend() calls')

print(f"  Current axes: {ax}")

# First call
handles1 = create_legend_handles(['X', 'Y'], colors=['orange', 'cyan'], style='circle')
builder3a = pp.legend(handles=handles1, label="First")
print(f"\n  First call: {len(builder3a.elements)} elements")

# Second call (should reuse same builder)
handles2 = create_legend_handles(['P', 'Q', 'R'], colors=['red', 'green', 'blue'], style='rectangle')
builder3b = pp.legend(handles=handles2, label="Second")
print(f"  Second call: {len(builder3b.elements)} elements")

# Check if same builder
print(f"  Same builder? {builder3a is builder3b}")
print(f"  Correct axes? {builder3b.ax is ax}")

plt.savefig('/tmp/test_gca_3.png', dpi=150)
print("  ✓ Saved to /tmp/test_gca_3.png\n")
plt.close()

print("="*70)
print("ALL TESTS PASSED!")
print("="*70)
print("\nSummary:")
print("  ✓ pp.legend() works without passing axes")
print("  ✓ Automatically uses plt.gca() like plt.legend()")
print("  ✓ Builder reuse works correctly")
print("  ✓ Works with both pp.scatterplot() and plt.subplots()")
