"""Test realistic legend scenarios like those in the screenshots."""
import matplotlib.pyplot as plt
import numpy as np
from publiplots.utils.legend import LegendBuilder, create_legend_handles

np.random.seed(42)

# Test 1: Categorical groups (like first screenshot)
print("Test 1: Scatter Plot with Categorical Groups")
fig, ax = plt.subplots(figsize=(5, 4))
n_points = 50

# Create data for three groups
for i, (group, color) in enumerate([('High', 'green'), ('Medium', 'blue'), ('Low', 'orange')]):
    x = np.random.randn(n_points) + i * 30
    y = np.random.randn(n_points) * 15 + 100 + i * 30
    ax.scatter(x, y, c=color, alpha=0.6, s=80, edgecolors=color, linewidths=1.5)

ax.set_xlabel('X Variable', fontsize=12)
ax.set_ylabel('Y Variable', fontsize=12)
ax.set_title('Scatter Plot with Categorical Groups', fontsize=14)

# Add legend
builder = LegendBuilder(ax, x_offset=2, gap=2, vpad=5)
handles = create_legend_handles(
    labels=['High', 'Medium', 'Low'],
    colors=['green', 'blue', 'orange'],
    style='circle'
)
builder.add_legend(handles, label="group", frameon=False)

plt.tight_layout()
plt.savefig('/tmp/test1_categorical.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved to /tmp/test1_categorical.png\n")
plt.close()

# Test 2: Continuous color scale + size legend (like second screenshot)
print("Test 2: Scatter Plot with Continuous Color + Size")
fig, ax = plt.subplots(figsize=(6, 5))

# Create data with continuous color scale
n_points = 100
x = np.random.randn(n_points) * 20 + 40
y = np.random.randn(n_points) * 30 + 100
scores = np.random.rand(n_points) * 100
sizes = np.random.choice([30, 60, 90], n_points)

# Plot with color mapping
scatter = ax.scatter(x, y, c=scores, s=sizes, cmap='viridis',
                     alpha=0.6, edgecolors='black', linewidths=1.5)

ax.set_xlabel('X Variable', fontsize=12)
ax.set_ylabel('Y Variable', fontsize=12)
ax.set_title('Scatter Plot with Continuous Color Scale', fontsize=14)

# Add legends
builder = LegendBuilder(ax, x_offset=2, gap=3, vpad=5)

# Add size legend first
size_handles = create_legend_handles(
    labels=['30', '60', '90'],
    colors=['gray', 'gray', 'gray'],
    sizes=[30, 60, 90],
    style='circle'
)
builder.add_legend(size_handles, label="Magnitude", frameon=False)

# Add colorbar
builder.add_colorbar(
    mappable=scatter,
    label="Continuous Score",
    height=20,
    width=4.5,
    title_position='top'
)

plt.tight_layout()
plt.savefig('/tmp/test2_continuous.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved to /tmp/test2_continuous.png")
print(f"  - Final y position: {builder.current_y:.2f} mm")
print(f"  - Elements: {len(builder.elements)}\n")
plt.close()

# Test 3: Multiple legends like differential expression plot
print("Test 3: Differential Expression with Multiple Legends")
fig, ax = plt.subplots(figsize=(6, 4))

# Create a bubble plot
conditions = ['Ctrl', 'Trt1', 'Trt2', 'Trt3']
cell_types = ['TypeA', 'TypeB', 'TypeC', 'TypeD']

np.random.seed(42)
x_pos = []
y_pos = []
sizes = []
colors = []

for i, cond in enumerate(conditions):
    for j, cell_type in enumerate(cell_types):
        x_pos.append(i)
        y_pos.append(j)
        pval = np.random.rand() * 3
        sizes.append(pval * 50)

        # Random up/down/neutral
        direction = np.random.choice(['Up', 'Down', 'Neutral'])
        if direction == 'Up':
            colors.append('green')
        elif direction == 'Down':
            colors.append('red')
        else:
            colors.append('gray')

ax.scatter(x_pos, y_pos, s=sizes, c=colors, alpha=0.6,
           edgecolors=['green' if c=='green' else 'red' if c=='red' else 'gray' for c in colors],
           linewidths=2)

ax.set_xticks(range(len(conditions)))
ax.set_xticklabels(conditions)
ax.set_yticks(range(len(cell_types)))
ax.set_yticklabels(cell_types)
ax.set_xlabel('Condition', fontsize=12)
ax.set_ylabel('Cell Type', fontsize=12)
ax.set_title('Differential Expression Analysis', fontsize=14)

# Add legends
builder = LegendBuilder(ax, x_offset=2, gap=3, vpad=5)

# Add size legend (pvalue)
pval_handles = create_legend_handles(
    labels=['1.0', '2.0', '3.0'],
    colors=['gray', 'gray', 'gray'],
    sizes=[50, 100, 150],
    style='circle'
)
builder.add_legend(pval_handles, label="pvalue", frameon=False)

# Add category legend (Up/Down/Neutral)
category_handles = create_legend_handles(
    labels=['Up', 'Down', 'Neutral'],
    colors=['green', 'red', 'gray'],
    style='circle'
)
builder.add_legend(category_handles, label="category", frameon=False)

plt.tight_layout()
plt.savefig('/tmp/test3_differential.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved to /tmp/test3_differential.png")
print(f"  - Final y position: {builder.current_y:.2f} mm")
print(f"  - Elements: {len(builder.elements)}")
print(f"  - Columns: {len(builder.columns) + 1}\n")
plt.close()

print("="*60)
print("All realistic tests completed!")
print("Please check the images to verify proper positioning.")
