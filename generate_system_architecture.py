"""
Generate System Architecture Diagram
This creates a simple flowchart showing the ML pipeline
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Define box properties
box_style = dict(boxstyle="round,pad=0.5", facecolor="lightblue", edgecolor="black", linewidth=2)
arrow_style = dict(arrowstyle="->", lw=2, color="black")

# Define positions
y_positions = [9, 7.5, 6, 4.5, 3, 1.5]
x_center = 5

# Draw boxes
boxes = [
    ("Data Loading\n(train.csv, test.csv)", 2.5, y_positions[0]),
    ("Data Preprocessing\n& Feature Engineering", 2.5, y_positions[1]),
    ("Feature Scaling\n(RobustScaler)", 2.5, y_positions[2]),
    ("Model Training\n(Baseline + Advanced)", 7.5, y_positions[3]),
    ("Hyperparameter\nOptimization (Optuna)", 7.5, y_positions[4]),
    ("Evaluation &\nVisualization", 7.5, y_positions[5]),
]

# Draw boxes
for text, x, y in boxes:
    box = FancyBboxPatch((x-1.2, y-0.4), 2.4, 0.8, **box_style)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')

# Draw arrows
arrows = [
    ((2.5, y_positions[0]-0.4), (2.5, y_positions[1]+0.4)),
    ((2.5, y_positions[1]-0.4), (2.5, y_positions[2]+0.4)),
    ((2.5, y_positions[2]-0.4), (7.5, y_positions[3]+0.4)),
    ((7.5, y_positions[3]-0.4), (7.5, y_positions[4]+0.4)),
    ((7.5, y_positions[4]-0.4), (7.5, y_positions[5]+0.4)),
]

for (start, end) in arrows:
    arrow = FancyArrowPatch(start, end, **arrow_style)
    ax.add_patch(arrow)

# Add side annotations
ax.text(0.5, y_positions[1], "EDA\nAnalysis", ha='center', va='center', 
        fontsize=9, bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))
ax.text(0.5, y_positions[2], "PCA\nAnalysis", ha='center', va='center', 
        fontsize=9, bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))

# Add model types annotation
ax.text(10, y_positions[3], "LR, RF, SVM\nNB, KNN, AdaBoost\nLightGBM, XGBoost\nCatBoost, MLP", 
        ha='left', va='center', fontsize=8, 
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7))

# Set axis properties
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(6, 9.8, 'System Architecture: Complete ML Pipeline', 
        ha='center', va='top', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/system_architecture.png', dpi=300, bbox_inches='tight')
print("System architecture diagram saved to figures/system_architecture.png")
plt.close()

