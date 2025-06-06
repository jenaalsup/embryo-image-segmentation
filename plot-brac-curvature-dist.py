import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import EllipseModel
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import seaborn as sns
import glob

# --- 1. Load CSVs ---
target_slices = ["71", "83", "40"]
all_cells_files = [f for f in glob.glob("data/D3_slice*.csv") if any(f"slice{num}" in f for num in target_slices)]
brachyury_cells_files = [f for f in glob.glob("data/bra_D3_slice*.csv") if any(f"slice{num}" in f for num in target_slices)]
all_cells = pd.concat(
    [pd.read_csv(file).assign(Source=file) for file in all_cells_files],
    ignore_index=True
)
brachyury_cells = pd.concat(
    [pd.read_csv(file).assign(Source=file) for file in brachyury_cells_files],
    ignore_index=True
)

# --- 2. Classify based on proximity ---
all_coords = all_cells[["X", "Y"]].values
brachyury_coords = brachyury_cells[["X", "Y"]].values

tree = cKDTree(brachyury_coords)
radius = 10
matches = tree.query_ball_point(all_coords, r=radius)
is_positive = [len(m) > 0 for m in matches]

all_cells["Brachyury_Status"] = ["Brachyury-positive" if match else "Brachyury-negative" for match in is_positive]

# --- 3. Fit ellipse and compute curvature for all cells (for Plot 1) ---
model = EllipseModel()
model.estimate(all_coords)
xc, yc, a, b, theta = model.params

t_vals = np.linspace(0, 2 * np.pi, 1000)
x_ellipse = xc + a * np.cos(t_vals) * np.cos(theta) - b * np.sin(t_vals) * np.sin(theta)
y_ellipse = yc + a * np.cos(t_vals) * np.sin(theta) + b * np.sin(t_vals) * np.cos(theta)
ellipse_points = np.stack([x_ellipse, y_ellipse], axis=1)

raw_curvature = (a * b) / ((b**2 * np.cos(t_vals)**2 + a**2 * np.sin(t_vals)**2) ** 1.5)
normalized_curvatures = raw_curvature * np.sqrt(a * b)

dist_matrix = cdist(all_coords, ellipse_points)
closest_point_idx = np.argmin(dist_matrix, axis=1)
all_cells["Curvature"] = normalized_curvatures[closest_point_idx]

# --- 4. Plot XY Cell Map (only slice71) ---
slice71_cells = all_cells[all_cells["Source"].str.contains("slice71")].copy()
pos = slice71_cells[slice71_cells["Brachyury_Status"] == "Brachyury-positive"]
neg = slice71_cells[slice71_cells["Brachyury_Status"] == "Brachyury-negative"]

plt.figure(num=1, figsize=(6, 6))
plt.scatter(neg["X"], neg["Y"], s=10, color="gray", label="Brachyury-negative")
plt.scatter(pos["X"], pos["Y"], s=10, color="red", label="Brachyury-positive")
plt.gca().set_aspect("equal")
plt.gca().invert_yaxis()
plt.legend()
plt.title("D3 Brachyury Classification of Cells")
plt.xlabel("X")
plt.ylabel("Y")
plt.tight_layout()
#plt.savefig("brac-points-D3-slices71.eps", format="eps")
plt.show(block=False)

# --- 5. Plot Aspect Ratio vs. Curvature (ALL slices) ---
plt.figure(num=2, figsize=(7, 5))
colors = all_cells["Brachyury_Status"].map({"Brachyury-positive": "red", "Brachyury-negative": "blue"})
plt.scatter(
    all_cells["Curvature"],
    all_cells["AR"],
    c=colors,
    alpha=0.6
)
plt.xlabel("Curvature")
plt.ylabel("Aspect Ratio")
plt.title("D3 Aspect Ratio vs. Curvature by Brachyury Status (All Slices)")
for status, color in [("Brachyury-positive", "red"), ("Brachyury-negative", "blue")]:
    plt.scatter([], [], label=status, color=color)
plt.legend()
plt.tight_layout()
#plt.savefig("scatterplot-brac-curvature.eps", format="eps")
plt.show(block=False)

# --- 5. Plot Curvature Distribution by Brachyury Status(ALL slices) ---
# note: this plot is better if just slices 71 and 83 are used (don't include slice 40 because the brac are too spread out)
plt.figure(num=3, figsize=(7, 5))
sns.kdeplot(
    data=all_cells,                   
    x="Curvature",
    hue="Brachyury_Status",  
    common_norm=False,
    fill=True,
    alpha=0.5,
    palette={
        "Brachyury-positive": "red",
        "Brachyury-negative": "gray"
    }
)
plt.title("Curvature Distribution by Brachyury Status")
plt.xlabel("Curvature")
plt.ylabel("Density")
plt.grid(False)
plt.tight_layout()
plt.savefig("brac-curvature-distribution.eps", format="eps")
plt.show()

