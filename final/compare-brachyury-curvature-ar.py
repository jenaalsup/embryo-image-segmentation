import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import EllipseModel
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import seaborn as sns

# --- 1. Load CSVs ---
all_cells = pd.read_csv("data/D3_slice64.csv")
brachyury_cells = pd.read_csv("data/D3_slice64_brachyury.csv")

# --- 2. Classify based on proximity ---
all_coords = all_cells[["X", "Y"]].values
brachyury_coords = brachyury_cells[["X", "Y"]].values

tree = cKDTree(brachyury_coords)
radius = 10  # Adjust if needed
matches = tree.query_ball_point(all_coords, r=radius)
is_positive = [len(m) > 0 for m in matches]

all_cells["Brachyury_Status"] = ["Brachyury-positive" if match else "Brachyury-negative" for match in is_positive]

# --- 3. Fit ellipse to all cell positions ---
model = EllipseModel()
model.estimate(all_coords)
xc, yc, a, b, theta = model.params

# --- 4. Generate ellipse and curvature ---
t_vals = np.linspace(0, 2 * np.pi, 1000)
x_ellipse = xc + a * np.cos(t_vals) * np.cos(theta) - b * np.sin(t_vals) * np.sin(theta)
y_ellipse = yc + a * np.cos(t_vals) * np.sin(theta) + b * np.sin(t_vals) * np.cos(theta)
ellipse_points = np.stack([x_ellipse, y_ellipse], axis=1)

# Compute curvature
raw_curvature = (a * b) / ((b**2 * np.cos(t_vals)**2 + a**2 * np.sin(t_vals)**2) ** 1.5)
normalized_curvatures = raw_curvature * np.sqrt(a * b)

# --- 5. Match each nucleus to the ellipse and assign curvature ---
dist_matrix = cdist(all_coords, ellipse_points)
closest_point_idx = np.argmin(dist_matrix, axis=1)
all_cells["Curvature"] = normalized_curvatures[closest_point_idx]

# --- 6. Plot Aspect Ratio vs. Curvature ---
plt.figure(figsize=(7, 5))
colors = all_cells["Brachyury_Status"].map({"Brachyury-positive": "red", "Brachyury-negative": "blue"})
plt.scatter(
    all_cells["Curvature"],
    all_cells["AR"],
    c=colors,
    alpha=0.6,
    label=None
)
plt.xlabel("Curvature")
plt.ylabel("Aspect Ratio")
plt.title("D3 Aspect Ratio vs. Curvature by Brachyury Status")
for status, color in [("Brachyury-positive", "red"), ("Brachyury-negative", "blue")]:
    plt.scatter([], [], label=status, color=color)
plt.legend()
plt.tight_layout()
plt.show(block=False)

import matplotlib.pyplot as plt

# Separate points by Brachyury status
pos = all_cells[all_cells["Brachyury_Status"] == "Brachyury-positive"]
neg = all_cells[all_cells["Brachyury_Status"] == "Brachyury-negative"]

# Plot
plt.figure(figsize=(6, 6))
plt.scatter(neg["X"], neg["Y"], s=10, color="gray", label="Brachyury-negative")
plt.scatter(pos["X"], pos["Y"], s=10, color="red", label="Brachyury-positive")
plt.gca().set_aspect("equal")
plt.gca().invert_yaxis()  # ← Flip Y-axis to match image orientation
plt.legend()
plt.title("D3 Brachyury Classification of Cells")
plt.xlabel("X")
plt.ylabel("Y")
plt.tight_layout()
plt.show()
