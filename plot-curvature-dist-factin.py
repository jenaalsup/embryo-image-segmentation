import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import EllipseModel
from scipy.spatial.distance import cdist

# Load data
nuclei_df = pd.read_csv("data/D3_slice40_series4_new.csv")
contour_df = pd.read_csv("data/Ellipse_contour_D3_slice40_new.csv")

nuclei_xy = nuclei_df[["X", "Y"]].values
contour_xy = contour_df[["X", "Y"]].values

# Fit ellipse to contour
model = EllipseModel()
model.estimate(contour_xy)
xc, yc, a, b, theta = model.params
if a < b:
    a, b = b, a
    theta += np.pi / 2

# Generate ellipse points and curvature
t = np.linspace(0, 2 * np.pi, 1000)
x = xc + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
y = yc + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)
ellipse_points = np.stack([x, y], axis=1)
raw_curv = (a * b) / ((b**2 * np.cos(t)**2 + a**2 * np.sin(t)**2) ** 1.5)
curvature = raw_curv * np.sqrt(a * b)  # Normalize so circle = 1

# Match nuclei to closest ellipse point
closest_idx = np.argmin(cdist(nuclei_xy, ellipse_points), axis=1)
matched_curv = curvature[closest_idx]

# Plot
plt.figure(figsize=(7, 5))
plt.scatter(matched_curv, nuclei_df["AR"], color="red", alpha=0.6)
plt.xlabel("Curvature")
plt.ylabel("Aspect Ratio")
plt.title("Aspect Ratio vs. Curvature (New Method Only)")
plt.tight_layout()
plt.grid(False)
plt.show()
