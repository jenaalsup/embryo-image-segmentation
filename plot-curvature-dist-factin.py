import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.measure import EllipseModel
from scipy.spatial.distance import cdist

# Load data
nuclei_df = pd.read_csv("data/D3_slice40_series4_new.csv")
contour_df = pd.read_csv("data/Ellipse_contour_D3_slice40_new.csv")
nuclei_xy = nuclei_df[["X", "Y"]].values
contour_xy = contour_df[["X", "Y"]].values

# --- Fit ellipse to F-actin (contour) ---
model = EllipseModel()
model.estimate(contour_xy)
xc, yc, a, b, theta = model.params
if a < b:
    a, b = b, a
    theta += np.pi / 2
t = np.linspace(0, 2 * np.pi, 1000)
ellipse_points = np.stack([
    xc + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta),
    yc + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)
], axis=1)
curvature_factin = (a * b) / ((b**2 * np.cos(t)**2 + a**2 * np.sin(t)**2) ** 1.5)
curvature_factin /= np.max(curvature_factin)
matched_curv_factin = curvature_factin[np.argmin(cdist(nuclei_xy, ellipse_points), axis=1)]

# --- Fit ellipse to nuclei centers ---
model_old = EllipseModel()
model_old.estimate(nuclei_xy)
xc_o, yc_o, a_o, b_o, theta_o = model_old.params
if a_o < b_o:
    a_o, b_o = b_o, a_o
    theta_o += np.pi / 2
t_old = np.linspace(0, 2 * np.pi, 1000)
ellipse_points_old = np.stack([
    xc_o + a_o * np.cos(t_old) * np.cos(theta_o) - b_o * np.sin(t_old) * np.sin(theta_o),
    yc_o + a_o * np.cos(t_old) * np.sin(theta_o) + b_o * np.sin(t_old) * np.cos(theta_o)
], axis=1)
curvature_nuclei = (a_o * b_o) / ((b_o**2 * np.cos(t_old)**2 + a_o**2 * np.sin(t_old)**2) ** 1.5)
curvature_nuclei /= np.max(curvature_nuclei)
matched_curv_nuclei = curvature_nuclei[np.argmin(cdist(nuclei_xy, ellipse_points_old), axis=1)]

# --- Scatter plot ---
plt.figure(figsize=(7, 5))
plt.scatter(matched_curv_nuclei, nuclei_df["AR"], color="blue", alpha=0.6, label="Nuclei (Old)")
plt.scatter(matched_curv_factin, nuclei_df["AR"], color="red", alpha=0.6, label="F-actin (New)")
plt.xlabel("Curvature (Normalized)")
plt.ylabel("Aspect Ratio")
plt.title("Aspect Ratio vs. Curvature — Nuclei vs. F-actin")
plt.legend()
plt.tight_layout()
plt.grid(False)
plt.show(block=False)

# --- Density plot ---
df_density = pd.DataFrame({
    "Curvature": np.concatenate([matched_curv_nuclei, matched_curv_factin]),
    "Method": ["Nuclei"] * len(matched_curv_nuclei) + ["Factin"] * len(matched_curv_factin)
})

plt.figure(figsize=(7, 5))
sns.kdeplot(
    data=df_density,
    x="Curvature",
    hue="Method",
    common_norm=False,
    fill=True,
    palette={"Nuclei": "blue", "Factin": "red"},
    alpha=0.4
)
plt.title("Curvature Distribution — Nuclei vs. F-actin Method (on one slice)")
plt.xlabel("Curvature (Normalized)")
plt.ylabel("Density")
plt.tight_layout()
plt.grid(False)
plt.show()
