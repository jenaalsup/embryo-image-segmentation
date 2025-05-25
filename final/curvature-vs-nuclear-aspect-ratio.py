import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import EllipseModel
from scipy.spatial.distance import cdist
import glob
import os
import seaborn as sns

d2_files = glob.glob("data/D2_slice*.csv")
d3_files = glob.glob("data/D3_slice*.csv")
all_results = []

def compute_aspect_vs_curvature(file_path, day_label):
    df = pd.read_csv(file_path)

    xy = df[["X", "Y"]].values

    # fit ellipse
    ellipse = EllipseModel()
    ellipse.estimate(xy)
    xc, yc, a, b, theta = ellipse.params

    # generate 1000 points along the ellipse
    t_vals = np.linspace(0, 2 * np.pi, 1000)
    x_ellipse = xc + a * np.cos(t_vals) * np.cos(theta) - b * np.sin(t_vals) * np.sin(theta)
    y_ellipse = yc + a * np.cos(t_vals) * np.sin(theta) + b * np.sin(t_vals) * np.cos(theta)
    ellipse_points = np.stack([x_ellipse, y_ellipse], axis=1)

    # compute curvature at each point
    raw_curvature = (a * b) / ((b**2 * np.cos(t_vals)**2 + a**2 * np.sin(t_vals)**2) ** 1.5)
    normalized_curvatures = raw_curvature * np.sqrt(a * b)

    # match each nucleus to nearest ellipse point
    dist_matrix = cdist(xy, ellipse_points)
    closest_point_idx = np.argmin(dist_matrix, axis=1)

    result = pd.DataFrame({
        "Aspect Ratio": df["AR"],
        "Curvature": normalized_curvatures[closest_point_idx],
        "Day": day_label,
        "Source": os.path.basename(file_path)
    })

    return result

for f in d2_files:
    res = compute_aspect_vs_curvature(f, "Day 2")
    if res is not None:
        all_results.append(res)
for f in d3_files:
    res = compute_aspect_vs_curvature(f, "Day 3")
    if res is not None:
        all_results.append(res)

final_df = pd.concat(all_results, ignore_index=True)

# plot aspect ratio vs. curvature
plt.figure(figsize=(7, 5))
scatter = plt.scatter(
    final_df["Curvature"],
    final_df["Aspect Ratio"],
    c=final_df["Day"].map({"Day 2": "blue", "Day 3": "red"}),
    alpha=0.6,
    label=None
)
plt.xlabel("Curvature")
plt.ylabel("Aspect Ratio")
plt.title("Aspect Ratio vs. Curvature (Colored by Day)")
for day in ["Day 2", "Day 3"]:
    plt.scatter([], [], label=day, color="blue" if day == "Day 2" else "red")
plt.legend()
plt.tight_layout()
plt.show()
sns.lmplot(data=final_df, x="Curvature", y="Aspect Ratio", hue="Day")
