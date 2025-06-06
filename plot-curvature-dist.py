import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import EllipseModel
from scipy.spatial.distance import cdist
import glob
import os
import seaborn as sns

# Load file paths
d2_files = glob.glob("data/D2_slice*.csv")
d3_files = glob.glob("data/D3_slice*.csv")
all_results = []

# Function to compute ellipse-based curvature
def compute_aspect_vs_curvature(file_path, day_label):
    df = pd.read_csv(file_path)
    xy = df[["X", "Y"]].values

    ellipse = EllipseModel()
    if not ellipse.estimate(xy):
        return None

    xc, yc, a, b, theta = ellipse.params
    t_vals = np.linspace(0, 2 * np.pi, 1000)
    x_ellipse = xc + a * np.cos(t_vals) * np.cos(theta) - b * np.sin(t_vals) * np.sin(theta)
    y_ellipse = yc + a * np.cos(t_vals) * np.sin(theta) + b * np.sin(t_vals) * np.cos(theta)
    ellipse_points = np.stack([x_ellipse, y_ellipse], axis=1)

    raw_curvature = (a * b) / ((b**2 * np.cos(t_vals)**2 + a**2 * np.sin(t_vals)**2) ** 1.5)
    normalized_curvatures = raw_curvature * np.sqrt(a * b)

    dist_matrix = cdist(xy, ellipse_points)
    closest_point_idx = np.argmin(dist_matrix, axis=1)

    return pd.DataFrame({
        "Aspect Ratio": df["AR"],
        "Curvature": normalized_curvatures[closest_point_idx],
        "Day": day_label,
        "Source": os.path.basename(file_path)
    })

# Collect data
for f in d2_files:
    res = compute_aspect_vs_curvature(f, "Day 2")
    if res is not None:
        all_results.append(res)

for f in d3_files:
    res = compute_aspect_vs_curvature(f, "Day 3")
    if res is not None:
        all_results.append(res)

final_df = pd.concat(all_results, ignore_index=True)

# Set plot style
sns.set(style="whitegrid")

# --------------------
# Scatter Plot
# --------------------
plt.figure(num=1, figsize=(7, 5))
plt.scatter(
    final_df["Curvature"],
    final_df["Aspect Ratio"],
    c=final_df["Day"].map({"Day 2": "blue", "Day 3": "red"}),
    alpha=0.6
)
plt.xlabel("Curvature")
plt.ylabel("Aspect Ratio")
plt.title("Aspect Ratio vs. Curvature")
for day in ["Day 2", "Day 3"]:
    plt.scatter([], [], label=day, color="blue" if day == "Day 2" else "red")
plt.legend()
plt.tight_layout()
plt.grid(False)
plt.savefig("scatterplot-curvature-ar.eps", format="eps")
plt.show(block=False)

# --------------------
# Density Plot (Curvature by Day) aka Smoothed Histogram
# --------------------
plt.figure(num=2, figsize=(7, 5))
sns.kdeplot(
    data=final_df,
    x="Curvature",
    hue="Day",
    common_norm=False,
    fill=True,
    palette={"Day 2": "skyblue", "Day 3": "salmon"},
    alpha=0.5
)

plt.title("Curvature Distribution by Day")
plt.xlabel("Curvature")
plt.ylabel("Density")
plt.tight_layout()
plt.grid(False)
plt.savefig("curvature-distribution.eps", format="eps")
plt.show()

# # Generate curvature vs frequency table
# num_bins = 30  # You can change this
# final_df["Curvature_Bin"] = pd.cut(final_df["Curvature"], bins=num_bins)

# # Group and count
# curvature_table = final_df.groupby(["Curvature_Bin", "Day"]).size().reset_index(name="Frequency")

# # Convert bin to midpoint
# curvature_table["Curvature_Midpoint"] = curvature_table["Curvature_Bin"].apply(lambda x: x.mid)

# # Reorder columns
# curvature_table = curvature_table[["Curvature_Midpoint", "Day", "Frequency"]]

# # Save to CSV
# curvature_table.to_csv("curvature_vs_frequency.csv", index=False)
# print("Saved curvature_vs_frequency.csv")