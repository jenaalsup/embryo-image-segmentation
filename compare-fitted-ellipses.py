import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.measure import EllipseModel
from scipy.spatial.distance import pdist

# ------------------------------
# Utility Functions
# ------------------------------

def fit_ellipse(points):
    model = EllipseModel()
    model.estimate(points)
    xc, yc, a, b, theta = model.params
    if a < b:
        a, b = b, a
        theta += np.pi / 2
    t = np.linspace(0, 2 * np.pi, 1000)
    x = xc + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
    y = yc + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)
    return np.stack([x, y], axis=1)

def compute_aspect_ratio(points):
    cov = np.cov(points.T)
    eigvals = np.linalg.eigvalsh(cov)
    a, b = np.sqrt(np.sort(eigvals)[::-1])
    return b / a

def normalize_and_align(ellipse1, ellipse2, width, height):
    e1 = ellipse1.copy()
    e1[:, 0] /= width
    e1[:, 1] /= height

    e2 = ellipse2.copy()
    e2[:, 0] /= width
    e2[:, 1] = 1.0 - (e2[:, 1] / height)

    center1 = e1.mean(axis=0)
    center2 = e2.mean(axis=0)
    scale = np.ptp(e1, axis=0) / np.ptp(e2, axis=0)

    e2_scaled = (e2 - center2) * scale + center2
    shift = center1 - e2_scaled.mean(axis=0)
    return e1, e2_scaled + shift

# ------------------------------
# Load Data
# ------------------------------

img_nuclei = mpimg.imread("images/D3-slice63.png")
img_factin = mpimg.imread("images/D3-slice63-factin.png")
height, width = img_nuclei.shape[:2]

contour = np.loadtxt("data/Ellipse_contour_D3_slice63.csv", delimiter=",", skiprows=1)[:, :2]
xy_nuclei = pd.read_csv("data/D3_slice63_series1.csv")[["X", "Y"]].values

ellipse_factin = fit_ellipse(contour)
ellipse_nuclei = fit_ellipse(xy_nuclei)

# ------------------------------
# Plot F-actin Ellipse
# ------------------------------

plt.figure(figsize=(6, 6))
plt.imshow(img_factin, cmap="gray", origin="lower")
plt.plot(ellipse_factin[:, 0], ellipse_factin[:, 1], color="cyan", linewidth=2, label="F-actin Ellipse")
plt.title("F-actin Ellipse Overlay")
plt.axis("off")
plt.legend()
plt.tight_layout()
plt.show(block=False)

# ------------------------------
# Plot Nuclei: Full Image, Raw Points, and Ellipse
# ------------------------------

margin = 100
xmin, xmax = max(0, int(xy_nuclei[:, 0].min() - margin)), min(width, int(xy_nuclei[:, 0].max() + margin))
ymin, ymax = max(0, int(xy_nuclei[:, 1].min() - margin)), min(height, int(xy_nuclei[:, 1].max() + margin))

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(img_nuclei, cmap="gray", origin="upper", extent=[xmin, xmax, ymax, ymin])
axes[0].set_title("Original Nuclei Image")
axes[0].axis("off")

axes[1].imshow(img_nuclei, cmap="gray", origin="upper", extent=[xmin, xmax, ymax, ymin])
axes[1].scatter(xy_nuclei[:, 0], xy_nuclei[:, 1], s=3, color='yellow', label="Nuclei Points")
axes[1].set_title("Nuclei Segmentation")
axes[1].axis("off")
axes[1].legend()

axes[2].imshow(img_nuclei, cmap="gray", origin="upper", extent=[xmin, xmax, ymax, ymin])
axes[2].plot(ellipse_nuclei[:, 0], ellipse_nuclei[:, 1], color="magenta", linewidth=3, label="Nuclei Ellipse")
axes[2].scatter(xy_nuclei[:, 0], xy_nuclei[:, 1], s=1, color='yellow', alpha=0.7)
axes[2].set_title("Nuclei Ellipse Overlay")
axes[2].axis("off")
axes[2].legend()

plt.tight_layout()
plt.show(block=False)

# ------------------------------
# Plot F-actin: Zoom with Contour and Ellipse
# ------------------------------

xmin_f = max(0, int(ellipse_factin[:, 0].min() - margin))
xmax_f = min(width, int(ellipse_factin[:, 0].max() + margin))
ymin_f = max(0, int(ellipse_factin[:, 1].min() - margin))
ymax_f = min(height, int(ellipse_factin[:, 1].max() + margin))
img_factin_crop = img_factin[ymin_f:ymax_f, xmin_f:xmax_f]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(img_factin_crop, cmap="gray", origin="lower", extent=[xmin_f, xmax_f, ymin_f, ymax_f])
axes[0].set_title("Original F-actin Image")
axes[0].axis("off")

axes[1].imshow(img_factin_crop, cmap="gray", origin="lower", extent=[xmin_f, xmax_f, ymin_f, ymax_f])
axes[1].scatter(contour[:, 0], contour[:, 1], s=1, color="yellow", label="F-actin Contour")
axes[1].set_title("Contour Points")
axes[1].axis("off")
axes[1].legend()

axes[2].imshow(img_factin_crop, cmap="gray", origin="lower", extent=[xmin_f, xmax_f, ymin_f, ymax_f])
axes[2].plot(ellipse_factin[:, 0], ellipse_factin[:, 1], color="cyan", linewidth=3, label="F-actin Ellipse")
axes[2].scatter(contour[:, 0], contour[:, 1], s=1, color="yellow", alpha=0.7)
axes[2].set_title("Ellipse Overlay")
axes[2].axis("off")
axes[2].legend()

plt.tight_layout()
plt.show(block=False)

# ------------------------------
# Normalize and Overlay Comparison
# ------------------------------

ellipse1_norm, ellipse2_aligned = normalize_and_align(ellipse_factin, ellipse_nuclei, width, height)
ar1 = compute_aspect_ratio(ellipse1_norm)
ar2 = compute_aspect_ratio(ellipse2_aligned)

print(f"F-actin ellipse aspect ratio (minor/major): {ar1:.4f}")
print(f"Nuclei ellipse aspect ratio (minor/major): {ar2:.4f}")

plt.figure(figsize=(6, 6))
plt.plot(ellipse1_norm[:, 0], ellipse1_norm[:, 1], label="F-actin", color="blue", linestyle="--", linewidth=2)
plt.plot(ellipse2_aligned[:, 0], ellipse2_aligned[:, 1], label="Nuclei", color="red", alpha=0.8)
plt.gca().set_aspect("equal")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title("Normalized Ellipse Comparison")
plt.xlabel("X (normalized)")
plt.ylabel("Y (normalized)")
plt.legend()
plt.tight_layout()
#plt.savefig("ellipse-comparison-D3-slice63.eps", format="eps")
plt.show()
