# used to compare new nuclei segmentation vs. factin (inner membrane) segmentation

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

def align_ellipses(ellipse1, ellipse2):
    e1 = ellipse1.copy()
    e2 = ellipse2.copy()

    # Center both at origin
    center1 = e1.mean(axis=0)
    center2 = e2.mean(axis=0)
    e1_centered = e1 - center1
    e2_centered = e2 - center2

    # Uniform scale so that both have same size in major axis
    scale1 = np.max(np.ptp(e1_centered, axis=0))
    scale2 = np.max(np.ptp(e2_centered, axis=0))
    scale = scale1 / scale2
    e2_scaled = e2_centered * scale

    return e1_centered, e2_scaled

# ------------------------------
# Load Data
# ------------------------------

img_nuclei = mpimg.imread("images/D3-slice40-new.png")
img_factin = mpimg.imread("images/D3-slice40-factin-new.png")
height, width = img_nuclei.shape[:2]

contour = np.loadtxt("data/Ellipse_contour_D3_slice40_new.csv", delimiter=",", skiprows=1)[:, :2]
xy_nuclei = pd.read_csv("data/D3_slice40_series4_new.csv")[["X", "Y"]].values

ellipse_factin = fit_ellipse(contour)
ellipse_nuclei = fit_ellipse(xy_nuclei)

# ------------------------------
# Plot Nuclei: Full Image, Raw Points, and Ellipse
# ------------------------------

margin = 0
xmin, xmax = 0, width
ymin, ymax = 0, height
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Compute original bounds of points
x_min, x_max = xy_nuclei[:, 0].min(), xy_nuclei[:, 0].max()
y_min, y_max = xy_nuclei[:, 1].min(), xy_nuclei[:, 1].max()

# Define zoom region
pad = 20
x_lo, x_hi = x_min - pad, x_max + pad
y_lo, y_hi = y_min - pad, y_max + pad

# Compute center and scale
center_x = (x_lo + x_hi) / 2
center_y = (y_lo + y_hi) / 2 + 10
zoom_width = x_hi - x_lo
zoom_height = y_hi - y_lo
scale = min(width / zoom_width, height / zoom_height)  # uniform scale!

# Apply transform: shift to origin, scale, then recenter on image
def transform_points(points):
    shifted = points - np.array([center_x, center_y])
    scaled = shifted * scale
    recentered = scaled + np.array([width / 2, height / 2])
    return recentered

xy_zoomed = transform_points(xy_nuclei)
ellipse_zoomed = transform_points(ellipse_nuclei)

# Panel 1: original
axes[0].imshow(img_nuclei, cmap="gray", origin="lower")
axes[0].set_title("Original Nuclei Image")
axes[0].axis("off")

# Panel 2: zoomed overlay only
axes[1].imshow(img_nuclei, cmap="gray", origin="lower")
axes[1].scatter(xy_zoomed[:, 0], xy_zoomed[:, 1], s=3, color='yellow', label="Nuclei Points")
axes[1].set_title("Nuclei Segmentation")
axes[1].axis("off")
axes[1].legend()

# Panel 3: zoomed ellipse and overlay
axes[2].imshow(img_nuclei, cmap="gray", origin="lower")
axes[2].plot(ellipse_zoomed[:, 0], ellipse_zoomed[:, 1], color="magenta", linewidth=3, label="Nuclei Ellipse")
axes[2].scatter(xy_zoomed[:, 0], xy_zoomed[:, 1], s=1, color='yellow', alpha=0.7)
axes[2].set_title("Nuclei Ellipse Overlay")
axes[2].axis("off")
axes[2].legend()

plt.tight_layout()
plt.show(block=False)


# ------------------------------
# Plot F-actin: Zoom with Contour and Ellipse
# ------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Compute bounds and transformation for contour
x_min_f, x_max_f = contour[:, 0].min(), contour[:, 0].max()
y_min_f, y_max_f = contour[:, 1].min(), contour[:, 1].max()

pad_f = 50
x_lo_f, x_hi_f = x_min_f - pad_f, x_max_f + pad_f
y_lo_f, y_hi_f = y_min_f - pad_f, y_max_f + pad_f

center_x_f = (x_lo_f + x_hi_f) / 2 + 25
center_y_f = (y_lo_f + y_hi_f) / 2 + 45
zoom_width_f = x_hi_f - x_lo_f
zoom_height_f = y_hi_f - y_lo_f
scale_f = 0.65 * min(width / zoom_width_f, height / zoom_height_f)

def transform_factin(points):
    shifted = points - np.array([center_x_f, center_y_f])
    scaled = shifted * scale_f
    recentered = scaled + np.array([width / 2, height / 2])
    return recentered

contour_zoomed = transform_factin(contour)
ellipse_factin_zoomed = transform_factin(ellipse_factin)

# Panel 1: Original image
axes[0].imshow(img_factin, cmap="gray", origin="lower")
axes[0].set_title("Original F-actin Image")
axes[0].axis("off")

# Panel 2: Zoomed contour overlay
axes[1].imshow(img_factin, cmap="gray", origin="lower")
axes[1].scatter(contour_zoomed[:, 0], contour_zoomed[:, 1], s=1, color="yellow", label="F-actin Contour")
axes[1].set_xlim([0, width])
axes[1].set_ylim([0, height])
axes[1].set_title("Contour Points")
axes[1].axis("off")
axes[1].legend()

# Panel 3: Zoomed ellipse + contour
axes[2].imshow(img_factin, cmap="gray", origin="lower")
axes[2].plot(ellipse_factin_zoomed[:, 0], ellipse_factin_zoomed[:, 1], color="cyan", linewidth=3, label="F-actin Ellipse")
axes[2].scatter(contour_zoomed[:, 0], contour_zoomed[:, 1], s=1, color="yellow", alpha=0.7)
axes[2].set_xlim([0, width])
axes[2].set_ylim([0, height])
axes[2].set_title("Ellipse Overlay")
axes[2].axis("off")
axes[2].legend()

plt.tight_layout()
plt.show(block=False)

# ------------------------------
# Normalize and Overlay Comparison (using zoomed ellipses)
# ------------------------------

def compute_aspect_ratio(points):
    cov = np.cov(points.T)
    eigvals = np.linalg.eigvalsh(cov)
    a, b = np.sqrt(np.sort(eigvals)[::-1])  # a = major, b = minor
    return b / a

e1_aligned, e2_aligned = align_ellipses(ellipse_factin, ellipse_nuclei)

# Compute and print aspect ratios
ar1 = compute_aspect_ratio(e1_aligned)
ar2 = compute_aspect_ratio(e2_aligned)
print(f"F-actin ellipse aspect ratio (minor/major): {ar1:.4f}")
print(f"Nuclei ellipse aspect ratio (minor/major): {ar2:.4f}")

# Plot the aligned ellipses
plt.figure(figsize=(6, 6))
plt.plot(e1_aligned[:, 0], e1_aligned[:, 1], label="F-actin", color="cyan")
plt.plot(e2_aligned[:, 0], e2_aligned[:, 1], label="Nuclei", color="magenta")
plt.gca().set_aspect("equal")
plt.title("Aligned Ellipse Comparison (Preserved Aspect Ratio)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.tight_layout()
plt.savefig("ellipse-comparison-D3-slice40-new.eps", format="eps")
plt.show()
