import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import EllipseModel
import matplotlib.image as mpimg

def normalize_ellipse(points):
    center = points.mean(axis=0)
    centered = points - center
    scale = np.sqrt((centered**2).sum(axis=1).mean())
    return centered / scale

# === Load nuclei image early to get shape ===
img_nuclei = mpimg.imread("images/D3-slice63.png")
height, width = img_nuclei.shape[:2]

# === Fit F-actin ellipse from ImageJ contour ===
contour = np.loadtxt("data/Ellipse_contour_D3_slice63.csv", delimiter=",", skiprows=1)[:, :2]
model1 = EllipseModel()
model1.estimate(contour)
xc1, yc1, a1, b1, theta1 = model1.params
if a1 < b1:
    a1, b1 = b1, a1
    theta1 += np.pi / 2

# === Fit nuclei ellipse ===
xy = pd.read_csv("data/D3_slice63.csv")[["X", "Y"]].values

# Debug: Print original data range
print(f"Original data range - X: [{xy[:, 0].min():.6f}, {xy[:, 0].max():.6f}]")
print(f"Original data range - Y: [{xy[:, 1].min():.6f}, {xy[:, 1].max():.6f}]")
print(f"Image dimensions: {width} x {height}")

print(f"After scaling - X: [{xy[:, 0].min():.1f}, {xy[:, 0].max():.1f}]")
print(f"After scaling - Y: [{xy[:, 1].min():.1f}, {xy[:, 1].max():.1f}]")

# FIXED: Apply Y-flip to match image coordinate system
# The nuclei coordinates appear to be in a different coordinate system than the image
xy_original = xy.copy()  # Keep original for debugging
#xy[:, 1] = height - xy[:, 1]  # Flip Y coordinates

print(f"After Y-flip - X: [{xy[:, 0].min():.1f}, {xy[:, 0].max():.1f}]")
print(f"After Y-flip - Y: [{xy[:, 1].min():.1f}, {xy[:, 1].max():.1f}]")

# Fit ellipse with corrected coordinates
model2 = EllipseModel()
success = model2.estimate(xy)
print(f"Ellipse fitting successful: {success}")

if success:
    xc2, yc2, a2, b2, theta2 = model2.params
    print(f"Ellipse parameters before adjustment:")
    print(f"  Center: ({xc2:.1f}, {yc2:.1f})")
    print(f"  Semi-axes: a={a2:.1f}, b={b2:.1f}")
    print(f"  Angle: {np.rad2deg(theta2):.1f}°")
    
    if a2 < b2:
        a2, b2 = b2, a2
        theta2 += np.pi / 2
        print(f"Swapped axes - new a={a2:.1f}, b={b2:.1f}, angle={np.rad2deg(theta2):.1f}°")
else:
    print("ERROR: Ellipse fitting failed!")
    # Set default values to prevent crashes
    xc2, yc2, a2, b2, theta2 = width/2, height/2, 100, 100, 0

# === Generate points on both ellipses ===
t = np.linspace(0, 2 * np.pi, 1000)
x1 = xc1 + a1 * np.cos(t) * np.cos(theta1) - b1 * np.sin(t) * np.sin(theta1)
y1 = yc1 + a1 * np.cos(t) * np.sin(theta1) + b1 * np.sin(t) * np.cos(theta1)
ellipse1 = np.stack([x1, y1], axis=1)

x2 = xc2 + a2 * np.cos(t) * np.cos(theta2) - b2 * np.sin(t) * np.sin(theta2)
y2 = yc2 + a2 * np.cos(t) * np.sin(theta2) + b2 * np.sin(t) * np.cos(theta2)
ellipse2 = np.stack([x2, y2], axis=1)

# For display with origin='upper', the ellipse2 coordinates are already correct
ellipse2_display = ellipse2.copy()

# # === Normalize for comparison plot only ===
# ellipse1_norm = normalize_ellipse(ellipse1)
# ellipse2_norm = normalize_ellipse(ellipse2)

# # === Compare aspect ratios and angles ===
# aspect1 = min(a1, b1) / max(a1, b1)
# aspect2 = min(a2, b2) / max(a2, b2)
# angle1_deg = np.rad2deg(theta1)
# angle2_deg = np.rad2deg(theta2)
# print(f"F-actin:  aspect = {aspect1:.6f}, angle = {angle1_deg:.2f}°")
# print(f"Nuclei:   aspect = {aspect2:.6f}, angle = {angle2_deg:.2f}°")

# # === Plot normalized ellipses for comparison ===
# plt.figure(figsize=(6, 6))
# plt.plot(ellipse1_norm[:, 0], ellipse1_norm[:, 1], label="F-actin (normalized)", color="blue", linestyle="--", linewidth=2)
# plt.plot(ellipse2_norm[:, 0], ellipse2_norm[:, 1], label="Nuclei (normalized)", color="red", alpha=0.8)
# plt.gca().set_aspect("equal")
# plt.title("Normalized F-actin vs. Nuclei Ellipse")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.legend()
# plt.tight_layout()
# plt.show(block=False)

# === Plot F-actin ellipse over F-actin image ===
img_factin = mpimg.imread("images/D3_slice63_factin.png")
plt.figure(figsize=(6, 6))
plt.imshow(img_factin, cmap="gray", origin='lower')
plt.plot(ellipse1[:, 0], ellipse1[:, 1], color="cyan", linewidth=2, label="Fitted F-actin Ellipse")
plt.title("F-actin Ellipse Overlay on Original Image")
plt.axis("off")
plt.legend()
plt.tight_layout()
plt.show(block=False)

# === Comprehensive Nuclei Analysis Figure ===
# Use the corrected coordinates for display
xy_display = xy.copy()  # xy already has the corrected coordinates

# Calculate zoom region with bounds checking
margin = 100
x_min = max(0, int(xy[:, 0].min() - margin))
x_max = min(width, int(xy[:, 0].max() + margin))
y_min = max(0, int(xy[:, 1].min() - margin))
y_max = min(height, int(xy[:, 1].max() + margin))

# Crop the image for zoomed views
img_nuclei_crop = img_nuclei[y_min:y_max, x_min:x_max]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Full image
axes[0].imshow(img_nuclei, cmap="gray", origin='upper')
axes[0].set_title("Original Nuclei Image (Full)")
axes[0].axis("off")

# Zoomed nuclei segmentation points
axes[1].imshow(img_nuclei, cmap="gray", origin='upper', extent=[x_min, x_max, y_max, y_min])
axes[1].scatter(xy_display[:, 0], xy_display[:, 1], s=3, color='yellow', label='Raw nuclei points')
axes[1].set_title("Nuclei Segmentation (Zoomed)")
axes[1].axis("off")
axes[1].legend()

# Zoomed fitted ellipse overlay
axes[2].imshow(img_nuclei, cmap="gray", origin='upper', extent=[x_min, x_max, y_max, y_min])
axes[2].plot(ellipse2_display[:, 0], ellipse2_display[:, 1], color="magenta", linewidth=3, label="Fitted Nuclei Ellipse")
axes[2].scatter(xy_display[:, 0], xy_display[:, 1], s=1, color='yellow', alpha=0.7, label='Raw nuclei points')
axes[2].set_title("Fitted Ellipse Overlay (Zoomed)")
axes[2].axis("off")
axes[2].legend()

plt.tight_layout()
plt.show(block=False)

# === Comprehensive F-actin Analysis Figure ===
# Calculate zoom region for F-actin with bounds checking
factin_margin = 100
x_min_factin = max(0, int(ellipse1[:, 0].min() - factin_margin))
x_max_factin = min(img_factin.shape[1], int(ellipse1[:, 0].max() + factin_margin))
y_min_factin = max(0, int(ellipse1[:, 1].min() - factin_margin))
y_max_factin = min(img_factin.shape[0], int(ellipse1[:, 1].max() + factin_margin))

# Crop the F-actin image for zoomed views
img_factin_crop = img_factin[y_min_factin:y_max_factin, x_min_factin:x_max_factin]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Full F-actin image
axes[0].imshow(img_factin, cmap="gray", origin='lower')
axes[0].set_title("Original F-actin Image (Full)")
axes[0].axis("off")

# Zoomed F-actin contour points
axes[1].imshow(img_factin_crop, cmap="gray", origin='lower', extent=[x_min_factin, x_max_factin, y_min_factin, y_max_factin])
axes[1].scatter(contour[:, 0], contour[:, 1], s=1, color='yellow', label='F-actin contour points')
axes[1].set_title("F-actin Contour (Zoomed)")
axes[1].axis("off")
axes[1].legend()

# Zoomed F-actin ellipse overlay
axes[2].imshow(img_factin_crop, cmap="gray", origin='lower', extent=[x_min_factin, x_max_factin, y_min_factin, y_max_factin])
axes[2].plot(ellipse1[:, 0], ellipse1[:, 1], color="cyan", linewidth=3, label="Fitted F-actin Ellipse")
axes[2].scatter(contour[:, 0], contour[:, 1], s=1, color='yellow', alpha=0.7, label='F-actin contour points')
axes[2].set_title("Fitted F-actin Ellipse Overlay (Zoomed)")
axes[2].axis("off")
axes[2].legend()

plt.tight_layout()
plt.show(block=False)

# === Additional debugging plot to verify coordinate correction ===
plt.figure(figsize=(12, 6))

# Before correction
plt.subplot(1, 2, 1)
plt.imshow(img_nuclei, cmap="gray", origin='upper')
plt.scatter(xy_original[:, 0], height - xy_original[:, 1], s=2, color='red', alpha=0.7, label='Original coordinates (flipped for display)')
plt.title("Before Coordinate Correction")
plt.axis("off")
plt.legend()

# After correction
plt.subplot(1, 2, 2)
plt.imshow(img_nuclei, cmap="gray", origin='upper')
plt.scatter(xy[:, 0], xy[:, 1], s=2, color='yellow', alpha=0.7, label='Corrected coordinates')
plt.plot(ellipse2[:, 0], ellipse2[:, 1], color="magenta", linewidth=2, label="Fitted ellipse")
plt.title("After Coordinate Correction")
plt.axis("off")
plt.legend()

plt.tight_layout()
plt.show(block=False)
# --- Normalize F-actin ellipse to [0, 1] ---
ellipse1_norm = ellipse1.copy()
ellipse1_norm[:, 0] /= width
ellipse1_norm[:, 1] /= height

# --- Normalize nuclei ellipse to [0, 1] while preserving shape ---
ellipse2_norm = ellipse2_display.copy()
ellipse2_norm[:, 0] /= width
ellipse2_norm[:, 1] /= height
ellipse2_norm[:, 1] = 1.0 - ellipse2_norm[:, 1]  # Flip Y to match image coordinate convention

# --- Compute centers ---
center1 = ellipse1_norm.mean(axis=0)
center2 = ellipse2_norm.mean(axis=0)

# --- Compute bounding box size for scaling ---
f_size = np.ptp(ellipse1_norm, axis=0)  # F-actin width and height
n_size = np.ptp(ellipse2_norm, axis=0)  # Nuclei width and height
scale_factor = f_size / n_size

# --- Scale nuclei around its center to match F-actin size ---
ellipse2_scaled = (ellipse2_norm - center2) * scale_factor + center2

# --- Recompute center after scaling and align ---
center2_scaled = ellipse2_scaled.mean(axis=0)
shift = center1 - center2_scaled
ellipse2_aligned = ellipse2_scaled + shift

# --- Plot both ellipses in normalized [0,1] space ---
plt.figure(figsize=(6, 6))
plt.plot(ellipse1_norm[:, 0], ellipse1_norm[:, 1], label="F-actin ellipse", color="blue", linestyle="--", linewidth=2)
plt.plot(ellipse2_aligned[:, 0], ellipse2_aligned[:, 1], label="Nuclei ellipse", color="red", alpha=0.8)
plt.gca().set_aspect("equal")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title("Comparison of F-actin and Nuclei Ellipses (Normalized, Sized, Center-Aligned)")
plt.xlabel("X (normalized)")
plt.ylabel("Y (normalized)")
plt.legend()
plt.tight_layout()
plt.show()
