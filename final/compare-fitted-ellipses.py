import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import EllipseModel

def normalize_ellipse(points):
    center = points.mean(axis=0)
    centered = points - center
    scale = np.sqrt((centered**2).sum(axis=1).mean())
    return centered / scale

# === Fit F-actin ellipse from ImageJ contour ===
contour = np.loadtxt("data/Ellipse_contour_D3_slice63.csv", delimiter=",", skiprows=1)[:, :2]
model1 = EllipseModel()
model1.estimate(contour)
xc1, yc1, a1, b1, theta1 = model1.params

# === Fit nuclei ellipse ===
xy = pd.read_csv("data/D3_slice63.csv")[["X", "Y"]].values
model2 = EllipseModel()
model2.estimate(xy)
xc2, yc2, a2, b2, theta2 = model2.params

# === Generate points on both ellipses ===
t = np.linspace(0, 2 * np.pi, 1000)
x1 = xc1 + a1 * np.cos(t) * np.cos(theta1) - b1 * np.sin(t) * np.sin(theta1)
y1 = yc1 + a1 * np.cos(t) * np.sin(theta1) + b1 * np.sin(t) * np.cos(theta1)
ellipse1 = np.stack([x1, y1], axis=1)

x2 = xc2 + a2 * np.cos(t) * np.cos(theta2) - b2 * np.sin(t) * np.sin(theta2)
y2 = yc2 + a2 * np.cos(t) * np.sin(theta2) + b2 * np.sin(t) * np.cos(theta2)
ellipse2 = np.stack([x2, y2], axis=1)

# === Normalize ===
ellipse1_norm = normalize_ellipse(ellipse1)
ellipse2_norm = normalize_ellipse(ellipse2)

# === Compare ===
aspect1 = min(a1, b1) / max(a1, b1)
aspect2 = min(a2, b2) / max(a2, b2)
angle1_deg = np.rad2deg(theta1)
angle2_deg = np.rad2deg(theta2)

print(f"F-actin:  aspect = {aspect1:.6f}, angle = {angle1_deg:.2f}°")
print(f"Nuclei:   aspect = {aspect2:.6f}, angle = {angle2_deg:.2f}°")

plt.figure(figsize=(6, 6))
plt.plot(ellipse1_norm[:, 0], ellipse1_norm[:, 1], label="F-actin (normalized)", color="blue", linestyle="--", linewidth=2)
plt.plot(ellipse2_norm[:, 0], ellipse2_norm[:, 1], label="Nuclei (normalized)", color="red", alpha=0.8)
plt.gca().set_aspect("equal")
plt.title("Normalized F-actin vs. Nuclei Ellipse")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.tight_layout()
plt.show(block=False)

import matplotlib.image as mpimg

# === Load the original image ===
img = mpimg.imread("images/D3_slice63_factin.png")

plt.figure(figsize=(6, 6))
plt.imshow(img, cmap="gray", origin='lower')
plt.plot(ellipse1[:, 0], ellipse1[:, 1], color="cyan", linewidth=2, label="Fitted F-actin Ellipse")
plt.title("F-actin Ellipse Overlay on Original Image")
plt.axis("off")
plt.legend()
plt.tight_layout()
plt.show()