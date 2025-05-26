import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load points (adjust delimiter if needed)
contour = np.loadtxt("data/ellipse-contour.csv", delimiter=",", skiprows=1)  # shape (N, 2)
contour = contour[:, :2].astype(np.float32)

# Reshape to OpenCV format
contour = contour.reshape(-1, 1, 2)

# Fit ellipse
ellipse = cv2.fitEllipse(contour)

# Create a blank canvas
canvas = np.zeros((2048, 2048), dtype=np.uint8)

# Draw the contour
cv2.drawContours(canvas, [contour.astype(np.int32)], -1, 255, 1)

# Draw the ellipse
cv2.ellipse(canvas, ellipse, (207, 255, 4), 2)  

# Show result
plt.imshow(canvas, cmap='gray')
plt.title("Fitted Ellipse")
plt.show()
