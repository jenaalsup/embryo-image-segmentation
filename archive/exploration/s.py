import numpy as np
import matplotlib.pyplot as plt
from aicsimageio import AICSImage
from skimage import filters, measure, morphology
import pandas as pd
import os

# === File and series settings ===
filename = "SHEF6_D2N20-3dE_G3_DAPI_Factin-488_Bry-568_Sox2-647.lif"
series_index = 8  # Series 8 = high-res 3D stack
channel_index = 0  # DAPI is channel 1, which is index 0 (0-based)

# === Load the file ===
img = AICSImage(filename)
print("Image shape:", img.dims)

# Get the DAPI channel (C, Z, Y, X)
dapi_stack = img.get_image_data("CZYX")[channel_index]  # shape: (Z, Y, X)

# === Preprocessing ===
# Smooth the image
from scipy.ndimage import gaussian_filter
blurred = gaussian_filter(dapi_stack, sigma=1)

# Apply threshold
threshold = filters.threshold_otsu(blurred)
binary = blurred > threshold

# Clean up noise
binary = morphology.remove_small_objects(binary, min_size=50)
binary = morphology.binary_closing(binary)

# === Label and measure nuclei ===
labeled = measure.label(binary)
props = measure.regionprops_table(labeled, properties=["label", "area", "centroid", "bbox"])

# === Save stats ===
df = pd.DataFrame(props)
df.to_csv("nuclear_segmentation_stats.csv", index=False)

# === Save mid-Z visualization ===
mid_z = dapi_stack.shape[0] // 2
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].imshow(dapi_stack[mid_z], cmap="gray")
ax[0].set_title("Raw DAPI (mid-Z)")
ax[1].imshow(labeled[mid_z], cmap="nipy_spectral")
ax[1].set_title("Segmented Nuclei (mid-Z)")
plt.tight_layout()
plt.savefig("nuclear_segmentation_midZ.png")
plt.show()
