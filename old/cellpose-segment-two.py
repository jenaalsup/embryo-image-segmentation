from aicsimageio import AICSImage
from cellpose import models
from scipy.ndimage import median_filter
from skimage.exposure import rescale_intensity
from skimage.io import imsave
import numpy as np
import matplotlib.pyplot as plt

# === Load LIF file and select Series001 ===
filename = "SHEF6_D2N20-3dE_G3_DAPI_Factin-488_Bry-568_Sox2-647.lif"
img = AICSImage(filename)
img.set_scene("Series001")

# === Extract DAPI (channel 0) as CZYX and get middle Z ===
dapi_stack = img.get_image_data("CZYX")[0]  # channel 0 = DAPI
z_mid = dapi_stack.shape[0] // 2
dapi = dapi_stack[z_mid]

# === Step 1: Median filter to reduce speckle noise ===
denoised = median_filter(dapi, size=3)

# === Step 2: Milder contrast stretching ===
p5, p95 = np.percentile(denoised, (5, 95))
if p95 - p5 < 1e-5:
    # fallback to simple normalization
    preprocessed = denoised / denoised.max()
else:
    preprocessed = rescale_intensity(denoised, in_range=(p5, p95), out_range=(0, 1))

# === Step 3: Segment with Cellpose ===
model = models.Cellpose(model_type="nuclei")
masks, flows, styles, diams = model.eval(
    [preprocessed],
    diameter=20,               # Try 35–45 for tuning
    channels=[0, 0],
    flow_threshold=0.3,        # Accept weaker signals
    do_3D=False
)
masks = masks[0]

# === Step 4: Save outputs ===
imsave("series001_nuclei_labels.tif", masks.astype("uint16"))

# Save overlay for QC
plt.figure(figsize=(8, 8))
plt.imshow(preprocessed, cmap="gray")
plt.contour(masks, levels=[0.5], colors="red", linewidths=1)
plt.title("Nuclear segmentation overlay (Series001)")
plt.axis("off")
plt.tight_layout()
plt.savefig("series001_nuclei_overlay.png", dpi=300)
plt.show()
