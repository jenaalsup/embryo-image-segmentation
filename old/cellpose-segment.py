from aicsimageio import AICSImage
from cellpose import models
from skimage.io import imsave
import matplotlib.pyplot as plt
import numpy as np
import os

# File name
filename = "SHEF6_D2N20-3dE_G3_DAPI_Factin-488_Bry-568_Sox2-647.lif"

# Use actual scene names from your file
scene_names = [
    'Series001', 'Series002', #'Series003', 'Series004', 'Series005',
    #'Series006', 'Series007', 'Series008', 'Series009', 'Series010'
]

# Load Cellpose model for nuclei
model = models.Cellpose(model_type='nuclei')

# Create output folder
os.makedirs("outputs_cellpose", exist_ok=True)

# Loop through all scenes
for scene in scene_names:
    print(f"Processing {scene}...")

    # Load and set scene
    img = AICSImage(filename)
    img.set_scene(scene)

    # Get DAPI stack and select mid-Z
    dapi_stack = img.get_image_data("CZYX")[0]  # channel 0 = DAPI
    mid = dapi_stack.shape[0] // 2
    dapi_2d = dapi_stack[mid]

    # Run Cellpose
    masks, flows, styles, diams = model.eval([dapi_2d], diameter=45, channels=[0, 0])
    masks = masks[0]  # unpack single image result

    import matplotlib.pyplot as plt
    plt.imshow(dapi_2d, cmap="gray")
    plt.contour(masks, levels=[0.5], colors='r')
    plt.title("Mask contours on raw image")
    plt.show()

    # Save outputs
    base = scene.lower()
    imsave(f"outputs_cellpose/{base}_labels.tif", masks.astype("uint16"))
    plt.imsave(f"outputs_cellpose/{base}_preview.png", masks, cmap="nipy_spectral")

    print(f"✅ Saved output for {scene}")

print("🎉 All series segmented with Cellpose!")
