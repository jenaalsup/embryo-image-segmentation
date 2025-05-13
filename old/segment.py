from aicsimageio import AICSImage
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.io import imsave
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt
import os

# File name
filename = "SHEF6_D2N20-3dE_G3_DAPI_Factin-488_Bry-568_Sox2-647.lif"

# Load image
img = AICSImage(filename)

# Use actual scene names
scene_names = [
    'Series001', 'Series002', 'Series003', 'Series004', 'Series005',
    'Series006', 'Series007', 'Series008', 'Series009', 'Series010'
]

# Load pretrained StarDist model
model = StarDist2D.from_pretrained("2D_versatile_fluo")

# Output folder
os.makedirs("outputs", exist_ok=True)

# Loop through all specified scenes
for scene in scene_names:
    print(f"Segmenting {scene}...")

    # Set the correct scene
    img.set_scene(scene)

    # Load DAPI channel (index 0) as CZYX, then get channel 0
    dapi_stack = img.get_image_data("CZYX")[0]  # channel 0 = DAPI

    # Take middle Z-slice
    mid = dapi_stack.shape[0] // 2
    dapi_2d = dapi_stack[mid]

    # Normalize and segment
    img_norm = rescale_intensity(dapi_2d, in_range='image', out_range=(0, 1))
    labels, _ = model.predict_instances(img_norm)

    # Save results
    base_name = scene.lower()  # for filename safety
    imsave(f"outputs/{base_name}_labels.tif", labels.astype("uint16"))
    plt.imsave(f"outputs/{base_name}_preview.png", labels, cmap="nipy_spectral")

    print(f"✅ Saved output for {scene}")

print("🎉 All series segmented successfully!")
