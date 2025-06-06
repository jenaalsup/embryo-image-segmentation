import sys
import matplotlib.pyplot as plt

# Get args
if len(sys.argv) != 3:
    print("Usage: python segment.py [opencv|scikit|cellpose|stardist] [image_filename]")
    sys.exit(1)

method = sys.argv[1].lower()
image_path = sys.argv[2]

# ---------- OpenCV ----------
if method == 'opencv':
    import cv2
    import numpy as np

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    num_labels, labels = cv2.connectedComponents(cleaned)
    label_img = (labels.astype(np.float32) / num_labels)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(cleaned, cmap='gray')
    plt.title('Binary Mask')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(label_img, cmap='nipy_spectral')
    plt.title(f'Connected Components ({num_labels - 1} cells)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ---------- scikit-image ----------
elif method == 'scikit':
    from skimage import io, filters, measure, morphology

    img = io.imread(image_path, as_gray=True)
    blurred = filters.gaussian(img, sigma=1)
    thresh = filters.threshold_otsu(blurred)
    binary = blurred > thresh
    cleaned = morphology.remove_small_objects(binary, min_size=50)
    labeled = measure.label(cleaned)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Original')
    ax[1].imshow(labeled, cmap='nipy_spectral')
    ax[1].set_title('Segmented')
    for a in ax: a.axis('off')
    plt.tight_layout()
    plt.show()

# ---------- Cellpose ----------
elif method == 'cellpose':
    from cellpose import models
    from cellpose.io import imread

    img = imread(image_path)
    model = models.Cellpose(gpu=False, model_type='cyto')
    masks, _, _, _ = model.eval(img, diameter=None, channels=[0, 0])

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(img if img.ndim == 2 else img[..., :3])
    ax[0].set_title('Original')
    ax[1].imshow(masks, cmap='nipy_spectral')
    ax[1].set_title('Cellpose Segmentation')
    for a in ax: a.axis('off')
    plt.tight_layout()
    plt.show()

# ---------- StarDist ----------
elif method == 'stardist':
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize
    from skimage.io import imread

    img = imread(image_path)
    if img.ndim == 3:
        img = img[..., 0]
    img_norm = normalize(img, 1, 99.8)
    model = StarDist2D.from_pretrained('2D_versatile_fluo')
    labels, _ = model.predict_instances(img_norm)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Original')
    ax[1].imshow(labels, cmap='nipy_spectral')
    ax[1].set_title('StarDist Segmentation')
    for a in ax: a.axis('off')
    plt.tight_layout()
    plt.show()

else:
    print(f"Unknown method: {method}")
    sys.exit(1)


# HOW TO RUN
# python3 segmentation-playground.py scikit cells.png
# python segment.py opencv cells.png
# python segment.py cellpose cells.png
# python segment.py stardist cells.png
