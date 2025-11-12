"""
3D Outer Lumen Segmentation

Prerequisites
-------------
export IMAGE_PATH="..." # path to the image e.g. /Users/jenaalsup/Desktop/CKHRJQ~2.TIF
pip install SimpleITK scipy scikit-image numpy

python3 segment-outer.py
"""

import os
import time
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter, median_filter, grey_closing, binary_fill_holes, label
from skimage.morphology import ball, disk, binary_opening, binary_closing, remove_small_objects, remove_small_holes


# --------------------------------------------------- #
# Config
# --------------------------------------------------- #
IMAGE_PATH = os.path.expanduser(os.environ.get("IMAGE_PATH", os.path.expanduser("~/Desktop/CKHRJQ~2.TIF")))
_base = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
_dir  = os.path.dirname(os.path.abspath(IMAGE_PATH))
OUT_TIF = os.path.join(_dir, f"{_base}_outer_labels.tif")

# Preprocess
GAUSS_SIGMA       = (1.2, 1.2, 1.2)    # increase for noisier images; too high will blur/thicken edges
DESPECKLE_SIZE    = 3                  # raise to remove speckles; can erode thin structures if too large
GREY_CLOSE_RAD    = 1                  # raise to close small gaps; too high may merge nearby boundaries

# Threshold
THRESH_VAL_U8     = 30                 # lower if image is dim/low contrast (include more), higher to be stricter (exclude noise)

# Cleanup before CC
OPEN_RAD_3D       = 2                   # raise to remove small 3D specks; too high can thin or break narrow walls

# Keep lumens (3D)
MIN_SIZE_VOX      = 15000              # raise to ignore tiny blobs; lower if lumens are small

# Per-slice “fill inside outline”
SL_CLOSE_RAD      = 8                   # per-slice gap closing; raise to bridge gaps, may over-round
SL_OPEN_RAD       = 0                   # per-slice spur removal; raise to remove thin artifacts, may erode edges
SL_MIN_OBJECT_2D  = 400                 # drop small per-slice fragments; lower to keep finer fragments
SL_HOLE_AREA_2D   = 500_000            # fill holes up to this area per slice; raise to fill larger cavities

# Final 3D smoothing
FINAL_CLOSE_RAD   = 3                  # final smoothing/closing; raise to smooth more, may slightly inflate
FINAL_OPEN_RAD    = 0                  # final opening; raise to remove small bumps, may slightly erode


# --------------------------------------------------- #
# Helper Functions
# --------------------------------------------------- #

def load_and_normalize(path):
  """Load image and normalize intensities to uint8 (X,Y,Z)."""
  img_itk = sitk.ReadImage(path)
  arr_zyx = sitk.GetArrayFromImage(img_itk)                  # (Z,Y,X)
  arr_xyz = np.transpose(arr_zyx, (2, 1, 0)).astype(np.float32)
  X, Y, Z = arr_xyz.shape
  vmin, vmax = float(arr_xyz.min()), float(arr_xyz.max())
  arr8 = np.clip((arr_xyz - vmin) * (255.0 / max(1e-6, (vmax - vmin))), 0, 255).astype(np.uint8)
  print(f"[Log] Loaded: shape={(X, Y, Z)} min/max={vmin:.2f}/{vmax:.2f} → uint8")
  return arr8, (X, Y, Z)


def preprocess(volume_u8):
  """Denoise and close small gaps (Gaussian, median, grey closing)."""
  vol = gaussian_filter(volume_u8.astype(np.float32), sigma=GAUSS_SIGMA)
  vol = median_filter(vol, size=DESPECKLE_SIZE)
  vol = grey_closing(vol, footprint=ball(GREY_CLOSE_RAD))
  print(f"[Log] Pre-clean: Gaussian sigma={GAUSS_SIGMA}, Despeckle={DESPECKLE_SIZE}, GreyClose rad={GREY_CLOSE_RAD}")
  return vol


def contrast_rescale_robust(vol_f32):
  """Rescale contrast to 2–98% percentile range (uint8)."""
  lo, hi = np.percentile(vol_f32, (2.0, 98.0))
  vol = (np.clip(vol_f32, lo, hi) - lo) * (255.0 / max(1e-6, (hi - lo)))
  vol_u8 = vol.astype(np.uint8)
  print(f"[Log] Contrast: rescale 2–98% -> {int(vol_u8.min())}/{int(vol_u8.max())}")
  return vol_u8


def threshold_fixed(vol_u8):
  """Threshold to binary mask using THRESH_VAL_U8."""
  mask = (vol_u8 >= float(THRESH_VAL_U8))
  print(f"[Log] Threshold: value={THRESH_VAL_U8} → voxels={int(mask.sum())}")
  return mask


def clean_3d(mask_bool):
  """Fill 3D holes and apply light 3D opening."""
  mask = binary_fill_holes(mask_bool)
  if OPEN_RAD_3D > 0:
    mask = binary_opening(mask, footprint=ball(int(OPEN_RAD_3D)))
  print(f"[Log] Post-bin: FillHoles=3D, Opening rad={OPEN_RAD_3D} → voxels={int(mask.sum())}")
  return mask


def connected_components_keep(mask_bool):
  """Label components and keep those ≥ MIN_SIZE_VOX (3D)."""
  labels3d, nlab = label(mask_bool)
  if nlab == 0:   # if no objects, continue with empty set (avoid hard failure)
    print("[Log] 3D CC: total=0, kept=0")
    return labels3d, set()
  counts = np.bincount(labels3d.ravel()); counts[0] = 0
  keep_ids = np.where(counts >= MIN_SIZE_VOX)[0]
  if keep_ids.size == 0:
    keep_ids = np.where(counts > 0)[0]
  keep_set = set(int(i) for i in keep_ids)
  print(f"[Log] 3D CC: total={nlab}, kept={len(keep_ids)} (≥{MIN_SIZE_VOX} vox)")
  return labels3d, keep_set


def fill_inside_outline_per_slice(labels3d, keep_set, shape_xyz):
  """Fill interiors per slice for kept components and clean borders."""
  X, Y, Z = shape_xyz
  filled = np.zeros((X, Y, Z), dtype=np.uint8)
  se_close = disk(max(1, int(SL_CLOSE_RAD))) if SL_CLOSE_RAD > 0 else None
  se_open  = disk(max(1, int(SL_OPEN_RAD)))  if SL_OPEN_RAD  > 0 else None

  for z in range(Z):
    lab_z = labels3d[:, :, z]
    if lab_z.max() == 0:
      continue
    for lid in np.unique(lab_z):
      if lid == 0 or lid not in keep_set:
        continue
      sl = (lab_z == lid)
      if se_close is not None:
        sl = binary_closing(sl, footprint=se_close)
      sl = remove_small_objects(sl, min_size=int(SL_MIN_OBJECT_2D))
      sl = binary_fill_holes(sl)
      sl = remove_small_holes(sl, area_threshold=int(SL_HOLE_AREA_2D))
      if se_open is not None:
        sl = binary_opening(sl, footprint=se_open)
      filled[:, :, z] |= sl.astype(np.uint8)

  # moved outside the loop to process all slices
  print("[Log] Slice-wise fill-inside-outline complete.")
  return filled


def smooth_3d(mask_u8):
  """Light 3D closing/opening to smooth the final mask."""
  bin_mask = mask_u8.astype(bool)
  if FINAL_CLOSE_RAD > 0:
    bin_mask = binary_closing(bin_mask, footprint=ball(int(FINAL_CLOSE_RAD)))
  if FINAL_OPEN_RAD > 0:
    bin_mask = binary_opening(bin_mask, footprint=ball(int(FINAL_OPEN_RAD)))
  return bin_mask.astype(np.uint8)


# --------------------------------------------------- #
# Main Flow
# --------------------------------------------------- #

def main():
  t0 = time.time()
  # 1) Load and normalize
  arr8, shape_xyz = load_and_normalize(IMAGE_PATH)
  print(f"[Log] Input path: {IMAGE_PATH}")

  # 2) Preprocess
  prep_f32 = preprocess(arr8)

  # 3) Contrast preview and save
  preview_u8 = contrast_rescale_robust(prep_f32)
  print(f"[Log] Preview stats: min={int(preview_u8.min())} max={int(preview_u8.max())}")

  # 4) Threshold to binary
  mask_bool = threshold_fixed(preview_u8)
  print(f"[Log] After threshold: shape={preview_u8.shape} nonzero={int(mask_bool.sum())}")

  # 5) 3D hole fill + opening
  mask_bool = clean_3d(mask_bool)
  print(f"[Log] After clean_3d: nonzero={int(mask_bool.sum())}")

  # 6) 3D connected components
  labels3d, keep_set = connected_components_keep(mask_bool)
  if not keep_set:
    print("[Log] Warning: no components kept; output will be empty")

  # 7) Slice-wise fill inside outline
  filled_u8 = fill_inside_outline_per_slice(labels3d, keep_set, shape_xyz)
  print(f"[Log] After fill_inside: nonzero={int(filled_u8.sum())}")

  # 8) Gentle 3D smoothing
  final_mask = smooth_3d(filled_u8)
  uniq = np.unique(final_mask)
  print(f"[Log] Final mask unique values: {uniq[:10]} (len={len(uniq)}) nonzero={int(final_mask.sum())}")

  # 9) Split into per-lumen 3D masks and save each (0/255)
  labels_final, nlab_final = label(final_mask)
  if nlab_final == 0:
    print("[Log] Warning: no components in final mask; nothing to save")
    print(f"[Log] Completed in {time.time() - t0:.2f}s")
    return
  counts = np.bincount(labels_final.ravel()); counts[0] = 0
  lids = np.nonzero(counts)[0]
  # order by size descending
  lids = lids[np.argsort(counts[lids])[::-1]]
  for idx, lid in enumerate(lids, start=1):
    per_mask = (labels_final == lid).astype(np.uint8)
    out_path = os.path.join(_dir, f"{_base}_lumen{idx:03d}_segmented.tif")
    sitk.WriteImage(sitk.GetImageFromArray(np.transpose(per_mask * 255, (2, 1, 0))), out_path, True)
    print(f"[Log] Saved lumen {idx:03d}: voxels={int(per_mask.sum())} -> {out_path}")
  print(f"[Log] Completed in {time.time() - t0:.2f}s")


if __name__ == "__main__":
  main()
