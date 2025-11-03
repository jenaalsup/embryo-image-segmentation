#!/usr/bin/env python3
"""
Fully automatic lumen inner-boundary segmentation using nnInteractive.
Hardcoded input file: /Users/jenaalsup/Desktop/CKHRJQ~2.TIF
Outputs: /Users/jenaalsup/Desktop/CKHRJQ~2_mask.tif and PNG slice previews.
"""

import os, warnings, math, random
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import SimpleITK as sitk
from tifffile import imwrite
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing, binary_fill_holes, label as cc3d
from huggingface_hub import snapshot_download
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
from pathlib import Path
import time

INPUT_PATH = Path(r"C:\Users\Zernicka-Goetz Lab\Desktop\Jena\CKHRJQ_2.TIF")
OUTPUT_PATH = INPUT_PATH.with_name(INPUT_PATH.stem + "_mask.tif")

def read_tiff_as_xyz(path):
    itk = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(itk)  # (Z,Y,X)
    return np.transpose(arr, (2,1,0)).astype(np.float32), itk  # (X,Y,Z)

def rescale_for_seeding(vol):
    lo, hi = np.percentile(vol, (2, 98))
    return np.clip((vol - lo) / max(1e-6, (hi - lo)), 0, 1)

from scipy.ndimage import distance_transform_edt as edt

def auto_prompts(vol):
    """Automatically pick central positive points for lumen and boundary negatives."""
    X, Y, Z = vol.shape
    binmask = np.zeros_like(vol, bool)

    # Quick thresholding and smoothing to detect lumen candidates
    for z in range(Z):
        sl = gaussian_filter(vol[..., z], 1)
        thr = max(0.5, np.percentile(sl, 95))
        bw = binary_opening(sl > thr, np.ones((3, 3)))
        binmask[..., z] = bw
    binmask = binary_closing(binmask, np.ones((3, 3, 3)))

    lbl, n = cc3d(binmask)
    pos, neg = [], []
    rng = np.random.default_rng(13)

    for i in range(1, n + 1):
        comp = lbl == i
        if comp.sum() < 250:  # skip tiny objects
            continue

        # Distance transform to find the center
        dt = edt(comp)
        # Pick 1-3 highest distance voxels as positive points (central lumen)
        center_coords = np.argwhere(dt)
        distances = dt[tuple(center_coords.T)]
        top_idx = distances.argsort()[-min(3, len(distances)):]
        for idx in top_idx:
            pos.append(tuple(map(int, center_coords[idx])))

        # Negative points: randomly around the boundary
        boundary = comp & (dt < 2)  # voxels near the edge of object
        bcoords = np.argwhere(boundary)
        for c in bcoords[rng.choice(len(bcoords), size=min(5, len(bcoords)), replace=False)]:
            neg.append(tuple(map(int, c)))

    # Safety: add a few random negatives if none found
    while len(neg) < 5:
        p = tuple(rng.integers(0, s) for s in (X, Y, Z))
        if not binmask[p]:
            neg.append(p)

    return pos, neg


def setup_session():
    if torch.__version__.startswith("2.9"):
        warnings.warn("torch 2.9.x may OOM; use torch<=2.8.", RuntimeWarning)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sess = nnInteractiveInferenceSession(device=device, do_autozoom=False, use_pinned_memory=True)
    path = snapshot_download("nnInteractive/nnInteractive", allow_patterns=["nnInteractive_v1.0/*"],
                             local_dir=os.path.expanduser("~/.cache/nninteractive"))
    sess.initialize_from_trained_model_folder(os.path.join(path,"nnInteractive_v1.0"))
    return sess

def postprocess(mask):
    from scipy.ndimage import binary_closing, binary_fill_holes, label
    m = binary_closing(mask>0, np.ones((3,3,3)))
    m = binary_fill_holes(m)
    lbl, n = label(m)
    keep = np.zeros_like(m)
    for i in range(1,n+1):
        comp = lbl==i
        if comp.sum()>=2000: keep |= comp
    return (keep*255).astype(np.uint8)

def save_outputs(mask_xyz_uint8, out_path, preview_slices=(0.25,0.5,0.75)):
    import imageio.v2 as imageio
    # Convert mask from (X,Y,Z) to (Z,Y,X) for saving
    zyx = np.transpose(mask_xyz_uint8, (2,1,0))
    # Save full 3D mask as TIFF
    imwrite(out_path, zyx, dtype=np.uint8)
    
    Z = zyx.shape[0]
    for f in preview_slices:
        z = int(round((Z-1) * f))
        # Construct preview PNG path using Path methods
        preview_path = out_path.with_name(out_path.stem + f"_z{z:03d}.png")
        imageio.imwrite(preview_path, zyx[z])

def main():
    start_total = time.time()

    t0 = time.time()
    vol, _ = read_tiff_as_xyz(INPUT_PATH)
    print(f"[{time.time()-t0:.2f}s] Loaded image")

    t0 = time.time()
    norm = rescale_for_seeding(vol)
    print(f"[{time.time()-t0:.2f}s] Rescaled image for seeding")

    t0 = time.time()
    pos, neg = auto_prompts(norm)
    print(f"[{time.time()-t0:.2f}s] Generated auto prompts")
    print(f"  Positive points: {len(pos)}, Negative points: {len(neg)}")

    t0 = time.time()
    sess = setup_session()
    print(f"[{time.time()-t0:.2f}s] Session setup complete")

    t0 = time.time()
    sess.set_image(vol[None])
    target = torch.zeros(vol.shape, dtype=torch.uint8)
    sess.set_target_buffer(target)
    print(f"[{time.time()-t0:.2f}s] Image loaded into session")

    # --- Add logging for point interactions ---
    print("Starting point interactions...")
    total_points = len(pos) + len(neg)
    start_points = time.time()
    for i, p in enumerate(pos):
        sess.add_point_interaction(p, True)
        if (i+1) % 10 == 0 or i == len(pos)-1:
            elapsed = time.time() - start_points
            avg_time = elapsed / (i+1)
            remaining = avg_time * (total_points - (i+1))
            print(f"[Pos {i+1}/{len(pos)}] Avg {avg_time:.2f}s per point, est {remaining/60:.1f} min remaining")
    for i, p in enumerate(neg):
        sess.add_point_interaction(p, False)
        idx = i + len(pos)
        if (i+1) % 10 == 0 or i == len(neg)-1:
            elapsed = time.time() - start_points
            avg_time = elapsed / (idx+1)
            remaining = avg_time * (total_points - (idx+1))
            print(f"[Neg {i+1}/{len(neg)}] Avg {avg_time:.2f}s per point, est {remaining/60:.1f} min remaining")

    mask = sess.target_buffer.clone().cpu().numpy()
    print(f"[{time.time()-start_points:.2f}s] Inference complete")

    t0 = time.time()
    out = postprocess(mask)
    print(f"[{time.time()-t0:.2f}s] Postprocessing done")

    t0 = time.time()
    save_outputs(out, OUTPUT_PATH)
    print(f"[{time.time()-t0:.2f}s] Outputs saved")

    print(f"✅ Total time: {time.time()-start_total:.2f}s")

if __name__ == "__main__":
    main()
