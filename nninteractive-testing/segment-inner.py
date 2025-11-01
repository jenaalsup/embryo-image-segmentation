#!/usr/bin/env python3
"""
Fully automatic lumen inner-boundary segmentation using nnInteractive.
Hardcoded input file: /Users/jenaalsup/Desktop/CKHRJQ~2.TIF
Outputs: /Users/jenaalsup/Desktop/CKHRJQ~2_mask.tif and PNG slice previews.
"""

import os, warnings, math, random
import numpy as np
import torch
import SimpleITK as sitk
from tifffile import imwrite
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing, binary_fill_holes, label as cc3d
from huggingface_hub import snapshot_download
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

INPUT_PATH = "/Users/jenaalsup/Desktop/CKHRJQ~2.TIF"
OUTPUT_PATH = INPUT_PATH.replace(".TIF", "_mask.tif")

def read_tiff_as_xyz(path):
    itk = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(itk)  # (Z,Y,X)
    return np.transpose(arr, (2,1,0)).astype(np.float32), itk  # (X,Y,Z)

def rescale_for_seeding(vol):
    lo, hi = np.percentile(vol, (2, 98))
    return np.clip((vol - lo) / max(1e-6, (hi - lo)), 0, 1)

def auto_prompts(vol):
    """Find bright ring-like ridges and sample positive and negative points."""
    from scipy.ndimage import binary_opening, binary_closing
    X,Y,Z = vol.shape
    binmask = np.zeros_like(vol, bool)
    for z in range(Z):
        sl = gaussian_filter(vol[...,z], 1)
        thr = max(0.5, np.percentile(sl, 95))
        bw = binary_opening(sl > thr, np.ones((3,3)))
        binmask[...,z] = bw
    binmask = binary_closing(binmask, np.ones((3,3,3)))
    lbl, n = cc3d(binmask)
    rng = np.random.default_rng(13)
    pos, neg = [], []
    for i in range(1,n+1):
        coords = np.argwhere(lbl==i)
        if coords.shape[0] < 250: continue
        for c in coords[rng.choice(coords.shape[0], size=min(12, coords.shape[0]), replace=False)]:
            pos.append(tuple(map(int,c)))
    while len(neg) < 200:
        p = tuple(rng.integers(0, s) for s in (X,Y,Z))
        if not binmask[p]: neg.append(p)
    return pos, neg

def setup_session():
    if torch.__version__.startswith("2.9"):
        warnings.warn("torch 2.9.x may OOM; use torch<=2.8.", RuntimeWarning)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sess = nnInteractiveInferenceSession(device=device, do_autozoom=True, use_pinned_memory=True)
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
    zyx = np.transpose(mask_xyz_uint8, (2,1,0))
    imwrite(out_path, zyx, dtype=np.uint8)
    Z = zyx.shape[0]
    for f in preview_slices:
        z = int(round((Z-1)*f))
        imageio.imwrite(out_path.replace(".tif", f"_z{z:03d}.png"), zyx[z])

def main():
    vol, _ = read_tiff_as_xyz(INPUT_PATH)
    norm = rescale_for_seeding(vol)
    pos, neg = auto_prompts(norm)
    sess = setup_session()
    sess.set_image(vol[None])
    target = torch.zeros(vol.shape, dtype=torch.uint8)
    sess.set_target_buffer(target)
    for p in pos: sess.add_point_interaction(p, True)
    for p in neg: sess.add_point_interaction(p, False)
    mask = sess.target_buffer.clone().cpu().numpy()
    out = postprocess(mask)
    save_outputs(out, OUTPUT_PATH)
    print("✅ Saved:", OUTPUT_PATH)

if __name__ == "__main__":
    main()
