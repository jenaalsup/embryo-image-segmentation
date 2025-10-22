#!/usr/bin/env python3
import os
import numpy as np
import torch
import SimpleITK as sitk
from huggingface_hub import snapshot_download
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

# Minimal, hardcoded config
IMAGE_PATH = "/Users/jenaalsup/Desktop/CK7RY1~U.TIF"
OUTPUT_PATH = "nninteractive_mask.tif"
MODEL_REPO = "nnInteractive/nnInteractive"
MODEL_NAME = "nnInteractive_v1.0"
MODEL_DIR = os.path.expanduser("~/.cache/nninteractive")


def main():
    # Force CPU to avoid MPS limitations on 3D ops
    device = torch.device("cpu")
    print(f"[INFO] Device: {device}")

    # Ensure model is available
    model_root = snapshot_download(
        repo_id=MODEL_REPO,
        allow_patterns=[f"{MODEL_NAME}/*"],
        local_dir=MODEL_DIR,
    )
    model_path = os.path.join(model_root, MODEL_NAME)

    # Load image (SimpleITK) -> numpy (1, x, y, z)
    img_itk = sitk.ReadImage(IMAGE_PATH)
    arr_zyx = sitk.GetArrayFromImage(img_itk)  # (z, y, x)
    arr_xyz = np.transpose(arr_zyx, (2, 1, 0))  # (x, y, z)
    img = arr_xyz[None]
    print(f"[INFO] Loaded: {IMAGE_PATH} with shape {img.shape} (1,x,y,z)")

    # Rough auto-detection of lumenoids to place scripted prompts
    # 1) Otsu threshold in SITK to get bright structures
    bin_img = sitk.OtsuThreshold(img_itk, 0, 1)
    # 2) Connected components
    cc = sitk.ConnectedComponent(bin_img)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    labels = [l for l in stats.GetLabels() if stats.GetPhysicalSize(l) > 0]
    if not labels:
        print("[WARN] No components detected by Otsu. Exiting.")
        return

    # Initialize session
    session = nnInteractiveInferenceSession(
        device=device,
        use_torch_compile=False,
        verbose=False,
        torch_n_threads=os.cpu_count(),
        do_autozoom=True,
        use_pinned_memory=bool(device.type == "cuda"),
    )
    session.initialize_from_trained_model_folder(model_path)

    # Prepare output directory and filename prefix
    base = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    out_dir = os.path.join(os.path.dirname(IMAGE_PATH), f"{base}_nnInteractive")
    os.makedirs(out_dir, exist_ok=True)

    # Full-size accumulators (xyz)
    vol_shape = img.shape[1:]
    lumen_union = np.zeros(vol_shape, dtype=np.uint8)

    # Iterate components (each lumenoid)
    obj_idx = 0
    for lbl in labels:
        # Bounding box in index space: (x, y, z, sizeX, sizeY, sizeZ)
        x0, y0, z0, sx, sy, sz = stats.GetBoundingBox(lbl)
        if sx < 5 or sy < 5 or sz < 3:
            continue  # skip tiny components
        obj_idx += 1
        x1, y1, z1 = x0 + sx, y0 + sy, z0 + sz

        # Pad ROI to give the model context
        pad_xy, pad_z = 16, 4
        x0p = max(0, x0 - pad_xy); y0p = max(0, y0 - pad_xy); z0p = max(0, z0 - pad_z)
        x1p = min(vol_shape[0], x1 + pad_xy); y1p = min(vol_shape[1], y1 + pad_xy); z1p = min(vol_shape[2], z1 + pad_z)

        # Extract ROI from ITK image (index=x,y,z, size)
        roi_itk = sitk.RegionOfInterest(img_itk, [int(x1p - x0p), int(y1p - y0p), int(z1p - z0p)], [int(x0p), int(y0p), int(z0p)])
        roi_zyx = sitk.GetArrayFromImage(roi_itk)
        roi_xyz = np.transpose(roi_zyx, (2, 1, 0))  # (sxr, syr, szr)
        if roi_xyz.size == 0:
            continue

        # Local sizes and mid-slice
        sxr, syr, szr = roi_xyz.shape
        z_mid_local = szr // 2

        # Seed: pick brightest voxel in mid-slice (ring-ish) within ROI
        roi_mid = roi_xyz[:, :, z_mid_local]
        peak_idx = np.unravel_index(np.argmax(roi_mid), roi_mid.shape)
        px_l, py_l = int(peak_idx[0]), int(peak_idx[1])

        # Set ROI as current image
        session.set_image(roi_xyz[None])

        # ========== Outer (filled) per-object ==========
        session.reset_interactions()
        session.set_target_buffer(torch.zeros((sxr, syr, szr), dtype=torch.uint8))
        bbox_local = [[0, sxr], [0, syr], [int(z_mid_local), int(z_mid_local + 1)]]
        session.add_bbox_interaction(bbox_local, include_interaction=True)
        session.add_point_interaction((px_l, py_l, int(z_mid_local)), include_interaction=True)
        outer_local = session.target_buffer.detach().cpu().numpy().astype(np.uint8)

        # Place into full-size volume and save per-object file
        outer_full = np.zeros(vol_shape, dtype=np.uint8)
        outer_full[x0p:x1p, y0p:y1p, z0p:z1p] = np.maximum(outer_full[x0p:x1p, y0p:y1p, z0p:z1p], outer_local)
        outer_zyx = np.transpose(outer_full, (2, 1, 0))
        out_path = os.path.join(out_dir, f"{base}_{obj_idx:04d}.tif")
        sitk.WriteImage(sitk.GetImageFromArray(outer_zyx), out_path)
        print(f"[INFO] Saved outer mask: {out_path}")

        # ========== Inner (lumen) per-object, accumulate union ==========
        session.reset_interactions()
        session.set_target_buffer(torch.zeros((sxr, syr, szr), dtype=torch.uint8))
        session.add_bbox_interaction(bbox_local, include_interaction=True)
        # Positive at ROI center, negative at ring
        cx_l, cy_l = sxr // 2, syr // 2
        session.add_point_interaction((cx_l, cy_l, int(z_mid_local)), include_interaction=True)
        session.add_point_interaction((px_l, py_l, int(z_mid_local)), include_interaction=False)
        inner_local = session.target_buffer.detach().cpu().numpy().astype(np.uint8)
        # Accumulate into full-size union
        lumen_union[x0p:x1p, y0p:y1p, z0p:z1p] = np.maximum(lumen_union[x0p:x1p, y0p:y1p, z0p:z1p], inner_local)

    # Save combined inner (base 0000)
    union_zyx = np.transpose(lumen_union, (2, 1, 0))
    base_path = os.path.join(out_dir, f"{base}_0000.tif")
    sitk.WriteImage(sitk.GetImageFromArray(union_zyx), base_path)
    print(f"[INFO] Saved combined inner (base) mask: {base_path}")


if __name__ == "__main__":
    main()
