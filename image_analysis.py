#!/usr/bin/env python

# Original author: Ilya Schneider: schnil1010@gmail.com, +34654455293 (WhatsApp), +41762292666 (Phone calls)
# Modifications by: Dr. Roman Vetter, vetterro@ethz.ch

import os

import tifffile

import pandas as pd

import numpy as np
from numpy.linalg import eig, inv

from scipy.optimize import minimize
from scipy.ndimage import binary_fill_holes
from scipy.spatial import cKDTree
        
from skimage import exposure, draw
from skimage.filters import threshold_otsu, gaussian
from skimage.measure import label, regionprops, regionprops_table, marching_cubes
from skimage.morphology import remove_small_objects, binary_opening, binary_closing, binary_erosion, binary_dilation, disk
from skimage.segmentation import clear_border, find_boundaries

import matplotlib.pyplot as plt

import h5py

import pyvista as pv

import trimesh
from trimesh import Trimesh
from trimesh.smoothing import filter_mut_dif_laplacian

import pymeshfix

import napari


## Script options

# The input directory holding the images to process
segmentation_dir = "nn_interactive/output_differentiation/second_batch/d2_out"

# The output file for all the quantifications
csv_file = "lumenoids_data_d2_2.csv"

# The voxel size in µm (check in the properties of the tiff file in ImageJ)
voxel_size_um = np.array([0.568, 0.455, 0.455], dtype=float)

# The number of smoothing iterations to apply to the mesh. Larger numbers will be more smooth, but may distort the geometry.
n_smoothing_iterations = 100 # 750 - 800 was used

# -1 if experiment had no control, 0 if it is the treatment group from the experiment, 1 if it is control in the experiment
control = -1

# day of the experiment on which the image was acquired
imaging_day = 2

# visualization
view_binary_images = False
view_slices = False
view_in_napari = False
view_result = True


## Make the final output df with all the data about vesicles

def list_folders(directory):
    """
    Lists all folders (subdirectories) in the specified directory.
    
    Args:
        directory (str): Path to the directory to scan.
    
    Returns:
        list: Sorted list of folder names (excluding files).
    """
    folders = [
        name for name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, name))
    ]
    folders.sort()
    return folders


## Segmentation

def create_individual_binary_stacks(folder_path, output_folder=None, view_binary_images=False):
    """
    Creates individual binary stacks for each input file where:
    - Non-overlapping object pixels = White (1)
    - Overlapping object pixels = Black (0, background)
    - Background = Black (0)
    - Removes elements from the file containing "0000" in its name
    
    Args:
        folder_path (str): Path to directory containing TIFF files
        output_folder (str, optional): Folder to save binary stacks as TIFFs
        view_binary_images (bool): Whether to display the binary images
    
    Returns:
        list: List of binary stacks (z, y, x) where 1=white (non-overlapping), 0=black (background/overlap)
               One stack per input file (excluding the "0000" file)
    """
    # Get all TIFF files
    tiff_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.tif', '.tiff'))]
    tiff_files.sort()
    
    if not tiff_files:
        raise ValueError("No TIFF files found in the specified directory")

    # Identify and load the base file (containing "0000")
    base_images = None
    base_filename = None
    for filename in tiff_files:
        if "0000" in filename:
            base_filename = filename
            filepath = os.path.join(folder_path, filename)
            with tifffile.TiffFile(filepath) as tif:
                base_images = tif.asarray()
            break
    
    # Load all other files (excluding base)
    processed_stacks = []
    for filename in tiff_files:
        if filename == base_filename:
            continue  # Skip the base file
            
        filepath = os.path.join(folder_path, filename)
        with tifffile.TiffFile(filepath) as tif:
            stack = tif.asarray()
            
        # Initialize binary stack for this file
        binary_stack = np.zeros_like(stack, dtype=np.uint8)
        
        for page_idx in range(len(stack)):
            img = stack[page_idx]
            img_norm = exposure.rescale_intensity(img, out_range=(0, 1))
            binary = img_norm > threshold_otsu(img_norm)
            
            # Subtract base image elements if available
            if base_images is not None and page_idx < len(base_images):
                base_img = base_images[page_idx]
                base_norm = exposure.rescale_intensity(base_img, out_range=(0, 1))
                base_binary = base_norm > threshold_otsu(base_norm)
                binary = np.logical_and(binary, ~base_binary)
            
            binary_stack[page_idx] = binary.astype(np.uint8) * 255  # Scale to 0-255
            
            # Optional plotting
            if view_binary_images:
                plt.figure(figsize=(5, 5))
                plt.imshow(binary_stack[page_idx], cmap='gray')
                plt.title(f'{filename} - Page {page_idx + 1}')
                plt.axis('off')
                plt.show()
        
        processed_stacks.append(binary_stack)
        
        # Save individual stacks if output folder specified
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, f'processed_{filename}')
            tifffile.imwrite(output_path, binary_stack, metadata={'axes': 'ZYX'})
    
    return processed_stacks


def replace_small_objects(original_image, target_label=0, new_label=2, min_object_area=1000, smooth_edges=True, sigma=2.0, no_smoothing_at_all=False):
    """
    Identifies small objects with a specific label and replaces them with a new label, then fixes holes.
    Optionally smooths the edges of the modified regions with more aggressive filtering.

    Args:
        original_image (numpy.ndarray): The original labeled image.
        target_label (int): The label of objects to check (default: 0).
        new_label (int): The label to replace small objects with (default: 2).
        min_object_area (int): The minimum area threshold to retain the object (default: 1000).
        smooth_edges (bool): Whether to apply edge smoothing (default: True).
        sigma (float): Gaussian blur intensity for additional smoothing (default: 2.0).
        no_smoothing_at_all (bool): If True, completely disables all smoothing including morphological operations (default: False).

    Returns:
        numpy.ndarray: The processed image with small objects replaced, holes fixed, and edges smoothed.
    """
    # Create a binary mask for the target label
    binary_mask = (original_image == target_label).astype(np.uint8)
    
    # Label connected components in the binary mask
    labeled_mask = label(binary_mask)
    
    # Get properties of each labeled region
    regions = regionprops(labeled_mask)
    
    # Create a copy of the original image for modification
    processed_image = np.copy(original_image)
    
    # Iterate through regions and replace small objects
    for region in regions:
        if region.area < min_object_area:
            coords = region.coords
            processed_image[coords[:, 0], coords[:, 1]] = new_label
    
    if no_smoothing_at_all:
        # Just return the image with replaced labels, no processing at all
        return processed_image
    
    # Fix holes using binary closing (to remove gaps and smooth the mask)
    fixed_mask = binary_closing(processed_image == new_label, footprint=disk(3)).astype(np.uint8) * new_label
    
    if smooth_edges:
        # Apply binary opening (erosion + dilation) for sharper edges
        smooth_mask = binary_dilation(binary_erosion(fixed_mask, footprint=disk(2)), footprint=disk(2))
        
        # Apply Gaussian filter for further softening
        smooth_mask = gaussian(smooth_mask, sigma=sigma) > 0.5
        
        # Remove the clear_border call to preserve edge-touching objects
        smooth_mask = smooth_mask.astype(np.uint8) * new_label
        
        return smooth_mask
    
    return fixed_mask


# Function to display only the largest object in each region
def extract_largest_objects_masks(mask, min_object_area=1000):
    """
    Extracts the largest object in each region without overlapping backgrounds.

    Args:
        mask (numpy.ndarray): The mask to process.
        min_object_area (int): The minimum area for an object to be considered (default: 1000).

    Returns:
        numpy.ndarray: A mask with only the largest object(s) in each region.
    """
    # Label connected components
    labeled_mask = label(mask)
    
    # Get properties of each labeled region
    regions = regionprops(labeled_mask)
    
    # Create a new mask to store only the largest objects
    largest_objects_mask = np.zeros_like(mask, dtype=np.uint8)
    
    # Iterate through regions and keep only the largest object in each region
    for region in regions:
        # Skip regions where the largest object is smaller than the threshold
        if region.area < min_object_area:
            continue
        
        # Extract the region from the mask
        region_mask = mask[region.bbox[0]:region.bbox[2], region.bbox[1]:region.bbox[3]]
        
        # Label the objects within the region
        region_labeled = label(region_mask)
        
        # Get properties of all objects in the region
        region_props = regionprops(region_labeled)
        
        # Skip regions with no objects
        if not region_props:
            continue
        
        # Find the largest object in the region
        largest_object = max(region_props, key=lambda x: x.area)
        
        # Map the largest object back to the original mask using its exact coordinates
        largest_objects_mask[
            region.bbox[0] + largest_object.coords[:, 0],
            region.bbox[1] + largest_object.coords[:, 1]
        ] = 2  # Assign label 2 to the largest object
    
    return largest_objects_mask


## Creating meshes and extracting data from them

def get_inner_and_outer_surface(
        organoid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Get masks of the inner and inner surfaces of the organoid.

    Parameters
    ----------
    organoid_mask : np.ndarray
        The mask of the organoid. There should be no holes in the tissue.

    Returns
    -------
    np.ndarray
        The binary mask of the inner surface.
    np.ndarray
        The binary mask of the inner surface.
    """

    # find the boundaries of the tissue
    boundary_image = find_boundaries(organoid_mask, mode="outer")

    # give each surface a unique label
    label_image, num_surfaces = label(boundary_image, return_num=True)

    if num_surfaces != 2:
        raise ValueError(
            "There should be exactly two surfaces in the organoid mask."
            " Look for holes in the mask."
        )

    # figure out which surface is the outer and which is the inner.
    # the outer surface is the bigger one.
    rp_table = regionprops_table(label_image, properties=["label", "area"])

    outer_label = rp_table["label"][np.argmax(rp_table["area"])]
    inner_label = rp_table["label"][np.argmin(rp_table["area"])]

    return label_image == outer_label, label_image == inner_label

def mesh_surface(
        surface_mask: np.ndarray,
        voxel_size: np.ndarray,
        n_mesh_smoothing_iterations: int = 0,
        diffusion_coefficient: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Mesh the surface of the organoid.

    Parameters
    ----------
    surface_mask : np.ndarray
        The mask of the surface to mesh.
    voxel_size : np.ndarray
        The size of the voxels in the image. The units are in micrometers.
        They are ordered ZYX.
    n_mesh_smoothing_iterations : int
        The number of smoothing iterations to apply to the mesh.
        Larger numbers will be more smooth, but may distort the geometry
        The default is 0, which means no smoothing.
    diffusion_coefficient : float
        The diffusion coefficient for smoothing. 0 is no diffusion.
        Default value is 0.5.
        https://trimsh.org/trimesh.smoothing.html#trimesh.smoothing.filter_mut_dif_laplacian

    Returns
    -------
    np.ndarray
        The (n, 3) array of vertices of the mesh.
    np.ndarray
        The (n, 3) array of the faces of the mesh.
    """
    # make an initial rough mesh
    vertices, faces, _, _ = marching_cubes(surface_mask, 0)

    vertices_clean, faces_clean = pymeshfix.clean_from_arrays(vertices, faces)

    # scale the vertices to the correct size
    vertices_clean *= voxel_size

    # create the mesh object
    mesh = Trimesh(vertices=vertices_clean, faces=faces_clean)

    # optionally clean up the mesh
    if n_mesh_smoothing_iterations > 0:
        filter_mut_dif_laplacian(
            mesh, iterations=n_mesh_smoothing_iterations, lamb=diffusion_coefficient
        )

    return mesh.vertices, mesh.faces

def robust_mesh_distance(small_mesh, large_mesh):
    """
    More robust distance calculation with multiple fallback methods
    """
    try:
        # First try the efficient proximity query
        prox_query = trimesh.proximity.ProximityQuery(large_mesh)
        distances, _ = prox_query.vertex(small_mesh.vertices)
    except AttributeError:
        # Fallback to point-to-point distance if ProximityQuery fails
        tree = cKDTree(large_mesh.vertices)
        distances, _ = tree.query(small_mesh.vertices)
    
    return {
        'min': np.min(distances),
        'max': np.max(distances),
        'mean': np.mean(distances),
        'median': np.median(distances),
        'all_distances': distances
    }

def improved_geometric_distance(params, points):
    """
    Improved distance metric that combines:
    1. Algebraic distance (faster convergence)
    2. Geometric distance (better accuracy)
    3. Outlier rejection
    """
    center = params[:3]
    radii = params[3:6]
    euler = params[6:]
    rotation = trimesh.transformations.euler_matrix(*euler)[:3, :3]
    
    # Transform points to ellipsoid local coordinates
    local = (points - center).dot(rotation.T)
    scaled = local / radii
    
    # Calculate both algebraic and geometric distances
    algebraic = np.sum(scaled**2, axis=1) - 1  # Faster to compute
    geometric = np.linalg.norm(scaled, axis=1) - 1  # More accurate
    
    # Use a combination of both distances
    # Weight geometric distance more for points close to surface
    weights = np.exp(-np.abs(algebraic))
    combined = weights * geometric + (1 - weights) * algebraic
    
    # Robust fitting by downweighting outliers
    robust_weights = 1 / (1 + np.abs(combined))
    return np.sum(robust_weights * combined**2)


def fit_ellipsoid_optimization(mesh, initial_guess=None, max_iter=500, tol=1e-6):
    points = mesh.vertices
    
    if initial_guess is None:
        # Improved initialization using PCA with scale correction
        center = points.mean(axis=0)
        centered = points - center
        cov = np.cov(centered.T)
        vals, vecs = eig(cov)
        
        # Correct PCA scaling for ellipsoid (PCA tends to overestimate)
        # We use the fact that for uniform distribution in ellipsoid, PCA eigenvalues = radii^2/5
        radii = np.sqrt(5 * vals)
        euler = trimesh.transformations.euler_from_matrix(vecs)
        initial_guess = np.concatenate([center, radii, euler])
    
    # Add bounds to prevent degenerate solutions
    bounds = [
        (None, None), (None, None), (None, None),  # Center (unconstrained)
        (1e-3, None), (1e-3, None), (1e-3, None),  # Radii (positive)
        (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)  # Euler angles
    ]
    
    # First optimization with improved distance metric
    result = minimize(
        improved_geometric_distance, 
        initial_guess, 
        args=(points,),
        method='L-BFGS-B', 
        bounds=bounds,
        options={
            'maxiter': max_iter,
            'ftol': tol,
            'gtol': tol
        }
    )
    
    # If we detect a potentially flat ellipsoid, add a regularization term
    params = result.x
    radii = params[3:6]
    aspect_ratio = np.max(radii) / np.min(radii)
    
    if aspect_ratio > 10:  # Very elongated ellipsoid
        print("Applying flat/elongated mesh regularization")
        
        def regularized_loss(params, points):
            # Original geometric distance
            base_loss = improved_geometric_distance(params, points)
            
            # Regularization to prevent overestimation of small radii
            radii = params[3:6]
            min_radius_penalty = 1e3 * np.exp(-np.min(radii))  # Penalize very small radii
            
            # Encourage aspect ratio to match mesh's
            mesh_extents = mesh.extents
            mesh_aspect = np.max(mesh_extents) / np.min(mesh_extents)
            aspect_penalty = (aspect_ratio - mesh_aspect)**2
            
            return base_loss + min_radius_penalty + aspect_penalty
        
        # Re-optimize with regularization
        result = minimize(
            regularized_loss,
            params,
            args=(points,),
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': max_iter,
                'ftol': tol,
                'gtol': tol
            }
        )
        params = result.x
    
    # Extract final parameters
    center = params[:3]
    radii = params[3:6]
    euler = params[6:]
    rotation = trimesh.transformations.euler_matrix(*euler)[:3, :3]
    
    # Final adjustment - ensure radii match mesh extents in each direction
    # Project all points onto the ellipsoid axes and compare with mesh extent
    local_points = (points - center).dot(rotation.T)
    for i in range(3):
        axis_projection = np.abs(local_points[:, i])
        mesh_extent = mesh.extents[i]
        if mesh_extent > 0:
            scale_factor = np.max(axis_projection) / (radii[i] + 1e-6)
            if 0.5 < scale_factor < 2:  # Only adjust if reasonable
                radii[i] *= scale_factor
    
    return center, radii, rotation

# Create an ellipsoidal mesh with given location, radii, orientation, and color
def create_ellipsoid(center, radii, rotation, color=[255, 0, 255, 100]):
    ellipsoid = trimesh.creation.icosphere(subdivisions=3)  # The number of subdivisions defines the mesh resolution, increase for a finer mesh
    ellipsoid.vertices *= radii  # Scale
    ellipsoid.vertices = ellipsoid.vertices @ rotation.T  # Rotate
    ellipsoid.vertices += center  # Translate
    ellipsoid.visual.face_colors = color
    return ellipsoid


## Run & save everything

output_folders = list_folders(segmentation_dir)
file = output_folders[0]

folder = os.path.join(segmentation_dir, file)

individual_stacks = create_individual_binary_stacks(folder, view_binary_images=view_binary_images)

print('Number of images:', len(individual_stacks))

# Initialize a list to store final processed slices
object_z_stacks = []

for entity in individual_stacks:
    composite = entity
    final_slices = []
        
    black_slide = np.zeros_like(composite[0])

    for slice_idx in range(composite.shape[0]):
        # Get the current slice
        current_slice = composite[slice_idx]
        
        # Remove the singleton dimension (1024, 1024, 1) -> (1024, 1024)
        current_slice = np.squeeze(current_slice)
        
        # Skip if the slice is completely black
        if np.all(current_slice == 0):
            # print(f"Skipping slice {slice_idx} as it's completely black")
            continue

        largest_objects_mask = extract_largest_objects_masks(current_slice, min_object_area=2000)

        # Special treatment for certain slices:
        # if 16 <= slice_idx <= 33 :
        #     noise_removed = replace_small_objects(largest_objects_mask, min_object_area=600, smooth_edges=True, sigma=2, no_smoothing_at_all=True) # sigma=2
        #     processed_mask = replace_small_objects(noise_removed, min_object_area=400, smooth_edges=True, sigma=1, no_smoothing_at_all=True) # sigma=1

        # if slice_idx == 62:
        #     noise_removed = replace_small_objects(largest_objects_mask, min_object_area=5000, smooth_edges=True, sigma=2, no_smoothing_at_all=False) # sigma=2
        #     processed_mask = replace_small_objects(noise_removed, min_object_area=2000, smooth_edges=True, sigma=1, no_smoothing_at_all=False) # sigma=1

        # else:
        noise_removed = replace_small_objects(largest_objects_mask, min_object_area=300, smooth_edges=True, sigma=2, no_smoothing_at_all=True) # sigma=2

        # Smoothen and remove small holes
        processed_mask = replace_small_objects(noise_removed, min_object_area=150, smooth_edges=True, sigma=1, no_smoothing_at_all=True) # sigma=1

        # if slice_idx == 0:
        #     final_image = extract_largest_objects_masks(processed_mask, min_object_area=6000)
        # else:
        final_image = extract_largest_objects_masks(processed_mask, min_object_area=3000)


        # Append the final processed slice to the list
        final_slices.append(final_image)

        # Visualize the results (optional)
        if view_slices:
            plt.figure(figsize=(20, 7))
            plt.subplot(1, 5, 1)
            plt.imshow(current_slice, cmap="gray")
            plt.title(f"Original Image (Slice {slice_idx})")
            plt.axis("off")

            plt.subplot(1, 5, 2)
            plt.imshow(largest_objects_mask, cmap="gray")
            plt.title(f"Largest Objects (Slice {slice_idx})")
            plt.axis("off")

            plt.subplot(1, 5, 3)
            plt.imshow(noise_removed, cmap="gray")
            plt.title(f"Holes filled and smoothed 1 (Slice {slice_idx})")
            plt.axis("off")
            
            plt.subplot(1, 5, 4)
            plt.imshow(processed_mask, cmap="gray")
            plt.title(f"Holes filled and smoothed 2 (Slice {slice_idx})")
            plt.axis("off")

            plt.subplot(1, 5, 5)
            plt.imshow(final_image, cmap="gray")
            plt.title(f"Final Largest Object (Slice {slice_idx})")
            plt.axis("off")
            
            plt.show()

    # Stack the black slide at the beginning of the object_z_stack
    object_z_stack_with_black = np.vstack([
        black_slide[np.newaxis, ...],  # add batch dim (1, H, W)
        final_slices,                 # original (Z, H, W)
        black_slide[np.newaxis, ...]
    ])
    
    object_z_stacks.append(object_z_stack_with_black)

# Format of the output dataframe:
"""
- file: image file of the lumenoid
- control: -1 if experiment had no control, 0 if it is the treatment group from the experiment, 1 if it is control in the experiment
- imaging_day: day of the experiment on which the image was acquired
- object_id: on one image file there might be various lumenoids, so each of them get an id within the image
- v_outer_mesh, v_inner_mesh: volumes of the entire lumenoid and the lumen based on the reconstructed mesh
- area_outer_mesh, area_inner_mesh: area of the entire lumenoid and the lumen based on the reconstructed mesh
- mean_height_mesh, median_height_mesh, min_height_mesh, max_height_mesh: statistics of the distances between the inner and outer mesh (cell height/epithelium thickness) based on shooting rays from nodes of the inner mesh towards the nodes of the outer mesh
- a_outer_ellipsoid, b_outer_ellipsoid, c_outer_ellipsoid: semi-axes of the ellipsoid fitted to the outer mesh
- a_inner_ellipsoid, b_inner_ellipsoid, c_inner_ellipsoid: semi-axes of the ellipsoid fitted to the inner (lumen) mesh
- iou_outer_mesh, iou_inner_mesh: intersection over union (IoU) score between the fitted ellipsoids and meshes
"""
lumenoid_df = pd.DataFrame(columns=['file','control','imaging_day','object_id','v_outer_mesh','v_inner_mesh','area_outer_mesh','area_inner_mesh','mean_height_mesh','median_height_mesh','min_height_mesh','max_height_mesh','a_outer_ellipsoid','b_outer_ellipsoid','c_outer_ellipsoid','a_inner_ellipsoid','b_inner_ellipsoid','c_inner_ellipsoid','iou_outer_mesh','iou_inner_mesh'])

objects_processed = 0

for obj_idx, object_z_stack in enumerate(object_z_stacks):

    lumenoid_df.loc[objects_processed,'file'] = file
    lumenoid_df.loc[objects_processed,'object_id'] = obj_idx

    if control == -1:
        lumenoid_df.loc[objects_processed,'control'] = -1
    elif control:
        lumenoid_df.loc[objects_processed,'control'] = 1
    else:
        lumenoid_df.loc[objects_processed,'control'] = 0

    lumenoid_df.loc[objects_processed,'imaging_day'] = imaging_day

    test_organoid = object_z_stack
    
    outer_surface_mask, inner_surface_mask = get_inner_and_outer_surface(test_organoid)

    outer_vertices, outer_faces = mesh_surface(
        outer_surface_mask,
        voxel_size_um,
        n_mesh_smoothing_iterations=n_smoothing_iterations,
    )
    inner_vertices, inner_faces = mesh_surface(
        inner_surface_mask,
        voxel_size_um,
        n_mesh_smoothing_iterations=n_smoothing_iterations
    )

    outer_mesh = trimesh.Trimesh(vertices=outer_vertices, faces=outer_faces)
    inner_mesh = trimesh.Trimesh(vertices=inner_vertices, faces=inner_faces)

    if not outer_mesh.is_watertight:
        print("Outer mesh is not watertight")

    if not inner_mesh.is_watertight:
        print("Inner mesh is not watertight")

    if outer_mesh.volume < 0:
        outer_mesh.invert()
        # print("Inverting outer mesh")

    lumenoid_df.loc[objects_processed,'v_outer_mesh'] = outer_mesh.volume
    lumenoid_df.loc[objects_processed,'area_outer_mesh'] = outer_mesh.area

    if inner_mesh.volume < 0:
        inner_mesh.invert()
        # print("Inverting inner mesh")

    lumenoid_df.loc[objects_processed,'v_inner_mesh'] = inner_mesh.volume
    lumenoid_df.loc[objects_processed,'area_inner_mesh'] = inner_mesh.area

    mesh_heights = robust_mesh_distance(inner_mesh, outer_mesh)
    # print(f"Mesh heights: {mesh_heights}")

    lumenoid_df.loc[objects_processed,'mean_height_mesh'] = mesh_heights["mean"]
    lumenoid_df.loc[objects_processed,'median_height_mesh'] = mesh_heights["median"]
    lumenoid_df.loc[objects_processed,'min_height_mesh'] = mesh_heights["min"]
    lumenoid_df.loc[objects_processed,'max_height_mesh'] = mesh_heights["max"]

    ## Fit the ellipses
    o_center_opt, o_radii_opt, o_rotation_opt = fit_ellipsoid_optimization(outer_mesh)
    # print("Outer ellipsoid center:", o_center_opt)
    print("Outer ellipsoid radii (a,b,c):", o_radii_opt)

    o_ellipsoid_opt = create_ellipsoid(o_center_opt, o_radii_opt, o_rotation_opt, [0, 0, 255, 100])

    lumenoid_df.loc[objects_processed,'a_outer_ellipsoid'] = o_radii_opt[0]
    lumenoid_df.loc[objects_processed,'b_outer_ellipsoid'] = o_radii_opt[1]
    lumenoid_df.loc[objects_processed,'c_outer_ellipsoid'] = o_radii_opt[2]

    i_center_opt, i_radii_opt, i_rotation_opt = fit_ellipsoid_optimization(inner_mesh)
    # print("Inner ellipsoid center:", i_center_opt)
    print("Inner ellipsoid radii (a,b,c):", i_radii_opt)

    i_ellipsoid_opt = create_ellipsoid(i_center_opt, i_radii_opt, i_rotation_opt, [0, 0, 255, 100])

    lumenoid_df.loc[objects_processed,'a_inner_ellipsoid'] = i_radii_opt[0]
    lumenoid_df.loc[objects_processed,'b_inner_ellipsoid'] = i_radii_opt[1]
    lumenoid_df.loc[objects_processed,'c_inner_ellipsoid'] = i_radii_opt[2]

    # Quantify the quality of fit by the Intersection over Union score
    outer_mesh_pv = pv.wrap(outer_mesh)
    outer_ellipsoid_pv = pv.wrap(o_ellipsoid_opt)
    intersection = outer_mesh_pv.boolean_intersection(outer_ellipsoid_pv)
    union = outer_mesh_pv.boolean_union(outer_ellipsoid_pv)
    vol_intersection = intersection.volume
    vol_union = union.volume
    iou_outer_mesh = vol_intersection / vol_union
    lumenoid_df.loc[objects_processed,'iou_outer_mesh'] = iou_outer_mesh
    print("IoU of outer mesh:", iou_outer_mesh)

    inner_mesh_pv = pv.wrap(inner_mesh)
    inner_ellipsoid_pv = pv.wrap(i_ellipsoid_opt)
    intersection = inner_mesh_pv.boolean_intersection(inner_ellipsoid_pv)
    union = inner_mesh_pv.boolean_union(inner_ellipsoid_pv)
    vol_intersection = intersection.volume
    vol_union = union.volume
    iou_inner_mesh = vol_intersection / vol_union
    lumenoid_df.loc[objects_processed,'iou_inner_mesh'] = iou_inner_mesh
    print("IoU of inner mesh:", iou_inner_mesh)

    objects_processed += 1

    if view_in_napari:
        viewer = napari.Viewer()
        viewer.add_image(
            test_organoid,
            name="organoid mask",
            scale=voxel_size_um
        )
        viewer.add_image(
            outer_surface_mask,
            name="outer surface mask",
            scale=voxel_size_um
        )
        viewer.add_image(
            inner_surface_mask,
            name="inner surface mask",
            scale=voxel_size_um
        )
        viewer.add_surface(
            (outer_vertices, outer_faces),
            name="outer surface",
            opacity=0.5,
        )
        viewer.add_surface(
            (inner_vertices, inner_faces),
            name="inner surface",
            opacity=0.5,
        )
        napari.run()

# write the results to the CSV file
lumenoid_df.to_csv(csv_file)

# optional: view the fitted ellipsoids with PyVista
if view_result:
    # Create a wireframe version of the original mesh for better visualization
    mesh_wire = outer_mesh.copy()
    # mesh_wire.visual = trimesh.visual.ColorVisuals()
    # mesh_wire.visual.face_colors = [0, 255, 0, 0] # Make faces transparent
    mesh_wire.visual.edge_color = [255, 0, 0, 255] # Black edges

    scene = trimesh.Scene([
    mesh_wire,          # Original mesh as wireframe
    o_ellipsoid_opt       # Optimized-fit ellipsoid (blue)
    ])
    scene.show()


    # Create a wireframe version of the original mesh for better visualization
    mesh_wire = inner_mesh.copy()
    # mesh_wire.visual.face_colors = [0, 255, 0, 0] # Make faces transparent
    mesh_wire.visual.edge_color = [0, 0, 0, 255] # Black edges

    scene = trimesh.Scene([
    mesh_wire,          # Original mesh as wireframe
    i_ellipsoid_opt       # Optimized-fit ellipsoid (blue)
    ])
    scene.show()


    # Convert outer surface to PyVista meshes
    pv_mesh = pv.wrap(outer_mesh)
    pv_ellipsoid = pv.wrap(o_ellipsoid_opt)

    # Define slice plane (normal vector + origin point)
    slice_normal = [1, 0, 0]  # Slice along X-axis
    slice_origin = o_center_opt  # Center of ellipsoid (or any point)

    # Slice both meshes
    sliced_mesh = pv_mesh.slice(normal=slice_normal, origin=slice_origin)
    sliced_ellipsoid = pv_ellipsoid.slice(normal=slice_normal, origin=slice_origin)

    # Plot the slices
    p = pv.Plotter()
    p.add_mesh(sliced_mesh, color="green", line_width=3, label="Mesh Slice")
    p.add_mesh(sliced_ellipsoid, color="magenta", line_width=3, label="Ellipsoid Slice")
    p.add_legend()
    p.show()


    # Convert inner surface to PyVista meshes
    pv_mesh = pv.wrap(inner_mesh)
    pv_ellipsoid = pv.wrap(i_ellipsoid_opt)

    # Define slice plane (normal vector + origin point)
    slice_normal = [1, 0, 0]  # Slice along X-axis
    slice_origin = i_center_opt  # Center of ellipsoid (or any point)

    # Slice both meshes
    sliced_mesh = pv_mesh.slice(normal=slice_normal, origin=slice_origin)
    sliced_ellipsoid = pv_ellipsoid.slice(normal=slice_normal, origin=slice_origin)

    # Plot the slices
    p = pv.Plotter()
    p.add_mesh(sliced_mesh, color="green", line_width=3, label="Mesh Slice")
    p.add_mesh(sliced_ellipsoid, color="magenta", line_width=3, label="Ellipsoid Slice")
    p.add_legend()
    p.show()
