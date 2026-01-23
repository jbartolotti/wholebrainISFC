"""
Participant-level functional connectivity change calculations.

This module handles processing individual participants' data including:
- Validating input data
- Downsampling nifti files
- Gray matter masking
- Calculating functional connectivity matrices
- Computing FC change between conditions
"""

import os
from typing import Tuple
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom


def validate_participant_data(participant_id: str, data_dir: str, 
                              treatment_condition: str, 
                              control_condition: str) -> Tuple[str, str]:
    """
    Confirm input data for a participant.
    
    Verifies that nifti files exist for both treatment and control conditions,
    already warped to MNI template.
    
    Parameters
    ----------
    participant_id : str
        Participant identifier
    data_dir : str
        Directory containing the brain nifti data
    treatment_condition : str
        Name of the treatment condition
    control_condition : str
        Name of the control condition
    
    Returns
    -------
    tuple
        (treatment_file, control_file) paths to validated nifti files
    
    Raises
    ------
    FileNotFoundError
        If required nifti files are not found
    """
    # Try both .nii and .nii.gz extensions
    treatment_name = f"{participant_id}_{treatment_condition}"
    control_name = f"{participant_id}_{control_condition}"
    
    treatment_file = None
    control_file = None
    
    # Look for treatment file
    for ext in ['.nii.gz', '.nii']:
        candidate = os.path.join(data_dir, treatment_name + ext)
        if os.path.exists(candidate):
            treatment_file = candidate
            break
    
    # Look for control file
    for ext in ['.nii.gz', '.nii']:
        candidate = os.path.join(data_dir, control_name + ext)
        if os.path.exists(candidate):
            control_file = candidate
            break
    
    if treatment_file is None:
        raise FileNotFoundError(f"Treatment condition file not found for participant {participant_id}: "
                                f"expected {os.path.join(data_dir, treatment_name + '.[nii|nii.gz]')}")
    
    if control_file is None:
        raise FileNotFoundError(f"Control condition file not found for participant {participant_id}: "
                                f"expected {os.path.join(data_dir, control_name + '.[nii|nii.gz]')}")
    
    return treatment_file, control_file


def downsample_nifti(nifti_file: str, target_resolution: float = 6.0) -> np.ndarray:
    """
    Downsample nifti file to target resolution (e.g., 6x6x6 mm³ from 3mm³).
    
    Parameters
    ----------
    nifti_file : str
        Path to input nifti file
    target_resolution : float
        Target voxel resolution in mm (default 6.0 for 6x6x6 mm³)
    
    Returns
    -------
    np.ndarray
        Downsampled image data
    """
    # Load the nifti file
    img = nib.load(nifti_file)
    data = img.get_fdata()
    affine = img.affine
    
    # Get current voxel dimensions (spatial dimensions only)
    voxel_dims = np.array([np.linalg.norm(affine[:3, i]) for i in range(3)])
    
    # Calculate scaling factors for spatial dimensions
    spatial_scaling = voxel_dims / target_resolution
    
    # Create scaling factors for all dimensions
    # For 4D data (x, y, z, time), we only downsample spatial dimensions, not time
    if data.ndim == 4:
        scaling_factors = list(spatial_scaling) + [1.0]  # Don't scale time dimension
    else:
        scaling_factors = spatial_scaling
    
    # Downsample using scipy zoom with linear interpolation for continuous data
    downsampled = zoom(data, scaling_factors, order=1)
    
    return downsampled


def downsample_mask(mask_file: str, target_resolution: float = 6.0) -> np.ndarray:
    """
    Downsample a 3D mask to match target resolution.
    
    Uses nearest-neighbor interpolation to preserve binary nature of mask.
    
    Parameters
    ----------
    mask_file : str
        Path to mask file
    target_resolution : float
        Target voxel resolution in mm (default 6.0 for 6x6x6 mm³)
    
    Returns
    -------
    np.ndarray
        Downsampled mask data
    """
    # Load the mask file
    img = nib.load(mask_file)
    data = img.get_fdata()
    affine = img.affine
    
    # Get current voxel dimensions
    voxel_dims = np.array([np.linalg.norm(affine[:3, i]) for i in range(3)])
    
    # Calculate scaling factors
    spatial_scaling = voxel_dims / target_resolution
    
    # Downsample using nearest neighbor (order=0) for masks to preserve binary structure
    downsampled = zoom(data, spatial_scaling, order=0)
    
    return downsampled


def apply_mask_and_extract_timeseries(nifti_file: str, mask_file: str) -> np.ndarray:
    """
    Apply grey matter mask and extract time series from all voxels in the mask.
    
    Masking is done at the original resolution to avoid edge artifacts from
    resampling binary masks.
    
    Parameters
    ----------
    nifti_file : str
        Path to input 4D nifti file
    mask_file : str
        Path to grey matter mask file
    
    Returns
    -------
    np.ndarray
        Time series matrix of shape (n_timepoints, n_voxels_in_mask)
    """
    # Load the nifti file
    img = nib.load(nifti_file)
    data = img.get_fdata()
    
    # Load the mask
    mask_img = nib.load(mask_file)
    mask = mask_img.get_fdata()
    
    # Ensure data is 4D
    if data.ndim != 4:
        raise ValueError(f"Expected 4D nifti data with time dimension, got shape {data.shape}")
    
    n_timepoints = data.shape[3]
    
    # Flatten spatial dimensions
    data_reshaped = data.reshape(-1, n_timepoints)
    mask_flat = mask.flatten()
    
    # Extract voxels where mask > 0
    masked_voxels = data_reshaped[mask_flat > 0, :]
    
    # Return as (n_timepoints, n_voxels)
    return masked_voxels.T


def create_global_mask(participant_ids: list, conditions: list, data_dir: str,
                       target_resolution: float = 6.0) -> np.ndarray:
    """
    Create a global mask as the union (logical OR) of all individual masks.
    
    Parameters
    ----------
    participant_ids : list
        List of participant identifiers
    conditions : list
        List of condition names
    data_dir : str
        Directory containing mask files
    target_resolution : float
        Target resolution for downsampling masks
    
    Returns
    -------
    np.ndarray
        Global mask (3D array) with value 1 where any individual mask has data
    """
    global_mask = None
    
    for participant_id in participant_ids:
        for condition in conditions:
            try:
                mask_file = find_gm_mask_file(participant_id, condition, data_dir)
                downsampled_mask = downsample_mask(mask_file, target_resolution)
                
                if global_mask is None:
                    # Initialize with first mask
                    global_mask = (downsampled_mask > 0).astype(np.float32)
                else:
                    # Logical OR - union of all masks
                    global_mask = np.logical_or(global_mask, downsampled_mask > 0).astype(np.float32)
            except FileNotFoundError:
                print(f"  Warning: Mask not found for {participant_id}, {condition}. Skipping.")
                continue
    
    if global_mask is None:
        raise ValueError("No valid masks found to create global mask")
    
    return global_mask


def apply_global_and_individual_mask(downsampled_data: np.ndarray,
                                     global_mask: np.ndarray,
                                     individual_mask: np.ndarray) -> np.ndarray:
    """
    Extract time series using global mask, filling invalid voxels with NaN.
    
    Extracts time series for all voxels in the global mask. For voxels that are
    in the global mask but NOT in the individual mask, fills with np.nan.
    
    Parameters
    ----------
    downsampled_data : np.ndarray
        Downsampled 4D functional data (x, y, z, time)
    global_mask : np.ndarray
        3D global mask covering all participants/conditions
    individual_mask : np.ndarray
        3D individual mask for this specific participant/condition
    
    Returns
    -------
    np.ndarray
        Time series matrix of shape (n_timepoints, n_global_voxels)
        Invalid voxels filled with np.nan
    """
    # Ensure data is 4D
    if downsampled_data.ndim != 4:
        raise ValueError(f"Expected 4D nifti data with time dimension, got shape {downsampled_data.shape}")
    
    n_timepoints = downsampled_data.shape[3]
    
    # Flatten spatial dimensions
    data_reshaped = downsampled_data.reshape(-1, n_timepoints)
    global_mask_flat = global_mask.flatten()
    individual_mask_flat = individual_mask.flatten()
    
    # Initialize output with NaN
    n_global_voxels = int(np.sum(global_mask_flat > 0))
    timeseries = np.full((n_timepoints, n_global_voxels), np.nan, dtype=np.float64)
    
    # Get indices for global mask voxels
    global_indices = np.where(global_mask_flat > 0)[0]
    
    # For each global voxel, check if it's in individual mask
    for i, global_idx in enumerate(global_indices):
        if individual_mask_flat[global_idx] > 0:
            # Valid voxel - extract actual time series
            timeseries[:, i] = data_reshaped[global_idx, :]
        # else: leave as NaN (already initialized)
    
    return timeseries


def find_gm_mask_file(participant_id: str, condition: str, data_dir: str) -> str:
    """
    Find the participant-specific grey matter mask file.
    
    Looks for a file named {participant_id}_{condition}_GM.nii[.gz] in data_dir.
    
    Parameters
    ----------
    participant_id : str
        Participant identifier
    condition : str
        Condition name
    data_dir : str
        Directory containing the mask file
    
    Returns
    -------
    str
        Path to the grey matter mask file
    
    Raises
    ------
    FileNotFoundError
        If the mask file is not found
    """
    gm_mask_name = f"{participant_id}_{condition}_GM"
    
    # Try both .nii.gz and .nii extensions
    for ext in ['.nii.gz', '.nii']:
        candidate = os.path.join(data_dir, gm_mask_name + ext)
        if os.path.exists(candidate):
            return candidate
    
    raise FileNotFoundError(f"Grey matter mask not found for participant {participant_id}, "
                           f"condition {condition}: expected {os.path.join(data_dir, gm_mask_name + '.[nii|nii.gz]')}")


def load_censoring_vector(participant_id: str, condition: str, data_dir: str, n_timepoints: int) -> np.ndarray:
    """
    Load censoring vector for a participant and condition.
    
    Looks for a file named {participant_id}_{condition}_censor.[1D|tsv|csv|txt] in data_dir.
    
    Parameters
    ----------
    participant_id : str
        Participant identifier
    condition : str
        Condition name
    data_dir : str
        Directory containing the censoring file
    n_timepoints : int
        Expected number of timepoints (for validation)
    
    Returns
    -------
    np.ndarray
        Boolean array of shape (n_timepoints,) where True = keep, False = censor
    
    Raises
    ------
    FileNotFoundError
        If the censoring file is not found
    ValueError
        If the censoring vector length doesn't match n_timepoints
    """
    censor_base = f"{participant_id}_{condition}_censor"
    censor_file = None
    
    # Try various file extensions
    for ext in ['.1D', '.tsv', '.csv', '.txt']:
        candidate = os.path.join(data_dir, censor_base + ext)
        if os.path.exists(candidate):
            censor_file = candidate
            break
    
    if censor_file is None:
        raise FileNotFoundError(f"Censoring file not found for participant {participant_id}, "
                               f"condition {condition}: expected {os.path.join(data_dir, censor_base + '.[1D|tsv|csv|txt]')}")
    
    # Load the censoring vector
    censor_vector = np.loadtxt(censor_file)
    
    # Ensure it's 1D
    if censor_vector.ndim != 1:
        raise ValueError(f"Censoring vector must be 1D, got shape {censor_vector.shape}")
    
    # Validate length
    if len(censor_vector) != n_timepoints:
        raise ValueError(f"Censoring vector length {len(censor_vector)} does not match "
                        f"number of timepoints {n_timepoints}")
    
    # Convert to boolean: 1 = keep (True), 0 = censor (False)
    censor_mask = censor_vector.astype(bool)
    
    return censor_mask


def apply_censoring_to_timeseries(timeseries: np.ndarray, participant_id: str,
                                   condition: str, data_dir: str) -> np.ndarray:
    """
    Apply censoring to time series data.
    
    Parameters
    ----------
    timeseries : np.ndarray
        Time series matrix of shape (n_timepoints, n_voxels)
    participant_id : str
        Participant identifier
    condition : str
        Condition name
    data_dir : str
        Directory containing censoring file
    
    Returns
    -------
    np.ndarray
        Censored time series of shape (n_timepoints_kept, n_voxels)
    """
    n_timepoints = timeseries.shape[0]
    censor_mask = load_censoring_vector(participant_id, condition, data_dir, n_timepoints)
    # Keep only non-censored timepoints
    return timeseries[censor_mask, :]


def calculate_fc_matrix(timeseries: np.ndarray) -> np.ndarray:
    """
    Calculate functional connectivity matrix using Pearson correlation.
    
    Cross-correlates all voxels to get an NxN correlation matrix, where N is
    the number of voxels. Applies Fisher's Z transformation to correlation values.
    
    Parameters
    ----------
    timeseries : np.ndarray
        Time series matrix of shape (n_timepoints, n_voxels)
    
    Returns
    -------
    np.ndarray
        NxN functional connectivity matrix with Fisher's Z-transformed values
    """
    # Calculate Pearson correlation matrix
    corr_matrix = pearson_correlation_matrix(timeseries)
    
    # Apply Fisher's Z transformation
    z_matrix = fishers_z_transform(corr_matrix)
    
    return z_matrix


def pearson_correlation_matrix(timeseries: np.ndarray) -> np.ndarray:
    """
    Calculate Pearson correlation matrix between all voxel time series.
    
    Handles NaN values by computing correlations only on valid (non-NaN) timepoints.
    Voxels with all NaN will have NaN correlations with all other voxels.
    
    Parameters
    ----------
    timeseries : np.ndarray
        Time series matrix of shape (n_timepoints, n_voxels)
    
    Returns
    -------
    np.ndarray
        NxN correlation matrix (may contain NaN for invalid voxels)
    """
    n_voxels = timeseries.shape[1]
    corr_matrix = np.zeros((n_voxels, n_voxels))
    
    # Use numpy's corrcoef which handles NaN via masking
    for i in range(n_voxels):
        for j in range(i, n_voxels):
            # Find valid (non-NaN) timepoints for both voxels
            valid_mask = ~(np.isnan(timeseries[:, i]) | np.isnan(timeseries[:, j]))
            
            if np.sum(valid_mask) > 1:  # Need at least 2 points for correlation
                valid_i = timeseries[valid_mask, i]
                valid_j = timeseries[valid_mask, j]
                
                # Calculate correlation
                if np.std(valid_i) > 0 and np.std(valid_j) > 0:
                    corr = np.corrcoef(valid_i, valid_j)[0, 1]
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr
                else:
                    # Zero variance - set to NaN
                    corr_matrix[i, j] = np.nan
                    corr_matrix[j, i] = np.nan
            else:
                # Not enough valid data - set to NaN
                corr_matrix[i, j] = np.nan
                corr_matrix[j, i] = np.nan
    
    return corr_matrix


def fishers_z_transform(correlation_matrix: np.ndarray) -> np.ndarray:
    """
    Apply Fisher's Z transformation to correlation values.
    
    Transforms r values to z-values using: z = 0.5 * ln((1+r)/(1-r))
    
    Parameters
    ----------
    correlation_matrix : np.ndarray
        Matrix of correlation values
    
    Returns
    -------
    np.ndarray
        Matrix with Fisher's Z-transformed values
    """
    # Clip values to prevent log of zero or negative numbers
    # Correlations should be in [-1, 1], clip to slightly less extreme values
    r_clipped = np.clip(correlation_matrix, -0.9999, 0.9999)
    
    # Apply Fisher's Z transformation: z = 0.5 * ln((1+r)/(1-r))
    z_transformed = 0.5 * np.log((1 + r_clipped) / (1 - r_clipped))
    
    return z_transformed


def calculate_fc_change(treatment_fc: np.ndarray, control_fc: np.ndarray) -> np.ndarray:
    """
    Calculate FC change by subtracting control condition from treatment condition.
    
    Parameters
    ----------
    treatment_fc : np.ndarray
        NxN FC matrix for treatment condition
    control_fc : np.ndarray
        NxN FC matrix for control condition
    
    Returns
    -------
    np.ndarray
        NxN difference matrix representing FC change
    """
    # Calculate difference matrix
    fc_change = treatment_fc - control_fc
    
    return fc_change


def process_participant(participant_id: str, data_dir: str,
                        treatment_condition: str, control_condition: str,
                        global_mask: np.ndarray,
                        target_resolution: float = 6.0,
                        apply_censoring: bool = False) -> np.ndarray:
    """
    Complete processing pipeline for a single participant.
    
    Orchestrates all steps: validation, downsampling, masking, 
    FC calculation, and difference matrix computation.
    
    Uses a global mask to ensure all FC matrices have the same dimensions.
    Voxels outside individual masks are filled with NaN.
    
    Parameters
    ----------
    participant_id : str
        Participant identifier
    data_dir : str
        Directory containing brain nifti data and mask files
    treatment_condition : str
        Name of treatment condition
    control_condition : str
        Name of control condition
    global_mask : np.ndarray
        Global mask (union of all individual masks) at target resolution
    target_resolution : float
        Target voxel resolution in mm (default 6.0)
    apply_censoring : bool, optional
        If True, apply censoring to time series before FC calculation (default: False)
    
    Returns
    -------
    np.ndarray
        FC change matrix for the participant (may contain NaN for invalid voxels)
    """
    # Step 1: Validate input data
    print(f"Processing participant {participant_id}...")
    print(f"  Validating input data...")
    treatment_file, control_file = validate_participant_data(
        participant_id, data_dir, treatment_condition, control_condition
    )
    
    # Find mask files
    treatment_mask_file = find_gm_mask_file(participant_id, treatment_condition, data_dir)
    control_mask_file = find_gm_mask_file(participant_id, control_condition, data_dir)
    
    # Step 2: Downsample nifti files and masks
    print(f"  Downsampling to {target_resolution}x{target_resolution}x{target_resolution} mm³...")
    treatment_downsampled = downsample_nifti(treatment_file, target_resolution)
    control_downsampled = downsample_nifti(control_file, target_resolution)
    treatment_mask_downsampled = downsample_mask(treatment_mask_file, target_resolution)
    control_mask_downsampled = downsample_mask(control_mask_file, target_resolution)
    
    # Step 3: Apply global and individual masks, extract time series (with NaN for invalid voxels)
    print(f"  Applying masks and extracting time series...")
    treatment_timeseries = apply_global_and_individual_mask(
        treatment_downsampled, global_mask, treatment_mask_downsampled
    )
    control_timeseries = apply_global_and_individual_mask(
        control_downsampled, global_mask, control_mask_downsampled
    )
    
    # Step 4: Apply censoring if requested
    if apply_censoring:
        print(f"  Applying censoring...")
        treatment_timeseries = apply_censoring_to_timeseries(
            treatment_timeseries, participant_id, treatment_condition, data_dir
        )
        control_timeseries = apply_censoring_to_timeseries(
            control_timeseries, participant_id, control_condition, data_dir
        )
    
    # Step 5: Calculate FC matrices
    print(f"  Calculating functional connectivity matrices...")
    treatment_fc = calculate_fc_matrix(treatment_timeseries)
    control_fc = calculate_fc_matrix(control_timeseries)
    
    # Step 6: Calculate FC change
    print(f"  Computing FC change (treatment - control)...")
    fc_change = calculate_fc_change(treatment_fc, control_fc)
    
    print(f"  Completed processing for participant {participant_id}")
    
    return fc_change
