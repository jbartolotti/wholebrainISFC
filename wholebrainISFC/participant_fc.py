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
    
    Caches downsampled files with target resolution in filename. If cached file exists,
    loads it instead of re-downsampling.
    
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
    # Generate cached filename with resolution suffix
    base_path, ext = os.path.splitext(nifti_file)
    if ext == '.gz':
        base_path, ext2 = os.path.splitext(base_path)
        ext = ext2 + ext
    cached_file = f"{base_path}_{int(target_resolution)}mm{ext}"
    
    # Check if cached downsampled file exists
    if os.path.exists(cached_file):
        print(f"    Loading cached downsampled file: {os.path.basename(cached_file)}")
        img = nib.load(cached_file)
        return img.get_fdata()
    
    # Load the original nifti file
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
    print(f"    Downsampling and caching: {os.path.basename(cached_file)}")
    downsampled = zoom(data, scaling_factors, order=1)
    
    # Save the downsampled data for future use
    # Create new affine matrix with updated voxel dimensions
    new_affine = affine.copy()
    for i in range(3):
        new_affine[:3, i] = affine[:3, i] * (1.0 / spatial_scaling[i])
    
    downsampled_img = nib.Nifti1Image(downsampled, new_affine)
    nib.save(downsampled_img, cached_file)
    
    return downsampled


def downsample_mask(mask_file: str, target_resolution: float = 6.0) -> np.ndarray:
    """
    Downsample a mask to match target resolution.
    
    Uses nearest-neighbor interpolation to preserve binary nature of mask.
    Handles both 3D mask files and 4D files (takes mean across time for 4D).
    Caches downsampled masks with target resolution in filename.
    
    Parameters
    ----------
    mask_file : str
        Path to mask file (can be 3D or 4D)
    target_resolution : float
        Target voxel resolution in mm (default 6.0 for 6x6x6 mm³)
    
    Returns
    -------
    np.ndarray
        3D downsampled mask data
    """
    # Generate cached filename with resolution suffix
    base_path, ext = os.path.splitext(mask_file)
    if ext == '.gz':
        base_path, ext2 = os.path.splitext(base_path)
        ext = ext2 + ext
    cached_file = f"{base_path}_{int(target_resolution)}mm{ext}"
    
    # Check if cached downsampled file exists
    if os.path.exists(cached_file):
        print(f"    Loading cached downsampled mask: {os.path.basename(cached_file)}")
        img = nib.load(cached_file)
        return img.get_fdata()
    
    # Load the mask file
    img = nib.load(mask_file)
    data = img.get_fdata()
    affine = img.affine
    
    # Handle 4D data by taking mean across time and thresholding
    if data.ndim == 4:
        print(f"  Note: Mask file is 4D (shape: {data.shape}). Taking mean across time dimension.")
        data = np.mean(data, axis=3)
        # Create binary mask from mean
        data = (data > 0).astype(np.float32)
    elif data.ndim == 3:
        # Already 3D, ensure it's in the right format
        data = data.astype(np.float32)
    else:
        raise ValueError(f"Expected 3D or 4D mask file, got shape: {data.shape}")
    
    # Get current voxel dimensions
    voxel_dims = np.array([np.linalg.norm(affine[:3, i]) for i in range(3)])
    
    # Calculate scaling factors for 3D data
    spatial_scaling = voxel_dims / target_resolution
    
    # Downsample using nearest neighbor (order=0) for masks to preserve binary structure
    print(f"    Downsampling and caching mask: {os.path.basename(cached_file)}")
    downsampled = zoom(data, spatial_scaling, order=0)
    
    # Save the downsampled mask for future use
    # Create new affine matrix with updated voxel dimensions
    new_affine = affine.copy()
    for i in range(3):
        new_affine[:3, i] = affine[:3, i] * (1.0 / spatial_scaling[i])
    
    downsampled_img = nib.Nifti1Image(downsampled, new_affine)
    nib.save(downsampled_img, cached_file)
    
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


def debug_timeseries_nan(timeseries: np.ndarray, condition_name: str = "") -> None:
    """
    Debug function to analyze NaN values in timeseries data.
    
    Prints information about which voxels contain NaN values and how many.
    
    Parameters
    ----------
    timeseries : np.ndarray
        Time series matrix of shape (n_timepoints, n_voxels)
    condition_name : str, optional
        Name of the condition for output labeling
    """
    n_timepoints, n_voxels = timeseries.shape
    
    # Count NaN values per voxel
    nan_counts = np.sum(np.isnan(timeseries), axis=0)
    
    # Find voxels with at least one NaN
    voxels_with_nan = np.where(nan_counts > 0)[0]
    n_affected = len(voxels_with_nan)
    
    print(f"\n  NaN Analysis for {condition_name}:")
    print(f"    Total voxels: {n_voxels}")
    print(f"    Total timepoints: {n_timepoints}")
    print(f"    Voxels with at least one NaN: {n_affected} ({100*n_affected/n_voxels:.2f}%)")
    
    if n_affected == 0:
        print(f"    ✓ No NaN values found - timeseries is clean")
        return
    
    if n_affected <= 20:
        # Few affected voxels - print details for each
        print(f"\n    Details for affected voxels:")
        for voxel_idx in voxels_with_nan:
            n_nan = nan_counts[voxel_idx]
            pct = 100 * n_nan / n_timepoints
            print(f"      Voxel {voxel_idx:5d}: {n_nan:4d} NaN timepoints ({pct:.1f}%)")
    else:
        # Many affected voxels - print summary statistics
        print(f"\n    Summary statistics for NaN counts:")
        print(f"      Min NaN per voxel:  {np.min(nan_counts[nan_counts > 0]):d}")
        print(f"      Max NaN per voxel:  {np.max(nan_counts):.0f}")
        print(f"      Mean NaN per voxel: {np.mean(nan_counts[nan_counts > 0]):.1f}")
        print(f"      Median NaN per voxel: {np.median(nan_counts[nan_counts > 0]):.1f}")
        print(f"      Total NaN values:   {np.sum(nan_counts)}")
        
        # Show distribution
        print(f"\n    Distribution of NaN counts:")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        nan_counts_affected = nan_counts[nan_counts > 0]
        for p in percentiles:
            val = np.percentile(nan_counts_affected, p)
            print(f"      {p}th percentile: {val:.0f} NaN timepoints")


def calculate_fc_matrix(timeseries: np.ndarray, debug: bool = False, 
                        condition_name: str = "") -> np.ndarray:
    """
    Calculate functional connectivity matrix using Pearson correlation.
    
    Cross-correlates all voxels to get an NxN correlation matrix, where N is
    the number of voxels. Applies Fisher's Z transformation to correlation values.
    
    Parameters
    ----------
    timeseries : np.ndarray
        Time series matrix of shape (n_timepoints, n_voxels)
    debug : bool, optional
        If True, print debugging information about NaN values
    condition_name : str, optional
        Name of condition for debug output
    
    Returns
    -------
    np.ndarray
        NxN functional connectivity matrix with Fisher's Z-transformed values
    """
    if debug:
        debug_timeseries_nan(timeseries, condition_name)
    
    # Calculate Pearson correlation matrix
    corr_matrix = pearson_correlation_matrix(timeseries)
    
    # Apply Fisher's Z transformation
    z_matrix = fishers_z_transform(corr_matrix)
    
    return z_matrix


def pearson_correlation_matrix(timeseries: np.ndarray) -> np.ndarray:
    """
    Calculate Pearson correlation matrix between all voxel time series.
    
    Efficiently handles NaN values by computing correlations only on valid voxels
    (those without all-NaN columns), then expanding back to full size.
    
    Parameters
    ----------
    timeseries : np.ndarray
        Time series matrix of shape (n_timepoints, n_voxels)
        May contain all-NaN columns for voxels outside individual mask
    
    Returns
    -------
    np.ndarray
        NxN correlation matrix (contains NaN for invalid voxels)
    """
    n_voxels = timeseries.shape[1]
    
    # Find which voxels are valid (not all NaN)
    valid_voxels = ~np.all(np.isnan(timeseries), axis=0)
    n_valid = np.sum(valid_voxels)
    
    if n_valid == 0:
        # No valid voxels - return all NaN
        return np.full((n_voxels, n_voxels), np.nan)
    
    # Extract only valid voxels
    ts_valid = timeseries[:, valid_voxels]
    
    # Standardize (zero mean, unit variance)
    means = np.mean(ts_valid, axis=0)
    stds = np.std(ts_valid, axis=0)
    ts_standardized = (ts_valid - means) / stds
    
    # Fast matrix multiplication: (1/N) * X^T * X where X is standardized
    n_timepoints = ts_valid.shape[0]
    corr_valid = (ts_standardized.T @ ts_standardized) / n_timepoints
    
    # Create full-size correlation matrix filled with NaN
    corr_matrix = np.full((n_voxels, n_voxels), np.nan)
    
    # Insert valid correlations using fancy indexing (much faster than loops)
    valid_indices = np.where(valid_voxels)[0]
    ix = np.ix_(valid_indices, valid_indices)
    corr_matrix[ix] = corr_valid
    
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


def create_mean_fc_change_map(fc_change: np.ndarray, global_mask: np.ndarray,
                               output_file: str, target_resolution: float = 6.0,
                               mask_affine: np.ndarray | None = None) -> np.ndarray:
    """
    Create a 3D nifti map of mean FC change values for each voxel.
    
    For each voxel, calculates the mean of its FC change values with all other
    voxels (using upper triangle of the FC matrix to avoid double-counting).
    The FC change matrix is already in Fisher's Z space.
    
    Parameters
    ----------
    fc_change : np.ndarray
        NxN FC change matrix (already Fisher Z-transformed)
    global_mask : np.ndarray
        3D global mask indicating which voxels are included
    output_file : str
        Path to save the output nifti file
    target_resolution : float
        Voxel resolution in mm (for constructing affine matrix if none provided)
    mask_affine : np.ndarray, optional
        Affine to use for the output nifti. If None, a simple diagonal affine
        using target_resolution is created. Providing the affine from the
        downsampled global mask ensures spatial alignment with other files.
    
    Returns
    -------
    np.ndarray
        3D array with mean FC change value for each voxel
    """
    # Get the upper triangle indices (excluding diagonal)
    n_voxels = fc_change.shape[0]
    upper_tri_indices = np.triu_indices(n_voxels, k=1)
    
    # Initialize array to hold mean values for each voxel
    mean_fc_change = np.full(n_voxels, np.nan)
    
    # For each voxel, calculate mean of its correlations with other voxels
    # Using upper triangle to avoid double-counting
    for i in range(n_voxels):
        # Get all connections for this voxel from upper triangle
        # Voxel i appears in row i (columns > i) and column i (rows < i)
        row_values = fc_change[i, i+1:]  # Values where i is the row
        col_values = fc_change[:i, i]     # Values where i is the column
        
        # Combine all values for this voxel
        all_values = np.concatenate([row_values, col_values])
        
        # Calculate mean, ignoring NaN values
        if len(all_values) > 0:
            mean_fc_change[i] = np.nanmean(all_values)
    
    # Reshape mean values back to 3D space using global mask
    mean_fc_3d = np.full(global_mask.shape, np.nan)
    mask_indices = global_mask.flatten() > 0
    mean_fc_3d.flat[mask_indices] = mean_fc_change
    
    # Choose affine: prefer provided mask_affine for spatial alignment
    if mask_affine is not None:
        affine = mask_affine
    else:
        affine = np.eye(4)
        affine[0, 0] = target_resolution
        affine[1, 1] = target_resolution
        affine[2, 2] = target_resolution
    
    # Save as nifti file
    img = nib.Nifti1Image(mean_fc_3d, affine)
    nib.save(img, output_file)
    
    print(f"  Mean FC change map saved to: {output_file}")
    print(f"    Shape: {mean_fc_3d.shape}")
    print(f"    Valid voxels: {np.sum(~np.isnan(mean_fc_3d))}")
    print(f"    Mean: {np.nanmean(mean_fc_3d):.6f}")
    print(f"    Std: {np.nanstd(mean_fc_3d):.6f}")
    print(f"    Range: [{np.nanmin(mean_fc_3d):.6f}, {np.nanmax(mean_fc_3d):.6f}]")
    
    return mean_fc_3d


def save_fc_matrices_as_nifti(treatment_fc: np.ndarray, control_fc: np.ndarray,
                               fc_change: np.ndarray, global_mask: np.ndarray,
                               output_file: str, target_resolution: float = 6.0,
                               mask_affine: np.ndarray | None = None) -> None:
    """
    Save all three FC matrices (treatment, control, change) as a 4D NIfTI file.
    
    Creates a 4D NIfTI where each of the three volumes represents one FC matrix:
    - Volume 0: FC change (treatment - control)
    - Volume 1: Treatment FC
    - Volume 2: Control FC
    
    Parameters
    ----------
    treatment_fc : np.ndarray
        NxN treatment FC matrix (already Fisher Z-transformed)
    control_fc : np.ndarray
        NxN control FC matrix (already Fisher Z-transformed)
    fc_change : np.ndarray
        NxN FC change matrix
    global_mask : np.ndarray
        3D global mask indicating which voxels are included
    output_file : str
        Path to save the output 4D NIfTI file
    target_resolution : float
        Voxel resolution in mm (for constructing affine matrix if none provided)
    mask_affine : np.ndarray, optional
        Affine to use for the output nifti. If None, a simple diagonal affine
        using target_resolution is created.
    """
    # Helper function to reshape FC matrix to 3D using global mask
    def reshape_fc_to_3d(fc_matrix: np.ndarray, mask: np.ndarray) -> np.ndarray:
        n_voxels = fc_matrix.shape[0]
        result_3d = np.full(mask.shape, np.nan)
        mask_indices = mask.flatten() > 0
        result_3d.flat[mask_indices] = np.arange(n_voxels)
        
        # Create a 3D array where each voxel contains its index
        voxel_indices = result_3d.copy()
        
        # Now create the output 3D array with mean FC values for visualization
        output_3d = np.full(mask.shape, np.nan)
        
        # For each voxel, store the mean of its correlations (upper triangle)
        valid_mask = voxel_indices >= 0
        for i in range(n_voxels):
            row_values = fc_matrix[i, i+1:]
            col_values = fc_matrix[:i, i]
            all_values = np.concatenate([row_values, col_values])
            if len(all_values) > 0:
                mean_val = np.nanmean(all_values)
                output_3d[voxel_indices == i] = mean_val
        
        return output_3d
    
    # Reshape each FC matrix to 3D
    fc_change_3d = reshape_fc_to_3d(fc_change, global_mask)
    treatment_fc_3d = reshape_fc_to_3d(treatment_fc, global_mask)
    control_fc_3d = reshape_fc_to_3d(control_fc, global_mask)
    
    # Stack into 4D array: (x, y, z, 3)
    data_4d = np.stack([fc_change_3d, treatment_fc_3d, control_fc_3d], axis=3)
    
    # Choose affine
    if mask_affine is not None:
        affine = mask_affine
    else:
        affine = np.eye(4)
        affine[0, 0] = target_resolution
        affine[1, 1] = target_resolution
        affine[2, 2] = target_resolution
    
    # Save as 4D NIfTI
    img = nib.Nifti1Image(data_4d, affine)
    nib.save(img, output_file)
    
    print(f"  4D FC matrices saved to: {output_file}")
    print(f"    Shape: {data_4d.shape}")
    print(f"    Volume 0 (FC Change): mean={np.nanmean(fc_change_3d):.6f}, range=[{np.nanmin(fc_change_3d):.6f}, {np.nanmax(fc_change_3d):.6f}]")
    print(f"    Volume 1 (Treatment FC): mean={np.nanmean(treatment_fc_3d):.6f}, range=[{np.nanmin(treatment_fc_3d):.6f}, {np.nanmax(treatment_fc_3d):.6f}]")
    print(f"    Volume 2 (Control FC): mean={np.nanmean(control_fc_3d):.6f}, range=[{np.nanmin(control_fc_3d):.6f}, {np.nanmax(control_fc_3d):.6f}]")
    
    return None


def process_participant(participant_id: str, data_dir: str,
                        treatment_condition: str, control_condition: str,
                        global_mask: np.ndarray,
                        target_resolution: float = 6.0,
                        apply_censoring: bool = False,
                        debug: bool = False) -> tuple:
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
    debug : bool, optional
        If True, print debugging information about NaN values in timeseries (default: False)
    
    Returns
    -------
    tuple
        (treatment_fc, control_fc, fc_change) - FC matrices for treatment and control conditions,
        and their difference (all may contain NaN for invalid voxels)
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
    treatment_fc = calculate_fc_matrix(treatment_timeseries, debug=debug, 
                                      condition_name=f"{participant_id} {treatment_condition}")
    control_fc = calculate_fc_matrix(control_timeseries, debug=debug,
                                    condition_name=f"{participant_id} {control_condition}")
    
    # Step 6: Calculate FC change
    print(f"  Computing FC change (treatment - control)...")
    fc_change = calculate_fc_change(treatment_fc, control_fc)
    
    print(f"  Completed processing for participant {participant_id}")
    
    return treatment_fc, control_fc, fc_change
