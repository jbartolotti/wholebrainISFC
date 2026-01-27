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
from typing import Optional
import numpy as np
import nibabel as nib

from .data_prep import (
    validate_participant_data,
    downsample_nifti,
    downsample_mask,
    _cached_downsample_path,
    prepare_masks,
    find_gm_mask_file,
    apply_censoring_to_timeseries,
)


def apply_global_and_individual_mask(data_4d: np.ndarray,
                                     global_mask: np.ndarray,
                                     individual_mask: np.ndarray,
                                     debug: bool = False) -> np.ndarray:
    """Extract time series using a global mask with individual-mask holes marked NaN.

    Returns an array shaped (n_timepoints, n_global_voxels). Voxels that are
    in the global mask but excluded by the individual mask are filled with NaN
    so downstream FC calculations retain alignment with the global mask layout.
    """

    if data_4d.ndim != 4:
        raise ValueError(f"Expected 4D data (x, y, z, t); got shape {data_4d.shape}")

    if data_4d.shape[:3] != global_mask.shape or data_4d.shape[:3] != individual_mask.shape:
        raise ValueError(
            "Data and mask shapes must match: "
            f"data {data_4d.shape[:3]}, global {global_mask.shape}, individual {individual_mask.shape}"
        )

    n_timepoints = data_4d.shape[3]
    global_flat = global_mask.flatten() > 0
    individual_flat = individual_mask.flatten() > 0
    global_indices = np.where(global_flat)[0]

    if global_indices.size == 0:
        raise ValueError("Global mask contains zero voxels after thresholding")

    data_flat = data_4d.reshape(-1, n_timepoints)
    data_in_global = data_flat[global_indices]  # shape: (n_voxels, n_timepoints)

    timeseries = data_in_global.T  # (n_timepoints, n_voxels)

    # Fill voxels excluded by individual mask with NaN but keep column alignment
    excluded = ~individual_flat[global_indices]
    if np.any(excluded):
        timeseries[:, excluded] = np.nan

    if debug:
        n_global = global_indices.size
        n_included = n_global - int(np.sum(excluded))
        print(f"    Masking: global voxels {n_global}, individual-included {n_included}, excluded {int(np.sum(excluded))}")

    return timeseries


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


def save_zero_variance_mask(timeseries: np.ndarray, global_mask: np.ndarray,
                            output_file: str, mask_affine: np.ndarray | None,
                            target_resolution: float, condition_name: str = "") -> None:
    """Create and save a mask of zero-variance voxels aligned to the global mask."""
    n_timepoints, n_voxels = timeseries.shape
    stds = np.nanstd(timeseries, axis=0)
    zero_var = (stds == 0) | np.isnan(stds)
    n_zero = int(np.sum(zero_var))
    n_valid = n_voxels - n_zero

    print(f"    {condition_name}: valid voxels {n_valid}, zero-variance voxels {n_zero} ({100*n_zero/n_voxels:.2f}%)")

    # Map to 3D using global mask
    zero_mask_3d = np.zeros(global_mask.shape, dtype=np.uint8)
    mask_indices = global_mask.flatten() > 0
    zero_mask_3d.flat[mask_indices] = zero_var.astype(np.uint8)

    # Choose affine
    if mask_affine is not None:
        affine = mask_affine
    else:
        affine = np.eye(4)
        affine[0, 0] = target_resolution
        affine[1, 1] = target_resolution
        affine[2, 2] = target_resolution

    nib.save(nib.Nifti1Image(zero_mask_3d, affine), output_file)
    print(f"    Zero-variance mask saved to: {output_file}")


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
    corr_matrix = pearson_correlation_matrix(timeseries, debug=debug)
    
    # Apply Fisher's Z transformation
    z_matrix = fishers_z_transform(corr_matrix)
    
    return z_matrix


def pearson_correlation_matrix(timeseries: np.ndarray, debug: bool = False) -> np.ndarray:
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
    print(f"    Number of valid voxels: {n_valid} out of {n_voxels} ({100*n_valid/n_voxels:.2f}%)")
    
    if n_valid == 0:
        # No valid voxels - return all NaN
        return np.full((n_voxels, n_voxels), np.nan)
    
    # Extract only valid voxels
    ts_valid = timeseries[:, valid_voxels]
    
    # Standardize (zero mean, unit variance)
    means = np.mean(ts_valid, axis=0)
    stds = np.std(ts_valid, axis=0)

    # Detect zero-variance voxels to avoid divide-by-zero and track impact
    zero_var_mask = stds == 0
    n_zero_var = int(np.sum(zero_var_mask))
    if debug and n_zero_var > 0:
        if n_zero_var <= 20:
            zero_idxs = np.where(valid_voxels)[0][zero_var_mask]
            print(f"    Zero-variance voxels: {n_zero_var} -> {zero_idxs}")
        else:
            print(f"    Zero-variance voxels: {n_zero_var} ({100*n_zero_var/n_valid:.2f}%)")

    stds_safe = stds.copy()
    stds_safe[zero_var_mask] = np.nan  # ensures division yields NaN without warnings
    ts_standardized = (ts_valid - means) / stds_safe
    
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
                        global_mask: Optional[np.ndarray] = None,
                        target_resolution: float = 6.0,
                        apply_censoring: bool = False,
                        debug: bool = False,
                        mask_affine: np.ndarray | None = None,
                        global_mask_file: Optional[str] = None,
                        allow_mask_resample: bool = False,
                        allow_no_mask: bool = False,
                        treatment_file: Optional[str] = None,
                        control_file: Optional[str] = None,
                        treatment_mask_file: Optional[str] = None,
                        control_mask_file: Optional[str] = None,
                        treatment_censor_file: Optional[str] = None,
                        control_censor_file: Optional[str] = None) -> tuple:
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
    global_mask : np.ndarray, optional
        Global mask (union of all individual masks) at target resolution. If None,
        will be loaded from global_mask_file or created from individual masks.
    target_resolution : float
        Target voxel resolution in mm (default 6.0)
    apply_censoring : bool, optional
        If True, apply censoring to time series before FC calculation (default: False)
    debug : bool, optional
        If True, print debugging information about NaN values in timeseries (default: False)
    mask_affine : np.ndarray, optional
        Affine to use for any debug mask outputs to ensure alignment
    global_mask_file : str, optional
        Path to a global mask to use; if provided will be downsampled and validated
    allow_mask_resample : bool, optional
        If True, automatically resample masks to match data dimensions and cache them
    allow_no_mask : bool, optional
        If True, allows running without masks (full-brain) with a warning
    
    Returns
    -------
    tuple
        (treatment_fc, control_fc, fc_change) - FC matrices for treatment and control conditions,
        and their difference (all may contain NaN for invalid voxels)
    """
    # Step 1: Validate input data
    print(f"Processing participant {participant_id}...")
    print(f"  Validating input data...")
    if treatment_file is None or control_file is None:
        treatment_file, control_file = validate_participant_data(
            participant_id, data_dir, treatment_condition, control_condition
        )
    
    # Find mask files (optional)
    if treatment_mask_file is None:
        try:
            treatment_mask_file = find_gm_mask_file(participant_id, treatment_condition, data_dir)
        except FileNotFoundError:
            print("  WARNING: Treatment mask file not found; will rely on global mask or fallback")
    if control_mask_file is None:
        try:
            control_mask_file = find_gm_mask_file(participant_id, control_condition, data_dir)
        except FileNotFoundError:
            print("  WARNING: Control mask file not found; will rely on global mask or fallback")
    
    # Step 2: Downsample nifti files
    print(f"  Downsampling to {target_resolution}x{target_resolution}x{target_resolution} mm³...")
    treatment_downsampled = downsample_nifti(treatment_file, target_resolution)
    control_downsampled = downsample_nifti(control_file, target_resolution)

    # Validate functional data dimensions
    treatment_shape = treatment_downsampled.shape[:3]
    control_shape = control_downsampled.shape[:3]
    if treatment_shape != control_shape:
        raise ValueError(f"Treatment and control data shapes differ: {treatment_shape} vs {control_shape}")

    # Load affine from cached downsampled file
    treatment_cached = _cached_downsample_path(treatment_file, target_resolution)
    if not os.path.exists(treatment_cached):
        raise FileNotFoundError(f"Cached downsampled file missing: {treatment_cached}")
    data_affine = nib.load(treatment_cached).affine
    if mask_affine is None:
        mask_affine = data_affine

    # Prepare masks: load, validate, resample, and union if needed
    global_mask, treatment_mask_downsampled, control_mask_downsampled, mask_affine = prepare_masks(
        participant_id=participant_id,
        treatment_condition=treatment_condition,
        control_condition=control_condition,
        data_dir=data_dir,
        target_resolution=target_resolution,
        data_shape=treatment_shape,
        data_affine=mask_affine,
        global_mask=global_mask,
        global_mask_file=global_mask_file,
        treatment_mask_file=treatment_mask_file,
        control_mask_file=control_mask_file,
        allow_mask_resample=allow_mask_resample,
        allow_no_mask=allow_no_mask,
        debug=debug
    )
    
    # Step 3: Apply global and individual masks, extract time series (with NaN for invalid voxels)
    print(f"  Applying masks and extracting time series...")
    treatment_timeseries = apply_global_and_individual_mask(
        treatment_downsampled, global_mask, treatment_mask_downsampled, debug=debug
    )
    control_timeseries = apply_global_and_individual_mask(
        control_downsampled, global_mask, control_mask_downsampled, debug=debug
    )
    
    # Step 4: Apply censoring if requested
    if apply_censoring:
        print(f"  Applying censoring...")
        treatment_timeseries = apply_censoring_to_timeseries(
            treatment_timeseries, participant_id, treatment_condition, data_dir,
            censor_file=treatment_censor_file
        )
        control_timeseries = apply_censoring_to_timeseries(
            control_timeseries, participant_id, control_condition, data_dir,
            censor_file=control_censor_file
        )

    # Debug: save zero-variance voxel masks
    if debug:
        tz_mask_path = os.path.join(data_dir, f"{participant_id}_{treatment_condition}_zero_var_mask.nii")
        cz_mask_path = os.path.join(data_dir, f"{participant_id}_{control_condition}_zero_var_mask.nii")
        save_zero_variance_mask(treatment_timeseries, global_mask, tz_mask_path,
                                mask_affine, target_resolution, condition_name="Treatment")
        save_zero_variance_mask(control_timeseries, global_mask, cz_mask_path,
                                mask_affine, target_resolution, condition_name="Control")
    
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
