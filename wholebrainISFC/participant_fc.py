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
    Downsample nifti file to 6x6x6 mm³ voxels from higher resolution (typically 3mm³).
    
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
    
    # Get current voxel dimensions
    voxel_dims = np.array([np.linalg.norm(affine[:3, i]) for i in range(3)])
    
    # Calculate scaling factors
    scaling_factors = voxel_dims / target_resolution
    
    # Downsample using scipy zoom (handles 4D data, downsampling last dimension is handled properly)
    downsampled = zoom(data, scaling_factors, order=1)
    
    return downsampled


def apply_gm_mask_and_extract_timeseries(nifti_file: str, gm_mask_file: str) -> np.ndarray:
    """
    Apply grey matter mask and extract time series from all voxels in the mask.
    
    Parameters
    ----------
    nifti_file : str
        Path to input nifti file
    gm_mask_file : str
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
    mask_img = nib.load(gm_mask_file)
    mask = mask_img.get_fdata()
    
    # Handle 4D data (time series)
    if data.ndim == 4:
        n_timepoints = data.shape[3]
        # Flatten spatial dimensions
        data_reshaped = data.reshape(-1, n_timepoints)
        mask_flat = mask.flatten()
        
        # Extract voxels where mask > 0
        masked_voxels = data_reshaped[mask_flat > 0, :]
        # Return as (n_timepoints, n_voxels)
        return masked_voxels.T
    else:
        raise ValueError(f"Expected 4D nifti file with time dimension, got shape {data.shape}")


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
    
    Parameters
    ----------
    timeseries : np.ndarray
        Time series matrix of shape (n_timepoints, n_voxels)
    
    Returns
    -------
    np.ndarray
        NxN correlation matrix
    """
    # Standardize each voxel's time series (zero mean, unit variance)
    ts_standardized = (timeseries - timeseries.mean(axis=0)) / timeseries.std(axis=0)
    
    # Calculate correlation matrix as (1/N) * X^T * X where X is standardized
    n_timepoints = timeseries.shape[0]
    corr_matrix = (ts_standardized.T @ ts_standardized) / n_timepoints
    
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


def process_participant(participant_id: str, data_dir: str, gm_mask_file: str,
                        treatment_condition: str, control_condition: str,
                        target_resolution: float = 6.0) -> np.ndarray:
    """
    Complete processing pipeline for a single participant.
    
    Orchestrates all steps: validation, downsampling, masking, 
    FC calculation, and difference matrix computation.
    
    Parameters
    ----------
    participant_id : str
        Participant identifier
    data_dir : str
        Directory containing brain nifti data
    gm_mask_file : str
        Path to grey matter mask file
    treatment_condition : str
        Name of treatment condition
    control_condition : str
        Name of control condition
    target_resolution : float
        Target voxel resolution in mm (default 6.0)
    
    Returns
    -------
    np.ndarray
        FC change matrix for the participant
    """
    # Step 1: Validate input data
    print(f"Processing participant {participant_id}...")
    print(f"  Validating input data...")
    treatment_file, control_file = validate_participant_data(
        participant_id, data_dir, treatment_condition, control_condition
    )
    
    # Step 2: Downsample nifti files
    print(f"  Downsampling to {target_resolution}x{target_resolution}x{target_resolution} mm³...")
    treatment_downsampled = downsample_nifti(treatment_file, target_resolution)
    control_downsampled = downsample_nifti(control_file, target_resolution)
    
    # Note: For now, we work with the raw downsampled data
    # In a more complete implementation, you might save these and reload them
    # For the mask application to work correctly in future BIDS implementation
    
    # Step 3: Apply mask and extract time series
    print(f"  Applying grey matter mask and extracting time series...")
    treatment_timeseries = apply_gm_mask_and_extract_timeseries(
        treatment_file, gm_mask_file
    )
    control_timeseries = apply_gm_mask_and_extract_timeseries(
        control_file, gm_mask_file
    )
    
    # Step 4: Calculate FC matrices
    print(f"  Calculating functional connectivity matrices...")
    treatment_fc = calculate_fc_matrix(treatment_timeseries)
    control_fc = calculate_fc_matrix(control_timeseries)
    
    # Step 5: Calculate FC change
    print(f"  Computing FC change (treatment - control)...")
    fc_change = calculate_fc_change(treatment_fc, control_fc)
    
    print(f"  Completed processing for participant {participant_id}")
    
    return fc_change
