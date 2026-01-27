"""
Data loading, validation, and mask preparation utilities for participant-level FC analysis.
"""

import os
from typing import Dict, List, Tuple, Optional

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


def validate_participant_data(participant_id: str, data_dir: str,
                              treatment_condition: str,
                              control_condition: str) -> Tuple[str, str]:
    """Confirm input data for a participant.

    Verifies that nifti files exist for both treatment and control conditions,
    already warped to MNI template.
    """
    treatment_name = f"{participant_id}_{treatment_condition}"
    control_name = f"{participant_id}_{control_condition}"

    treatment_file = None
    control_file = None

    for ext in ['.nii.gz', '.nii']:
        candidate = os.path.join(data_dir, treatment_name + ext)
        if os.path.exists(candidate):
            treatment_file = candidate
            break

    for ext in ['.nii.gz', '.nii']:
        candidate = os.path.join(data_dir, control_name + ext)
        if os.path.exists(candidate):
            control_file = candidate
            break

    if treatment_file is None:
        raise FileNotFoundError(
            f"Treatment condition file not found for participant {participant_id}: "
            f"expected {os.path.join(data_dir, treatment_name + '.[nii|nii.gz]')}"
        )

    if control_file is None:
        raise FileNotFoundError(
            f"Control condition file not found for participant {participant_id}: "
            f"expected {os.path.join(data_dir, control_name + '.[nii|nii.gz]')}"
        )

    return treatment_file, control_file


def _cached_downsample_path(nifti_file: str, target_resolution: float = 6.0) -> str:
    """Return the expected cached filename for a downsampled NIfTI."""
    base_path, ext = os.path.splitext(nifti_file)
    if ext == '.gz':
        base_path, ext2 = os.path.splitext(base_path)
        ext = ext2 + ext
    return f"{base_path}_{int(target_resolution)}mm{ext}"


def downsample_nifti(nifti_file: str, target_resolution: float = 6.0) -> np.ndarray:
    """Downsample nifti file to target resolution, with caching."""
    cached_file = _cached_downsample_path(nifti_file, target_resolution)

    if os.path.exists(cached_file):
        print(f"    Loading cached downsampled file: {os.path.basename(cached_file)}")
        img = nib.load(cached_file)
        return img.get_fdata()

    img = nib.load(nifti_file)
    data = img.get_fdata()
    affine = img.affine

    voxel_dims = np.array([np.linalg.norm(affine[:3, i]) for i in range(3)])
    spatial_scaling = voxel_dims / target_resolution

    if data.ndim == 4:
        scaling_factors = list(spatial_scaling) + [1.0]
    else:
        scaling_factors = spatial_scaling

    print(f"    Downsampling and caching: {os.path.basename(cached_file)}")
    downsampled = zoom(data, scaling_factors, order=1)

    new_affine = affine.copy()
    for i in range(3):
        new_affine[:3, i] = affine[:3, i] * (1.0 / spatial_scaling[i])

    nib.save(nib.Nifti1Image(downsampled, new_affine), cached_file)
    return downsampled


def downsample_mask(mask_file: str, target_resolution: float = 6.0) -> np.ndarray:
    """Downsample a mask to target resolution with nearest-neighbor and caching."""
    cached_file = _cached_downsample_path(mask_file, target_resolution)

    if os.path.exists(cached_file):
        print(f"    Loading cached downsampled mask: {os.path.basename(cached_file)}")
        img = nib.load(cached_file)
        return img.get_fdata()

    img = nib.load(mask_file)
    data = img.get_fdata()
    affine = img.affine

    if data.ndim == 4:
        print(f"  Note: Mask file is 4D (shape: {data.shape}). Taking mean across time dimension.")
        data = np.mean(data, axis=3)
        data = (data > 0).astype(np.float32)
    elif data.ndim == 3:
        data = data.astype(np.float32)
    else:
        raise ValueError(f"Expected 3D or 4D mask file, got shape: {data.shape}")

    voxel_dims = np.array([np.linalg.norm(affine[:3, i]) for i in range(3)])
    spatial_scaling = voxel_dims / target_resolution

    print(f"    Downsampling and caching mask: {os.path.basename(cached_file)}")
    downsampled = zoom(data, spatial_scaling, order=0)

    new_affine = affine.copy()
    for i in range(3):
        new_affine[:3, i] = affine[:3, i] * (1.0 / spatial_scaling[i])

    nib.save(nib.Nifti1Image(downsampled, new_affine), cached_file)
    return downsampled


def resample_mask_to_target_shape(mask: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Resample a mask to a target 3D shape using nearest neighbor."""
    if mask.shape == target_shape:
        return mask
    zoom_factors = [target_shape[i] / mask.shape[i] for i in range(3)]
    return zoom(mask, zoom_factors, order=0)


def create_global_mask(participant_ids: List[str], conditions: List[str], data_dir: str,
                       target_resolution: float = 6.0) -> np.ndarray:
    """Create a global mask as the union (logical OR) of all individual masks."""
    global_mask = None
    for pid in participant_ids:
        for condition in conditions:
            try:
                mask_file = find_gm_mask_file(pid, condition, data_dir)
                downsampled_mask = downsample_mask(mask_file, target_resolution)
                if global_mask is None:
                    global_mask = (downsampled_mask > 0).astype(np.float32)
                else:
                    global_mask = np.logical_or(global_mask, downsampled_mask > 0).astype(np.float32)
            except FileNotFoundError:
                print(f"  Warning: Mask not found for {pid}, {condition}. Skipping.")
                continue
    if global_mask is None:
        raise ValueError("No valid masks found to create global mask")
    return global_mask


def prepare_masks(participant_id: str,
                  treatment_condition: str,
                  control_condition: str,
                  data_dir: str,
                  target_resolution: float,
                  data_shape: tuple,
                  data_affine: np.ndarray,
                  global_mask: Optional[np.ndarray] = None,
                  global_mask_file: Optional[str] = None,
                  treatment_mask_file: Optional[str] = None,
                  control_mask_file: Optional[str] = None,
                  allow_mask_resample: bool = False,
                  allow_no_mask: bool = False,
                  debug: bool = False) -> tuple:
    """Load, validate, and optionally resample masks to match data dimensions."""

    target_shape = data_shape

    def _resample_and_cache(mask: np.ndarray, source_file: Optional[str], label: str) -> np.ndarray:
        if mask.shape == target_shape:
            return mask
        if not allow_mask_resample:
            raise ValueError(
                f"Spatial dimensions mismatch for {label}. Data: {target_shape}, Mask: {mask.shape}. "
                "Enable allow_mask_resample to resample masks automatically."
            )
        print(f"  WARNING: Resampling {label} mask from {mask.shape} to {target_shape}")
        resampled = resample_mask_to_target_shape(mask, target_shape)

        if source_file:
            base_path, ext = os.path.splitext(source_file)
            if ext == '.gz':
                base_path, ext2 = os.path.splitext(base_path)
                ext = ext2 + ext
            save_path = f"{base_path}_resampled_{target_shape[0]}x{target_shape[1]}x{target_shape[2]}.nii"
        else:
            save_path = os.path.join(
                data_dir,
                f"{participant_id}_{label}_resampled_{target_shape[0]}x{target_shape[1]}x{target_shape[2]}.nii"
            )

        nib.save(nib.Nifti1Image(resampled.astype(np.float32), data_affine), save_path)
        if debug:
            print(f"    DEBUG: Saved resampled {label} mask to {save_path}")
        return resampled

    if global_mask is None and global_mask_file is not None:
        print(f"  Loading global mask from file: {global_mask_file}")
        if not os.path.exists(global_mask_file):
            raise FileNotFoundError(f"Global mask file not found: {global_mask_file}")
        global_mask = downsample_mask(global_mask_file, target_resolution)

    treatment_mask = None
    control_mask = None
    if treatment_mask_file and os.path.exists(treatment_mask_file):
        treatment_mask = downsample_mask(treatment_mask_file, target_resolution)
    if control_mask_file and os.path.exists(control_mask_file):
        control_mask = downsample_mask(control_mask_file, target_resolution)

    if global_mask is None:
        masks_to_union = [m for m in [treatment_mask, control_mask] if m is not None]
        if masks_to_union:
            print("  No global mask provided; creating union of available individual masks")
            global_mask = np.logical_or.reduce([m > 0 for m in masks_to_union]).astype(np.float32)
        elif allow_no_mask:
            print("  WARNING: Proceeding without masks (full-brain) because allow_no_mask is set")
            global_mask = np.ones(target_shape, dtype=np.float32)
        else:
            raise ValueError("No masks available. Provide a global mask, individual masks, or set allow_no_mask=True.")

    if treatment_mask is None:
        print("  WARNING: Treatment mask missing; using global mask for treatment")
        treatment_mask = global_mask.copy()
    if control_mask is None:
        print("  WARNING: Control mask missing; using global mask for control")
        control_mask = global_mask.copy()

    global_mask = _resample_and_cache(global_mask, global_mask_file, "global")
    treatment_mask = _resample_and_cache(treatment_mask, treatment_mask_file, "treatment")
    control_mask = _resample_and_cache(control_mask, control_mask_file, "control")

    updated_global = False
    for label, indiv_mask in [("treatment", treatment_mask), ("control", control_mask)]:
        outside = (indiv_mask > 0) & (global_mask <= 0)
        if np.any(outside):
            n_out = int(np.sum(outside))
            print(f"  WARNING: {label} mask extends outside global mask by {n_out} voxels; expanding global mask")
            global_mask = np.logical_or(global_mask, indiv_mask > 0).astype(np.float32)
            updated_global = True

    if updated_global:
        if global_mask_file is not None:
            base_path, ext = os.path.splitext(global_mask_file)
            if ext == '.gz':
                base_path, ext2 = os.path.splitext(base_path)
                ext = ext2 + ext
            expanded_path = f"{base_path}_expanded_{target_shape[0]}x{target_shape[1]}x{target_shape[2]}.nii"
        else:
            expanded_path = os.path.join(
                data_dir,
                f"{participant_id}_global_expanded_{target_shape[0]}x{target_shape[1]}x{target_shape[2]}.nii"
            )
        nib.save(nib.Nifti1Image(global_mask.astype(np.float32), data_affine), expanded_path)
        if debug:
            print(f"    DEBUG: Saved expanded global mask to {expanded_path}")

    return global_mask.astype(np.float32), treatment_mask.astype(np.float32), control_mask.astype(np.float32), data_affine


def find_gm_mask_file(participant_id: str, condition: str, data_dir: str) -> str:
    """Find the participant-specific grey matter mask file."""
    gm_mask_name = f"{participant_id}_{condition}_GM"
    for ext in ['.nii.gz', '.nii']:
        candidate = os.path.join(data_dir, gm_mask_name + ext)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"Grey matter mask not found for participant {participant_id}, condition {condition}: "
        f"expected {os.path.join(data_dir, gm_mask_name + '.[nii|nii.gz]')}"
    )


def load_censoring_vector(participant_id: str, condition: str, data_dir: str, n_timepoints: int,
                          censor_file: Optional[str] = None) -> np.ndarray:
    """Load censoring vector for a participant and condition."""
    if censor_file:
        if not os.path.exists(censor_file):
            raise FileNotFoundError(f"Censoring file not found at provided path: {censor_file}")
    else:
        censor_base = f"{participant_id}_{condition}_censor"
        for ext in ['.1D', '.tsv', '.csv', '.txt']:
            candidate = os.path.join(data_dir, censor_base + ext)
            if os.path.exists(candidate):
                censor_file = candidate
                break
        if censor_file is None:
            raise FileNotFoundError(
                f"Censoring file not found for participant {participant_id}, condition {condition}: "
                f"expected {os.path.join(data_dir, censor_base + '.[1D|tsv|csv|txt]')}"
            )

    censor_vector = np.loadtxt(censor_file)
    if censor_vector.ndim != 1:
        raise ValueError(f"Censoring vector must be 1D, got shape {censor_vector.shape}")
    if len(censor_vector) != n_timepoints:
        raise ValueError(
            f"Censoring vector length {len(censor_vector)} does not match number of timepoints {n_timepoints}"
        )
    return censor_vector.astype(bool)


def apply_censoring_to_timeseries(timeseries: np.ndarray, participant_id: str,
                                   condition: str, data_dir: str,
                                   censor_file: Optional[str] = None) -> np.ndarray:
    """Apply censoring to time series data using participant/condition censor file."""
    n_timepoints = timeseries.shape[0]
    censor_mask = load_censoring_vector(participant_id, condition, data_dir, n_timepoints, censor_file)
    return timeseries[censor_mask, :]
