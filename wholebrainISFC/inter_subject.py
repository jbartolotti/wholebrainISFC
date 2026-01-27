"""
Inter-subject similarity and group-level analysis.

This module handles:
- Computing inter-subject FC similarity matrices
- Creating brain maps with inter-subject correlations
- Performing mixed-effects group analysis using AFNI
- Cluster analysis for significance thresholds
"""

import os
from typing import Tuple, Dict, List
import warnings

import nibabel as nib
import numpy as np


def calculate_inter_subject_similarity_matrix(voxel_fc_profiles: np.ndarray) -> np.ndarray:
    """Pairwise Pearson correlations across participants for one voxel.

    Expects rows = participants, columns = whole-brain voxels (FC-change row for the voxel).
    Handles NaNs pairwise: each correlation uses the intersection of finite columns.
    Diagonal is set to NaN (self-correlations ignored). Fisher's Z is applied.
    """

    n_subj = voxel_fc_profiles.shape[0]
    sim = np.full((n_subj, n_subj), np.nan, dtype=float)

    for i in range(n_subj):
        row_i = voxel_fc_profiles[i]
        for j in range(i, n_subj):
            row_j = voxel_fc_profiles[j]
            valid = np.isfinite(row_i) & np.isfinite(row_j)
            if np.sum(valid) < 2:
                continue
            xi = row_i[valid]
            xj = row_j[valid]
            # Pearson correlation
            xi_d = xi - xi.mean()
            xj_d = xj - xj.mean()
            denom = np.sqrt(np.sum(xi_d ** 2) * np.sum(xj_d ** 2))
            if denom == 0:
                continue
            r = float(np.sum(xi_d * xj_d) / denom)
            r = np.clip(r, -0.999999, 0.999999)
            sim[i, j] = r
            sim[j, i] = r

    # Zero the diagonal before Fisher Z to avoid inf
    np.fill_diagonal(sim, np.nan)
    with np.errstate(invalid='ignore'):
        sim_z = np.arctanh(sim)
    return sim_z


def extract_voxel_fc_column(fc_change_matrix: np.ndarray, voxel_idx: int) -> np.ndarray:
    """
    Extract a voxel's whole-brain correlation values (one column from FC matrix).
    
    Parameters
    ----------
    fc_change_matrix : np.ndarray
        NxN FC change matrix
    voxel_idx : int
        Index of the voxel to extract
    
    Returns
    -------
    np.ndarray
        Vector of shape (N,) containing correlations for this voxel
    """
    return fc_change_matrix[voxel_idx, :]


def calculate_inter_subject_statistics(correlations: np.ndarray, self_index: int) -> Tuple[float, float]:
    """Mean and t-stat for one participant's correlations to others.

    Expects a 1D vector containing that participant's row of the ISS matrix (with self at `self_index`).
    Self entry and NaNs are excluded. Returns (mean, t) in Fisher-Z space.
    """

    mask = np.isfinite(correlations)
    if self_index < correlations.size:
        mask[self_index] = False
    vals = correlations[mask]

    n = vals.size
    if n < 2:
        return np.nan, np.nan

    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals, ddof=1))
    if std_val == 0:
        return mean_val, np.nan

    t_stat = mean_val / (std_val / np.sqrt(n))
    return mean_val, float(t_stat)


def save_brain_map_with_statistics(output_file: str, mean_values: np.ndarray,
                                   t_statistics: np.ndarray, 
                                   reference_nifti: str) -> None:
    """Write a 2-volume NIfTI (mean, t-stat) aligned to the reference affine."""

    ref_img = nib.load(reference_nifti)
    data = np.stack([mean_values, t_statistics], axis=3)
    img = nib.Nifti1Image(data, ref_img.affine)
    nib.save(img, output_file)


def process_group_inter_subject_analysis(fc_change_matrices: Dict[str, np.ndarray],
                                         participant_ids: List[str],
                                         reference_nifti: str,
                                         output_dir: str) -> Dict[str, str]:
    """Compute ISS-derived mean/t maps per participant and save to disk.

    Uses the shared global mask (reference_nifti) for voxel alignment. If <3
    participants are available, logs a warning and skips processing.
    """

    n_subj = len(participant_ids)
    if n_subj < 3:
        warnings.warn("At least 3 participants are required for group ISS; skipping.")
        return {}

    os.makedirs(output_dir, exist_ok=True)

    ref_img = nib.load(reference_nifti)
    ref_data = ref_img.get_fdata()
    global_mask = ref_data > 0
    mask_indices = global_mask.flatten() > 0
    voxel_positions = np.where(mask_indices)[0]
    n_voxels = voxel_positions.size

    # Validate FC shapes
    for pid in participant_ids:
        fc = fc_change_matrices[pid]
        if fc.shape[0] != fc.shape[1]:
            raise ValueError(f"FC change matrix for {pid} is not square: {fc.shape}")
        if fc.shape[0] != n_voxels:
            raise ValueError(
                f"FC change matrix for {pid} has {fc.shape[0]} voxels but mask has {n_voxels} voxels"
            )

    outputs: Dict[str, str] = {}

    # Pre-allocate per-participant flat arrays
    mean_flat = {pid: np.full(mask_indices.shape[0], np.nan, dtype=float) for pid in participant_ids}
    t_flat = {pid: np.full(mask_indices.shape[0], np.nan, dtype=float) for pid in participant_ids}

    # Build list of FC matrices in participant order for quick access
    fc_list = [fc_change_matrices[pid] for pid in participant_ids]

    for idx, flat_idx in enumerate(voxel_positions):
        # Collect per-participant FC-change rows for this voxel
        rows = np.stack([fc[idx, :] for fc in fc_list], axis=0)

        # Compute ISS (Fisher-Z) for this voxel
        iss_z = calculate_inter_subject_similarity_matrix(rows)

        for subj_idx, pid in enumerate(participant_ids):
            mean_val, t_val = calculate_inter_subject_statistics(iss_z[subj_idx], self_index=subj_idx)
            mean_flat[pid][flat_idx] = mean_val
            t_flat[pid][flat_idx] = t_val

    # Save outputs per participant
    for pid in participant_ids:
        mean_vol = mean_flat[pid].reshape(global_mask.shape)
        t_vol = t_flat[pid].reshape(global_mask.shape)

        subj_dir = os.path.join(output_dir, f"sub-{pid}")
        os.makedirs(subj_dir, exist_ok=True)
        out_file = os.path.join(subj_dir, f"sub-{pid}_desc-ISS_mean_tstat.nii")
        save_brain_map_with_statistics(out_file, mean_vol, t_vol, reference_nifti)
        outputs[pid] = out_file

    return outputs


def run_3dmema_analysis(brain_maps: Dict[str, Tuple[str, str]], 
                        covariates: Dict[str, Dict[str, float]],
                        covariate_names: List[str],
                        output_prefix: str) -> str:
    """
    Run AFNI's 3dMEMA for mixed-effects meta-analysis.
    
    Performs 1-sample mixed-effects analysis using mean and t-statistic
    brain maps from each participant, with specified covariates.
    
    Parameters
    ----------
    brain_maps : dict
        Dictionary mapping participant_id to (mean_nifti, tstat_nifti) tuple
    covariates : dict
        Dictionary mapping participant_id to covariate value dictionaries
    covariate_names : list
        Names of covariates to include in the model
    output_prefix : str
        Output prefix for AFNI analysis files
    
    Returns
    -------
    str
        Path to AFNI output statistics file
    """
    pass


def cluster_analysis(stats_file: str, significance_levels: List[float] = [0.05, 0.01, 0.001]) -> Dict:
    """
    Perform cluster analysis to determine cluster size thresholds.
    
    Determines minimum cluster sizes required for given significance levels.
    
    Parameters
    ----------
    stats_file : str
        Path to AFNI statistics file
    significance_levels : list
        List of p-value thresholds to analyze
    
    Returns
    -------
    dict
        Dictionary mapping p-values to minimum cluster sizes
    """
    pass


def apply_cluster_threshold(stats_file: str, cluster_threshold: int, 
                            output_file: str) -> None:
    """
    Apply cluster size threshold to statistics file.
    
    Parameters
    ----------
    stats_file : str
        Path to AFNI statistics file
    cluster_threshold : int
        Minimum cluster size (in voxels)
    output_file : str
        Path for thresholded output file
    """
    pass
