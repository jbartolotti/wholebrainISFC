"""
Inter-subject similarity and group-level analysis.

This module handles:
- Computing inter-subject FC similarity matrices
- Creating brain maps with inter-subject correlations
- Performing mixed-effects group analysis using AFNI
- Cluster analysis for significance thresholds

Memory Optimization:
  For large datasets (e.g., 37 participants × 6162×6162 voxels = ~11 GB raw),
  use the voxel_chunk_size parameter in process_group_inter_subject_analysis():
  
    - voxel_chunk_size=500 (default): ~19 MB per chunk, 26% memory savings
    - voxel_chunk_size=300: ~11 MB per chunk, better for 8 GB systems
    - voxel_chunk_size=1000: ~38 MB per chunk, faster on 32+ GB systems
  
  Also enable use_float32=True to save 50% disk space. See MEMORY_OPTIMIZATION.md
  for detailed analysis and recommendations.
"""

import json
import os
from typing import Tuple, Dict, List, Optional, Any
import warnings

import nibabel as nib
import numpy as np
import pandas as pd


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
    """Write a 2-volume NIfTI (mean, t-stat) aligned to the reference affine.
    
    If output_file ends with .gz, the NIfTI will be automatically gzipped.
    """

    ref_img = nib.load(reference_nifti)
    data = np.stack([mean_values, t_statistics], axis=3)
    img = nib.Nifti1Image(data, ref_img.affine)
    nib.save(img, output_file)


def process_group_inter_subject_analysis(fc_change_matrices: Dict[str, np.ndarray],
                                         participant_ids: List[str],
                                         reference_nifti: str,
                                         output_dir: str,
                                         voxel_chunk_size: int = 500,
                                         use_float32: bool = True) -> Dict[str, str]:
    """Compute ISS-derived mean/t maps per participant and save to disk.

    Uses the shared global mask (reference_nifti) for voxel alignment. If <3
    participants are available, logs a warning and skips processing.
    
    Parameters
    ----------
    fc_change_matrices : Dict[str, np.ndarray]
        FC change matrices for each participant (can be large)
    participant_ids : List[str]
        List of participant IDs
    reference_nifti : str
        Path to reference NIfTI for mask and affine
    output_dir : str
        Output directory for ISS results
    voxel_chunk_size : int
        Number of voxels to process at once. Smaller = less memory, slower computation.
        Default 500 processes ~19 MB at a time for 37 participants.
        Reduce to 200-300 if running out of memory; increase to 1000+ if memory-rich system.
    use_float32 : bool
        Store output as float32 instead of float64 (saves 50% disk/memory for ISS results)
    
    Returns
    -------
    Dict[str, str]
        Paths to output NIfTI files for each participant
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
    
    # Use float32 for output to save space
    dtype_out = np.float32 if use_float32 else float
    
    # Pre-allocate per-participant flat arrays
    mean_flat = {pid: np.full(mask_indices.shape[0], np.nan, dtype=dtype_out) for pid in participant_ids}
    t_flat = {pid: np.full(mask_indices.shape[0], np.nan, dtype=dtype_out) for pid in participant_ids}

    # Build list of FC matrices in participant order for quick access
    fc_list = [fc_change_matrices[pid] for pid in participant_ids]
    
    # Estimate memory usage and report
    chunk_mem_mb = (voxel_chunk_size * n_subj * 6162 * 8) / (1024 * 1024)
    print(f"Processing {n_voxels} voxels in chunks of {voxel_chunk_size} (~{chunk_mem_mb:.1f} MB per chunk)")

    # Process voxels in chunks to avoid memory explosion
    for chunk_start in range(0, len(voxel_positions), voxel_chunk_size):
        chunk_end = min(chunk_start + voxel_chunk_size, len(voxel_positions))
        chunk_indices = voxel_positions[chunk_start:chunk_end]
        
        if (chunk_end - chunk_start) % 100 == 0 or chunk_end == len(voxel_positions):
            print(f"  Processing voxels {chunk_start}-{chunk_end}/{len(voxel_positions)}")

        # Process each voxel in the chunk
        for local_idx, flat_idx in enumerate(chunk_indices):
            idx = chunk_start + local_idx
            
            # Collect per-participant FC-change rows for this voxel
            # Build this incrementally to avoid large temporary array
            rows = np.empty((n_subj, 6162), dtype=np.float32)
            for subj_idx, fc in enumerate(fc_list):
                rows[subj_idx, :] = fc[idx, :]

            # Compute ISS (Fisher-Z) for this voxel
            iss_z = calculate_inter_subject_similarity_matrix(rows)

            for subj_idx, pid in enumerate(participant_ids):
                mean_val, t_val = calculate_inter_subject_statistics(iss_z[subj_idx], self_index=subj_idx)
                mean_flat[pid][flat_idx] = dtype_out(mean_val)
                t_flat[pid][flat_idx] = dtype_out(t_val)

    # Save outputs per participant
    for pid in participant_ids:
        mean_vol = mean_flat[pid].reshape(global_mask.shape)
        t_vol = t_flat[pid].reshape(global_mask.shape)

        # BIDS-compliant output directory: derivatives/wholebrainISFC/sub-{pid}/
        subj_dir = os.path.join(output_dir, f"sub-{pid}")
        os.makedirs(subj_dir, exist_ok=True)
        
        # BIDS-compliant filename: sub-{pid}_desc-ISS_mean_tstat.nii.gz
        # (2-volume: volume 0 = mean, volume 1 = t-stat, both in Fisher-Z space)
        out_file = os.path.join(subj_dir, f"sub-{pid}_desc-ISS_mean_tstat.nii.gz")
        save_brain_map_with_statistics(out_file, mean_vol, t_vol, reference_nifti)
        outputs[pid] = out_file
        
        # Free memory for this participant's results before saving next
        del mean_flat[pid]
        del t_flat[pid]

    return outputs



def load_participants_metadata(bids_dir: str) -> Dict[str, Any]:
    """Load BIDS participants.json metadata file.
    
    Parameters
    ----------
    bids_dir : str
        BIDS root directory.
    
    Returns
    -------
    dict
        Dictionary with variable metadata (Type, Levels, ReferenceLevels, etc.)
    """
    metadata_file = os.path.join(bids_dir, "participants.json")
    if not os.path.exists(metadata_file):
        print(f"Warning: participants.json not found at {metadata_file}")
        return {}
    
    with open(metadata_file, 'r') as f:
        return json.load(f)


def load_participants_data(bids_dir: str) -> pd.DataFrame:
    """Load BIDS participants.tsv file.
    
    Parameters
    ----------
    bids_dir : str
        BIDS root directory.
    
    Returns
    -------
    pd.DataFrame
        Participants data with participant_id as index.
    """
    participants_file = os.path.join(bids_dir, "participants.tsv")
    if not os.path.exists(participants_file):
        raise FileNotFoundError(f"participants.tsv not found: {participants_file}")
    
    df = pd.read_csv(participants_file, sep="\t")
    df = df.set_index("participant_id")
    return df


def prep_covariates_for_3dmema(
    participant_ids: List[str],
    covariate_names: List[str],
    bids_dir: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Prepare covariates for 3dMEMA analysis.
    
    Reads covariate values from participants.tsv, applies contrast coding
    for categorical variables (centered at 0) and z-scoring for continuous
    variables (mean=0, sd=1).
    
    For categorical variables, the reference level (from participants.json)
    receives the negative value in contrast coding.
    
    For continuous variables, if already normalized (mean≈0, sd≈1) or marked
    as "Standardized": true in participants.json, normalization is skipped.
    
    Parameters
    ----------
    participant_ids : list of str
        Participant IDs (without "sub-" prefix).
    covariate_names : list of str
        Names of covariates to include (must exist in participants.tsv).
    bids_dir : str
        BIDS root directory.
    
    Returns
    -------
    tuple of (covariate_data, metadata)
        covariate_data : dict mapping participant_id to dict of covariate values
        metadata : dict with covariate types and reference levels
    """
    metadata = load_participants_metadata(bids_dir)
    df = load_participants_data(bids_dir)
    
    # First pass: collect raw values
    raw_values = {cov: [] for cov in covariate_names}
    pid_mapping = {}  # Map clean_pid to row
    
    for pid in participant_ids:
        full_pid = f"sub-{pid}" if not pid.startswith("sub-") else pid
        clean_pid = pid.replace("sub-", "") if pid.startswith("sub-") else pid
        
        # Find row in dataframe (may be with or without sub- prefix)
        row = None
        for idx in df.index:
            if idx == full_pid or idx == clean_pid or idx == pid:
                row = df.loc[idx]
                break
        
        if row is None:
            raise ValueError(f"Participant {pid} not found in participants.tsv")
        
        pid_mapping[clean_pid] = row
        
        for cov_name in covariate_names:
            if cov_name not in row.index:
                raise ValueError(f"Covariate '{cov_name}' not found in participants.tsv for {pid}")
            
            value = row[cov_name]
            
            # Handle NaN/missing values
            if pd.isna(value):
                raise ValueError(f"Missing value for covariate '{cov_name}' in participant {pid}")
            
            raw_values[cov_name].append(value)
    
    # Second pass: apply coding schemes
    covariate_data = {}
    
    for cov_name in covariate_names:
        values = raw_values[cov_name]
        
        # Determine if categorical or continuous
        is_categorical = (cov_name in metadata and 
                         metadata[cov_name].get("Type") == "categorical")
        
        if is_categorical:
            # Contrast coding for categorical variables
            unique_levels = sorted(set(str(v) for v in values))
            n_levels = len(unique_levels)
            
            # Get reference level from metadata
            ref_level = None
            if cov_name in metadata:
                ref_level = metadata[cov_name].get("ReferenceLevels", None)
            
            if n_levels == 2:
                # Binary variable: use -0.5 and +0.5
                # Reference level gets -0.5 (if specified), otherwise first alphabetically
                if ref_level and str(ref_level) in unique_levels:
                    # Reference gets negative, other gets positive
                    other_level = [lev for lev in unique_levels if str(lev) != str(ref_level)][0]
                    level_codes = {str(ref_level): -0.5, other_level: +0.5}
                else:
                    # No reference specified or not found: use alphabetical order
                    level_codes = {unique_levels[0]: -0.5, unique_levels[1]: +0.5}
                print(f"  Contrast coding for '{cov_name}': {level_codes}")
            else:
                # Multi-level categorical: use deviation coding (sum to zero)
                # Reference level (if specified) gets most negative value
                if ref_level and str(ref_level) in unique_levels:
                    # Put reference first, then others alphabetically
                    ordered_levels = [str(ref_level)] + [lev for lev in unique_levels if str(lev) != str(ref_level)]
                else:
                    ordered_levels = unique_levels
                
                step = 1.0 / (n_levels - 1) if n_levels > 1 else 0
                level_codes = {ordered_levels[i]: -0.5 + i * step 
                              for i in range(n_levels)}
                print(f"  Contrast coding for '{cov_name}': {level_codes}")
            
            # Apply coding to each participant
            for idx, pid in enumerate(participant_ids):
                clean_pid = pid.replace("sub-", "") if pid.startswith("sub-") else pid
                if clean_pid not in covariate_data:
                    covariate_data[clean_pid] = {}
                str_value = str(values[idx])
                covariate_data[clean_pid][cov_name] = level_codes[str_value]
        else:
            # Continuous variable: check if already standardized, then z-score if needed
            numeric_values = np.array([float(v) for v in values])
            mean_val = np.mean(numeric_values)
            std_val = np.std(numeric_values, ddof=1)
            
            # Check if marked as pre-standardized in metadata
            is_standardized = (cov_name in metadata and 
                             metadata[cov_name].get("Standardized", False))
            
            # Or detect if already normalized (mean≈0, sd≈1, tolerance of 0.1)
            is_already_normalized = (abs(mean_val) < 0.1 and abs(std_val - 1.0) < 0.1)
            
            if is_standardized:
                print(f"  '{cov_name}' marked as pre-standardized; using raw values")
                z_scores = numeric_values
            elif is_already_normalized:
                print(f"  '{cov_name}' detected as already normalized (mean={mean_val:.3f}, sd={std_val:.3f}); using raw values")
                z_scores = numeric_values
            elif std_val == 0:
                print(f"  Warning: '{cov_name}' has zero variance; using raw values")
                z_scores = numeric_values
            else:
                z_scores = (numeric_values - mean_val) / std_val
                print(f"  Z-scored '{cov_name}': mean={mean_val:.2f}, sd={std_val:.2f}")
            
            # Apply z-scores to each participant
            for idx, pid in enumerate(participant_ids):
                clean_pid = pid.replace("sub-", "") if pid.startswith("sub-") else pid
                if clean_pid not in covariate_data:
                    covariate_data[clean_pid] = {}
                covariate_data[clean_pid][cov_name] = float(z_scores[idx])
    
    return covariate_data, metadata


def validate_covariates(covariate_names: List[str], covariate_data: Dict[str, Dict[str, Any]]) -> None:
    """Validate that all participants have non-NaN covariate values.
    
    Parameters
    ----------
    covariate_names : list of str
        Names of covariates.
    covariate_data : dict
        Mapping of participant_id to covariate values.
    
    Raises
    ------
    ValueError
        If any covariate values are missing or invalid.
    """
    for pid, covs in covariate_data.items():
        for cov_name in covariate_names:
            if cov_name not in covs:
                raise ValueError(f"Missing covariate '{cov_name}' for participant {pid}")
            if not np.isfinite(covs[cov_name]):
                raise ValueError(f"Invalid covariate value for '{cov_name}' in participant {pid}: {covs[cov_name]}")


def run_3dmema_analysis(
    iss_results: Dict[str, str],
    output_dir: str,
    set_label: str = "ISS",
    bids_dir: Optional[str] = None,
    covariate_names: Optional[List[str]] = None,
) -> str:
    """
    Run AFNI's 3dMEMA for mixed-effects meta-analysis on ISS results.
    
    Performs one-sample group analysis using mean and t-statistic volumes
    from each participant's ISS brain maps. Optionally includes covariates.
    
    Parameters
    ----------
    iss_results : dict
        Dictionary mapping participant_id to ISS NIfTI file path.
        Each file should contain 2 volumes: [0] = mean, [1] = t-stat.
    output_dir : str
        Output directory for 3dMEMA results.
    set_label : str
        Label for the analysis set (default "ISS").
    bids_dir : str, optional
        BIDS root directory (required if using covariates).
    covariate_names : list of str, optional
        Names of covariates to include in the model. Must exist in participants.tsv.
    
    Returns
    -------
    str
        Path to 3dMEMA output statistics file (+tlrc.HEAD).
        
    Raises
    ------
    RuntimeError
        If 3dMEMA command fails or is not found.
    FileNotFoundError
        If any ISS input files are missing.
    ValueError
        If covariates are requested but bids_dir is not provided, or covariate data is invalid.
    """
    import subprocess
    
    if len(iss_results) < 3:
        raise ValueError(f"3dMEMA requires at least 3 subjects; got {len(iss_results)}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Validate all input files exist
    for pid, nifti_path in iss_results.items():
        if not os.path.exists(nifti_path):
            raise FileNotFoundError(f"ISS file not found for {pid}: {nifti_path}")
    
    # Prepare covariates if requested
    covariate_data = {}
    covariate_file = None
    
    if covariate_names:
        if not bids_dir:
            raise ValueError("bids_dir is required to load covariates")
        if not covariate_names:
            covariate_names = []
        
        covariate_data, metadata = prep_covariates_for_3dmema(
            list(iss_results.keys()),
            covariate_names,
            bids_dir,
        )
        validate_covariates(covariate_names, covariate_data)
        print(f"  Covariates: {covariate_names}")
        for pid, covs in covariate_data.items():
            print(f"    {pid}: {covs}")
        
        # Write covariates to a file for 3dMEMA
        covariate_file = os.path.join(output_dir, f"covariates_{set_label}.txt")
        _write_covariates_file(covariate_file, covariate_names, covariate_data, 
                              sorted(iss_results.keys()))
        print(f"  Covariates file: {covariate_file}")
    
    # Build 3dMEMA command
    output_prefix = os.path.join(output_dir, f"3dMEMA_{set_label}")
    
    cmd = [
        "3dMEMA",
        "-prefix", output_prefix,
        "-set", set_label,
    ]
    
    # Add participant data: sub-ID mean_volume'[0]' tstat_volume'[1]'
    for pid, nifti_path in sorted(iss_results.items()):
        clean_pid = pid.replace("sub-", "") if pid.startswith("sub-") else pid
        cmd.extend([
            f"sub-{clean_pid}",
            f"{nifti_path}[0]",  # Mean volume
            f"{nifti_path}[1]",  # T-stat volume
        ])
    
    # Add covariates file and centering AFTER -set and subject data
    if covariate_file:
        cmd.extend(["-covariates", covariate_file])
        # Since we've already z-scored/contrast-coded, center at 0 (no additional centering)
        # Each covariate center is specified as: COV_NAME = CENTER_VALUE
        cmd.append("-covariates_center")
        for cov_name in covariate_names:
            cmd.extend([cov_name, "=", "0"])
    
    print("\nRunning 3dMEMA group analysis...")
    print(f"  Participants: {sorted(iss_results.keys())}")
    print(f"  Output prefix: {output_prefix}")
    if covariate_file:
        print(f"  Covariates file: {covariate_file}")
    print(f"  Command: {' '.join(cmd[:10])}...")  # Print first 10 args to avoid huge output
    
    # Run 3dMEMA
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        print("  3dMEMA stdout:")
        for line in result.stdout.strip().split('\n')[-10:]:  # Print last 10 lines
            print(f"    {line}")
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: 3dMEMA failed with return code {e.returncode}")
        print(f"  stderr: {e.stderr}")
        raise RuntimeError(f"3dMEMA failed: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("3dMEMA command not found. Ensure AFNI is installed and in PATH.")
    
    # Return the output file path (AFNI creates +tlrc.HEAD/BRIK)
    output_file = output_prefix + "+tlrc.HEAD"
    if not os.path.exists(output_file):
        # Try .nii if 3dMEMA was configured to output NIfTI
        output_file = output_prefix + ".nii"
        if not os.path.exists(output_file):
            output_file = output_prefix + ".nii.gz"
    
    return output_file


def _write_covariates_file(
    filepath: str,
    covariate_names: List[str],
    covariate_data: Dict[str, Dict[str, float]],
    participant_ids: List[str],
) -> None:
    """Write covariates to a text file in 3dMEMA format.
    
    Creates a tab-separated file with subject IDs in first column and
    covariate values in subsequent columns.
    
    Parameters
    ----------
    filepath : str
        Output file path.
    covariate_names : list of str
        Names of covariates (column headers).
    covariate_data : dict
        Mapping of participant_id to covariate values.
    participant_ids : list of str
        Participant IDs in desired order.
    """
    with open(filepath, 'w') as f:
        # Write header
        header = ['subj'] + covariate_names
        f.write('\t'.join(header) + '\n')
        
        # Write data rows
        for pid in participant_ids:
            clean_pid = pid.replace("sub-", "") if pid.startswith("sub-") else pid
            row = [f"sub-{clean_pid}"]
            if clean_pid in covariate_data:
                for cov_name in covariate_names:
                    row.append(str(covariate_data[clean_pid][cov_name]))
            f.write('\t'.join(row) + '\n')


def run_3dmema_with_covariates(brain_maps: Dict[str, Tuple[str, str]], 
                               covariates: Dict[str, Dict[str, float]],
                               covariate_names: List[str],
                               output_prefix: str) -> str:
    """
    Run AFNI's 3dMEMA for mixed-effects meta-analysis with covariates.
    
    Performs mixed-effects analysis using mean and t-statistic
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
    
    Notes
    -----
    Placeholder for future covariate-based analyses.
    """
    raise NotImplementedError("Covariate-based 3dMEMA not yet implemented.")


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
