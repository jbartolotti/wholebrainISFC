"""
Inter-subject similarity and group-level analysis.

This module handles:
- Computing inter-subject FC similarity matrices
- Creating brain maps with inter-subject correlations
- Performing mixed-effects group analysis using AFNI
- Cluster analysis for significance thresholds
"""

from typing import Tuple, Dict, List
import numpy as np


def calculate_inter_subject_similarity_matrix(voxel_fc_profiles: np.ndarray) -> np.ndarray:
    """
    Calculate inter-subject similarity matrix for a single voxel.
    
    For each voxel, uses whole-brain correlation values (FC change matrix columns)
    and cross-correlates each participant. Applies Fisher's Z transformation.
    
    Parameters
    ----------
    voxel_fc_profiles : np.ndarray
        Matrix of shape (n_participants, n_voxels_in_brain) containing the
        whole-brain FC values for a single voxel across all participants
    
    Returns
    -------
    np.ndarray
        SxS similarity matrix where S is number of participants,
        with Fisher's Z-transformed correlation values
    """
    pass


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
    pass


def calculate_inter_subject_statistics(similarity_matrix: np.ndarray) -> Tuple[float, float]:
    """
    Calculate mean and t-statistic of inter-subject correlations.
    
    Parameters
    ----------
    similarity_matrix : np.ndarray
        SxS similarity matrix
    
    Returns
    -------
    tuple
        (mean_correlation, t_statistic)
    """
    pass


def save_brain_map_with_statistics(output_file: str, mean_values: np.ndarray,
                                   t_statistics: np.ndarray, 
                                   reference_nifti: str) -> None:
    """
    Save mean and t-statistic values back into a brain nifti file with subbricks.
    
    For each participant and each voxel, creates a nifti file with separate
    subbricks for the mean inter-subject correlation and the t-statistic.
    
    Parameters
    ----------
    output_file : str
        Path where output nifti file should be saved
    mean_values : np.ndarray
        Brain map of mean inter-subject correlations
    t_statistics : np.ndarray
        Brain map of t-statistics
    reference_nifti : str
        Path to reference nifti file for header/affine information
    """
    pass


def process_group_inter_subject_analysis(fc_change_matrices: Dict[str, np.ndarray],
                                         participant_ids: List[str],
                                         reference_nifti: str,
                                         output_dir: str) -> Dict[str, np.ndarray]:
    """
    Process inter-subject analysis for all participants.
    
    For each participant and voxel, calculates mean and t-statistic of
    inter-subject FC similarity and saves as brain maps.
    
    Parameters
    ----------
    fc_change_matrices : dict
        Dictionary mapping participant_id to their NxN FC change matrix
    participant_ids : list
        List of participant identifiers
    reference_nifti : str
        Path to reference nifti file for anatomical structure
    output_dir : str
        Directory where output brain maps should be saved
    
    Returns
    -------
    dict
        Dictionary mapping participant_id to their output nifti file path
    """
    pass


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
