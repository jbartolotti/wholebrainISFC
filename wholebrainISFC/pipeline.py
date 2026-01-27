"""
Pipeline interfaces for whole-brain ISFC.

Provides user-facing entry points to run participant-level processing and, later,
group-level analyses. Keeps test scripts minimal by centralizing orchestration here.
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from . import config
from . import participant_fc
from . import inter_subject  # noqa: F401 (placeholder for future use)
from .data_prep import (
    downsample_mask,
    create_global_mask,
    _cached_downsample_path,
)


def _rescale_affine_to_resolution(affine: np.ndarray, target_resolution: float) -> np.ndarray:
    """Scale spatial axes of an affine to a desired voxel size, preserve translation."""
    new_affine = affine.copy()
    voxel_dims = np.array([np.linalg.norm(affine[:3, i]) for i in range(3)])
    scale = target_resolution / voxel_dims
    for i in range(3):
        new_affine[:3, i] = affine[:3, i] * scale[i]
    return new_affine


def _create_heatmap(data: np.ndarray, title: str, filename: str, cmap: str = 'RdBu_r', center_zero: bool = True) -> None:
    """Create a heatmap for an FC matrix with NaN shown as gray."""
    fig, ax = plt.subplots(figsize=(10, 8))
    current_cmap = plt.cm.get_cmap(cmap).copy()
    current_cmap.set_bad(color='gray')

    if center_zero:
        vmax = np.nanmax(np.abs(data))
        vmin = -vmax
    else:
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)

    im = ax.imshow(data, cmap=current_cmap, aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, label='Fisher Z', ax=ax)

    ax.set_title(title)
    ax.set_xlabel('Voxel Index')
    ax.set_ylabel('Voxel Index')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def run_participants(
    participant_ids: List[str],
    treatment_condition: str,
    control_condition: str,
    data_dir: str,
    global_mask_file: Optional[str] = None,
    target_resolution: float = config.DEFAULT_TARGET_RESOLUTION,
    apply_censoring: bool = True,
    debug: bool = False,
    allow_mask_resample: bool = False,
    allow_no_mask: bool = False,
    output_dir: Optional[str] = None,
    save_outputs: bool = True,
    save_heatmaps: bool = True,
    save_niftis: bool = True,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Run participant-level FC processing for multiple participants.
    
    Args:
        participant_ids: List of participant IDs to process.
        treatment_condition: Treatment condition label.
        control_condition: Control condition label.
        data_dir: Directory containing participant data.
        global_mask_file: Path to global mask file, or None to create from individual masks.
        target_resolution: Target voxel resolution in mm.
        apply_censoring: Whether to apply censoring.
        debug: Enable debug output.
        allow_mask_resample: Allow resampling individual masks to global mask space.
        allow_no_mask: Allow processing without a mask.
        output_dir: Directory for outputs (defaults to data_dir).
        save_outputs: Save .npy matrices.
        save_heatmaps: Save heatmap visualizations.
        save_niftis: Save NIfTI outputs.
    
    Returns:
        Dictionary mapping participant_id to (treatment_fc, control_fc, fc_change) tuples.
    """
    if output_dir is None:
        output_dir = data_dir
    os.makedirs(output_dir, exist_ok=True)

    # Prepare global mask and affine once for all participants
    mask_affine = None
    global_mask = None

    if global_mask_file:
        print(f"Loading global mask from file: {global_mask_file}")
        if not os.path.exists(global_mask_file):
            raise FileNotFoundError(f"Global mask file not found: {global_mask_file}")
        global_mask = downsample_mask(global_mask_file, target_resolution)
        cached_mask_path = _cached_downsample_path(global_mask_file, target_resolution)
        if os.path.exists(cached_mask_path):
            mask_affine = nib.load(cached_mask_path).affine
        else:
            mask_affine = _rescale_affine_to_resolution(nib.load(global_mask_file).affine, target_resolution)
        print(f"  Global mask loaded: {global_mask.shape}, {int(np.sum(global_mask > 0))} voxels\n")
    else:
        print("No global mask provided; will create from individual masks if available\n")

    # Process each participant
    results = {}
    for participant_id in participant_ids:
        print(f"Processing participant: {participant_id}")
        print("-" * 80)
        
        # Run participant processing
        treatment_fc, control_fc, fc_change = participant_fc.process_participant(
            participant_id=participant_id,
            data_dir=data_dir,
            treatment_condition=treatment_condition,
            control_condition=control_condition,
            global_mask=global_mask,
            target_resolution=target_resolution,
            apply_censoring=apply_censoring,
            debug=debug,
            mask_affine=mask_affine,
            global_mask_file=global_mask_file,
            allow_mask_resample=allow_mask_resample,
            allow_no_mask=allow_no_mask,
        )
        
        results[participant_id] = (treatment_fc, control_fc, fc_change)
        
        if not save_outputs:
            continue

        # Save matrices as .npy
        np.save(os.path.join(output_dir, f"results_{participant_id}_treatment_fc.npy"), treatment_fc)
        np.save(os.path.join(output_dir, f"results_{participant_id}_control_fc.npy"), control_fc)
        np.save(os.path.join(output_dir, f"results_{participant_id}_fc_change.npy"), fc_change)

        if save_heatmaps:
            _create_heatmap(
                treatment_fc,
                f"Treatment FC Matrix\nParticipant {participant_id}",
                os.path.join(output_dir, f"results_{participant_id}_treatment_fc.png"),
                cmap='viridis',
                center_zero=False,
            )
            _create_heatmap(
                control_fc,
                f"Control FC Matrix\nParticipant {participant_id}",
                os.path.join(output_dir, f"results_{participant_id}_control_fc.png"),
                cmap='viridis',
                center_zero=False,
            )
            _create_heatmap(
                fc_change,
                f"FC Change (Treatment - Control)\nParticipant {participant_id}",
                os.path.join(output_dir, f"results_{participant_id}_fc_change.png"),
                cmap='RdBu_r',
                center_zero=True,
            )

        if save_niftis:
            if global_mask is None:
                print("  WARNING: No global mask available; skipping NIfTI outputs")
            else:
                fc_matrices_file = os.path.join(output_dir, f"results_{participant_id}_fc_matrices_4d.nii")
                participant_fc.save_fc_matrices_as_nifti(
                    treatment_fc=treatment_fc,
                    control_fc=control_fc,
                    fc_change=fc_change,
                    global_mask=global_mask,
                    output_file=fc_matrices_file,
                    target_resolution=target_resolution,
                    mask_affine=mask_affine,
                )

                mean_fc_map_file = os.path.join(output_dir, f"results_{participant_id}_mean_fc_change.nii")
                participant_fc.create_mean_fc_change_map(
                    fc_change=fc_change,
                    global_mask=global_mask,
                    output_file=mean_fc_map_file,
                    target_resolution=target_resolution,
                    mask_affine=mask_affine,
                )
        
        print(f"✓ Participant {participant_id} complete\n")

    return results


# Placeholders for future higher-level orchestration
def run_analysis(config_file: str = None) -> None:
    raise NotImplementedError("Full pipeline orchestration not yet implemented.")


def validate_and_load_config(config_file: str = None) -> Dict:
    raise NotImplementedError("Config validation/loading not yet implemented.")


def process_all_participants(analysis_config: Dict) -> Dict[str, any]:
    raise NotImplementedError("Batch participant processing not yet implemented.")


def execute_group_analysis(participant_results: Dict[str, any], 
                           analysis_config: Dict,
                           output_dir: str) -> Dict:
    raise NotImplementedError("Group analysis not yet implemented.")


def generate_summary_report(results: Dict, output_file: str) -> None:
    raise NotImplementedError("Summary reporting not yet implemented.")
