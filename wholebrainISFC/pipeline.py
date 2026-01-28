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


def _pick_first_existing(candidates: List[str]) -> Optional[str]:
    """Return first existing path from candidates."""
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def resolve_bids_inputs(
    bids_dir: str,
    participant_id: str,
    treatment_label: str,
    control_label: str,
    session: Optional[str] = None,
    space: str = "MNI152NLin2009cAsym",
    res_label: Optional[str] = None,
) -> Dict[str, str]:
    """Resolve BIDS-formatted inputs for a participant.

    Looks for preproc BOLD, GM masks, and censor files using standard BIDS entities.
    """

    if res_label is None:
        res_label = "6"

    sub_prefix = f"sub-{participant_id}"
    ses_component = f"ses-{session}" if session else None
    stem_base = sub_prefix
    if ses_component:
        stem_base += f"_{ses_component}"

    def _with_ext(base: str, exts: List[str]) -> str:
        candidates = [base + ext for ext in exts]
        found = _pick_first_existing(candidates)
        if found is None:
            raise FileNotFoundError(
                f"Missing file; tried: {', '.join(candidates)}"
            )
        return found

    func_dir = os.path.join(bids_dir, sub_prefix, ses_component, "func") if ses_component else os.path.join(bids_dir, sub_prefix, "func")

    treat_stem = f"{stem_base}_task-{treatment_label}_space-{space}_desc-preproc_bold"
    ctrl_stem = f"{stem_base}_task-{control_label}_space-{space}_desc-preproc_bold"
    treatment_file = _with_ext(os.path.join(func_dir, treat_stem), [".nii.gz", ".nii"])
    control_file = _with_ext(os.path.join(func_dir, ctrl_stem), [".nii.gz", ".nii"])

    # Masks (optional; prefer target resolution, then fall back to any resolution and auto-downsample)
    mask_suffix = f"_space-{space}_desc-GM_mask"
    treatment_mask_file = _pick_first_existing([
        os.path.join(func_dir, f"{stem_base}_task-{treatment_label}_space-{space}_res-{res_label}_desc-GM_mask.nii.gz"),
        os.path.join(func_dir, f"{stem_base}_task-{treatment_label}_space-{space}_res-{res_label}_desc-GM_mask.nii"),
    ])
    if not treatment_mask_file:
        treatment_mask_file = _pick_first_existing([
            os.path.join(func_dir, f"{stem_base}_task-{treatment_label}{mask_suffix}.nii.gz"),
            os.path.join(func_dir, f"{stem_base}_task-{treatment_label}{mask_suffix}.nii"),
        ])
    
    control_mask_file = _pick_first_existing([
        os.path.join(func_dir, f"{stem_base}_task-{control_label}_space-{space}_res-{res_label}_desc-GM_mask.nii.gz"),
        os.path.join(func_dir, f"{stem_base}_task-{control_label}_space-{space}_res-{res_label}_desc-GM_mask.nii"),
    ])
    if not control_mask_file:
        control_mask_file = _pick_first_existing([
            os.path.join(func_dir, f"{stem_base}_task-{control_label}{mask_suffix}.nii.gz"),
            os.path.join(func_dir, f"{stem_base}_task-{control_label}{mask_suffix}.nii"),
        ])

    # Censor (TSV/CSV/1D)
    censor_stem_t = os.path.join(func_dir, f"{stem_base}_task-{treatment_label}_desc-censor_timeseries")
    censor_stem_c = os.path.join(func_dir, f"{stem_base}_task-{control_label}_desc-censor_timeseries")
    treatment_censor_file = _with_ext(censor_stem_t, [".tsv", ".csv", ".txt", ".1D"])
    control_censor_file = _with_ext(censor_stem_c, [".tsv", ".csv", ".txt", ".1D"])

    return {
        "treatment_file": treatment_file,
        "control_file": control_file,
        "treatment_mask_file": treatment_mask_file,
        "control_mask_file": control_mask_file,
        "treatment_censor_file": treatment_censor_file,
        "control_censor_file": control_censor_file,
        "func_dir": func_dir,
    }


def run_participants(
    participant_ids: List[str],
    treatment_label: str,
    control_label: str,
    data_dir: str,
    bids_dir: str,
    use_bids: bool = False,
    session: Optional[str] = None,
    space: str = "MNI152NLin2009cAsym",
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
    force_overwrite: bool = False,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Run participant-level FC processing for multiple participants.
    
    Args:
        participant_ids: List of participant IDs to process.
        treatment_label: Condition/task label for treatment arm.
        control_label: Condition/task label for control arm.
        data_dir: Directory containing participant data (ad-hoc mode).
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
        force_overwrite: If False, skip participant processing when all expected outputs already exist.
    
    Returns:
        Dictionary mapping participant_id to (treatment_fc, control_fc, fc_change) tuples.
    """
    if bids_dir is None:
        raise ValueError("bids_dir is required to place derivatives")

    deriv_root = os.path.join(bids_dir, "derivatives", "wholebrainISFC")
    if output_dir is None:
        output_dir = deriv_root
    os.makedirs(output_dir, exist_ok=True)

    # Prepare global mask and affine once for all participants
    mask_affine = None
    global_mask = None

    # If not provided and using BIDS, try default derivatives location
    # (prefer target resolution; fall back to any resolution and downsample as needed)
    if global_mask_file is None and use_bids:
        default_mask = _pick_first_existing([
            os.path.join(bids_dir, "derivatives", "wholebrainISFC", "masks",
                         f"global_space-{space}_res-{int(target_resolution)}_desc-union_mask.nii.gz"),
            os.path.join(bids_dir, "derivatives", "wholebrainISFC", "masks",
                         f"global_space-{space}_res-{int(target_resolution)}_desc-union_mask.nii"),
        ])
        if not default_mask:
            default_mask = _pick_first_existing([
                os.path.join(bids_dir, "derivatives", "wholebrainISFC", "masks",
                             f"global_space-{space}_desc-union_mask.nii.gz"),
                os.path.join(bids_dir, "derivatives", "wholebrainISFC", "masks",
                             f"global_space-{space}_desc-union_mask.nii"),
            ])
        if default_mask:
            global_mask_file = default_mask

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

        treat_task = treatment_label
        ctrl_task = control_label

        resolved = {}
        participant_data_dir = data_dir
        if use_bids:
            resolved = resolve_bids_inputs(
                bids_dir=bids_dir,
                participant_id=participant_id,
                treatment_label=treat_task,
                control_label=ctrl_task,
                session=session,
                space=space,
                res_label=str(int(target_resolution)),
            )
            participant_data_dir = resolved["func_dir"]

        # Build derivative paths (always under derivatives/wholebrainISFC)
        sub_dir_parts = [f"sub-{participant_id}"]
        if session:
            sub_dir_parts.append(f"ses-{session}")
        sub_dir_parts.append(f"res-{int(target_resolution)}")
        participant_out_dir = os.path.join(output_dir, *sub_dir_parts)
        figures_dir = os.path.join(participant_out_dir, "figures")
        os.makedirs(participant_out_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)

        stem_base = f"sub-{participant_id}"
        if session:
            stem_base += f"_ses-{session}"
        stem_base += f"_space-{space}_res-{int(target_resolution)}"

        # Expected outputs for skip logic
        required_files = []
        if save_outputs:
            required_files.extend([
                os.path.join(participant_out_dir, f"{stem_base}_task-{treat_task}_desc-FC_matrix.npy"),
                os.path.join(participant_out_dir, f"{stem_base}_task-{ctrl_task}_desc-FC_matrix.npy"),
                os.path.join(participant_out_dir, f"{stem_base}_desc-FCchange_matrix.npy"),
            ])
            if save_heatmaps:
                required_files.extend([
                    os.path.join(figures_dir, f"{stem_base}_task-{treat_task}_desc-FC_matrix.png"),
                    os.path.join(figures_dir, f"{stem_base}_task-{ctrl_task}_desc-FC_matrix.png"),
                    os.path.join(figures_dir, f"{stem_base}_desc-FCchange_matrix.png"),
                ])
            if save_niftis:
                required_files.extend([
                    os.path.join(participant_out_dir, f"{stem_base}_desc-FCstack_bold.nii"),
                    os.path.join(participant_out_dir, f"{stem_base}_desc-meanFCchange_map.nii"),
                ])

        if save_outputs and not force_overwrite and required_files and all(os.path.exists(f) for f in required_files):
            print(f"Outputs exist for participant {participant_id}; skipping (set force_overwrite=True to recompute)")
            # Load matrices to keep downstream consistency
            treatment_fc = np.load(os.path.join(participant_out_dir, f"{stem_base}_task-{treat_task}_desc-FC_matrix.npy"))
            control_fc = np.load(os.path.join(participant_out_dir, f"{stem_base}_task-{ctrl_task}_desc-FC_matrix.npy"))
            fc_change = np.load(os.path.join(participant_out_dir, f"{stem_base}_desc-FCchange_matrix.npy"))
            results[participant_id] = (treatment_fc, control_fc, fc_change)
            continue

        treatment_fc, control_fc, fc_change = participant_fc.process_participant(
            participant_id=participant_id,
            data_dir=participant_data_dir,
            treatment_condition=treatment_label,
            control_condition=control_label,
            global_mask=global_mask,
            target_resolution=target_resolution,
            apply_censoring=apply_censoring,
            debug=debug,
            mask_affine=mask_affine,
            global_mask_file=global_mask_file,
            allow_mask_resample=allow_mask_resample,
            allow_no_mask=allow_no_mask,
            treatment_file=resolved.get("treatment_file"),
            control_file=resolved.get("control_file"),
            treatment_mask_file=resolved.get("treatment_mask_file"),
            control_mask_file=resolved.get("control_mask_file"),
            treatment_censor_file=resolved.get("treatment_censor_file"),
            control_censor_file=resolved.get("control_censor_file"),
        )

        results[participant_id] = (treatment_fc, control_fc, fc_change)

        if not save_outputs:
            continue

        # Save matrices as .npy in derivatives
        np.save(os.path.join(participant_out_dir, f"{stem_base}_task-{treat_task}_desc-FC_matrix.npy"), treatment_fc)
        np.save(os.path.join(participant_out_dir, f"{stem_base}_task-{ctrl_task}_desc-FC_matrix.npy"), control_fc)
        np.save(os.path.join(participant_out_dir, f"{stem_base}_desc-FCchange_matrix.npy"), fc_change)

        if save_heatmaps:
            _create_heatmap(
                treatment_fc,
                f"Treatment FC Matrix\nParticipant {participant_id}",
                os.path.join(figures_dir, f"{stem_base}_task-{treat_task}_desc-FC_matrix.png"),
                cmap='viridis',
                center_zero=False,
            )
            _create_heatmap(
                control_fc,
                f"Control FC Matrix\nParticipant {participant_id}",
                os.path.join(figures_dir, f"{stem_base}_task-{ctrl_task}_desc-FC_matrix.png"),
                cmap='viridis',
                center_zero=False,
            )
            _create_heatmap(
                fc_change,
                f"FC Change (Treatment - Control)\nParticipant {participant_id}",
                os.path.join(figures_dir, f"{stem_base}_desc-FCchange_matrix.png"),
                cmap='RdBu_r',
                center_zero=True,
            )

        if save_niftis:
            if global_mask is None:
                print("  WARNING: No global mask available; skipping NIfTI outputs")
            else:
                fc_matrices_file = os.path.join(participant_out_dir, f"{stem_base}_desc-FCstack_bold.nii")
                participant_fc.save_fc_matrices_as_nifti(
                    treatment_fc=treatment_fc,
                    control_fc=control_fc,
                    fc_change=fc_change,
                    global_mask=global_mask,
                    output_file=fc_matrices_file,
                    target_resolution=target_resolution,
                    mask_affine=mask_affine,
                )

                mean_fc_map_file = os.path.join(participant_out_dir, f"{stem_base}_desc-meanFCchange_map.nii")
                participant_fc.create_mean_fc_change_map(
                    fc_change=fc_change,
                    global_mask=global_mask,
                    output_file=mean_fc_map_file,
                    target_resolution=target_resolution,
                    mask_affine=mask_affine,
                )

        print(f"✓ Participant {participant_id} complete\n")

    return results


def run_group_inter_subject_analysis(
    participant_ids: List[str],
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    bids_dir: str,
    output_dir: Optional[str] = None,
    target_resolution: float = config.DEFAULT_TARGET_RESOLUTION,
    space: str = "MNI152NLin2009cAsym",
) -> Dict[str, str]:
    """
    Run group-level inter-subject similarity (ISS) analysis.

    Computes per-voxel correlations of FC-change across participants, derives
    per-participant mean and t-statistics, and saves to BIDS derivatives.

    Parameters
    ----------
    participant_ids : list of str
        Participant IDs that were processed.
    results_dict : dict
        Dictionary mapping participant_id to (treatment_fc, control_fc, fc_change) tuples
        from run_participants.
    bids_dir : str
        BIDS root directory where derivatives are stored.
    output_dir : str, optional
        Output directory for ISS maps. Defaults to derivatives/wholebrainISFC.
    target_resolution : float
        Target voxel resolution in mm.
    space : str
        Brain space name (default "MNI152NLin2009cAsym").

    Returns
    -------
    dict
        Dictionary mapping participant_id to output NIfTI file paths.
    """
    from wholebrainISFC import inter_subject

    if output_dir is None:
        output_dir = os.path.join(bids_dir, "derivatives", "wholebrainISFC")

    if len(participant_ids) < 3:
        print(f"WARNING: Skipping group ISS (need ≥3 participants, have {len(participant_ids)})")
        return {}

    # Extract FC-change matrices
    fc_change_matrices = {pid: results_dict[pid][2] for pid in participant_ids}

    # Find the global mask as reference (should be in derivatives/masks/)
    ref_nifti = _pick_first_existing([
        os.path.join(bids_dir, "derivatives", "wholebrainISFC", "masks",
                     f"global_space-{space}_res-{int(target_resolution)}_desc-union_mask_{int(target_resolution)}mm.nii.gz"),
        os.path.join(bids_dir, "derivatives", "wholebrainISFC", "masks",
                     f"global_space-{space}_res-{int(target_resolution)}_desc-union_mask_{int(target_resolution)}mm.nii"),
        os.path.join(bids_dir, "derivatives", "wholebrainISFC", "masks",
                     f"global_space-{space}_desc-union_mask.nii.gz"),
        os.path.join(bids_dir, "derivatives", "wholebrainISFC", "masks",
                     f"global_space-{space}_desc-union_mask.nii"),
    ])
    
    if not ref_nifti:
        print("WARNING: Global mask not found in derivatives/masks/")
        print("  ISS analysis requires a global mask; skipping.")
        return {}

    print("\nRunning group-level inter-subject similarity analysis...")
    print(f"  Participants: {participant_ids}")
    print(f"  Reference mask: {ref_nifti}")

    iss_output_dir = os.path.join(output_dir, "group", "inter_subject_similarity")
    iss_results = inter_subject.process_group_inter_subject_analysis(
        fc_change_matrices=fc_change_matrices,
        participant_ids=participant_ids,
        reference_nifti=ref_nifti,
        output_dir=iss_output_dir,
    )

    if iss_results:
        print(f"✓ ISS analysis complete ({len(iss_results)} participants)")
        for pid, nifti_path in iss_results.items():
            print(f"  {pid}: {nifti_path}")
    else:
        print("⚠ ISS analysis produced no output")

    return iss_results


def run_group_3dmema_analysis(
    iss_results: Dict[str, str],
    bids_dir: str,
    output_dir: Optional[str] = None,
    set_label: str = "ISS",
    covariate_names: Optional[List[str]] = None,
) -> str:
    """
    Run AFNI 3dMEMA mixed-effects meta-analysis on ISS results.

    Parameters
    ----------
    iss_results : dict
        Dictionary mapping participant_id to ISS NIfTI file path (from run_group_inter_subject_analysis).
    bids_dir : str
        BIDS root directory where derivatives are stored.
    output_dir : str, optional
        Output directory for 3dMEMA results. Defaults to derivatives/wholebrainISFC/group/3dMEMA.
    set_label : str
        Label for the analysis set (default "ISS").
    covariate_names : list of str, optional
        Names of covariates to include (must exist in participants.tsv and participants.json).

    Returns
    -------
    str
        Path to 3dMEMA output file.
    """
    from wholebrainISFC import inter_subject

    if output_dir is None:
        output_dir = os.path.join(bids_dir, "derivatives", "wholebrainISFC", "group", "3dMEMA")

    if len(iss_results) < 3:
        print(f"WARNING: Skipping 3dMEMA (need ≥3 participants, have {len(iss_results)})")
        return ""

    try:
        mema_output = inter_subject.run_3dmema_analysis(
            iss_results=iss_results,
            output_dir=output_dir,
            set_label=set_label,
            bids_dir=bids_dir,
            covariate_names=covariate_names,
        )
        print(f"✓ 3dMEMA analysis complete: {mema_output}")
        return mema_output
    except Exception as e:
        print(f"⚠ 3dMEMA analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return ""


# Placeholders for future higher-level orchestration
def run_analysis(config_file: str = None) -> None:
    raise NotImplementedError("Full pipeline orchestration not yet implemented.")


def validate_and_load_config(config_file: str = None) -> Dict:
    raise NotImplementedError("Config validation/loading not yet implemented.")


def process_all_participants(analysis_config: Dict) -> Dict[str, any]:
    raise NotImplementedError("Batch participant processing not yet implemented.")


def generate_summary_report(results: Dict, output_file: str) -> None:
    raise NotImplementedError("Summary reporting not yet implemented.")
