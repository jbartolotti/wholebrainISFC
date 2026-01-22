"""
Utility functions for reading and processing NIfTI files.
"""

import nibabel as nib
import numpy as np


def get_nifti_mean(filepath):
    """
    Read a NIfTI file and return the mean value of all voxels.
    
    Parameters
    ----------
    filepath : str
        Path to the NIfTI file (.nii or .nii.gz)
    
    Returns
    -------
    float
        Mean value of all voxels in the image
    
    Examples
    --------
    >>> mean_val = get_nifti_mean('path/to/image.nii.gz')
    >>> print(f"Mean voxel value: {mean_val}")
    """
    # Load the NIfTI file
    img = nib.load(filepath)
    
    # Get the image data
    data = img.get_fdata()
    
    # Calculate and return the mean
    mean_value = np.mean(data)
    
    return mean_value
