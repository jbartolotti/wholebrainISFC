# wholebrainISFC

A Python library for preprocessing and analyzing whole-brain intrinsic functional connectivity (ISFC) fMRI data.

## Features

- Read and process NIfTI files
- Downsample fMRI data
- Apply brain masks
- Compute cross-correlation matrices
- AFNI command integration

## Installation

Install from GitHub:

```bash
pip install git+https://github.com/yourusername/wholebrainISFC.git
```

Or for development:

```bash
git clone https://github.com/yourusername/wholebrainISFC.git
cd wholebrainISFC
pip install -e .
```

## Quick Start

```python
from wholebrainISFC import nifti_utils

# Read a nifti file and get the mean value
mean_val = nifti_utils.get_nifti_mean('path/to/your/file.nii.gz')
print(f"Mean voxel value: {mean_val}")
```

## Dependencies

- nibabel - for reading/writing NIfTI files
- numpy - for numerical operations
- scipy - for scientific computing

## License

MIT
