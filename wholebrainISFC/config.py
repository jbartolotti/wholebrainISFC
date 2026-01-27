"""
Configuration module for reading and managing user inputs and analysis parameters.
"""

from typing import Dict, List, Tuple

# Default analysis settings
DEFAULT_TARGET_RESOLUTION = 6.0  # mm
DEFAULT_ALLOW_MASK_RESAMPLE = False
DEFAULT_ALLOW_NO_MASK = False


def read_user_inputs() -> Dict:
    """
    Read primary inputs from the user interactively.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'data_dir': str, directory containing brain nifti data
        - 'gm_mask': str, path to grey matter mask file
        - 'treatment_condition': str, name of treatment condition
        - 'control_condition': str, name of control condition
    """
    pass


def read_covariates(covariate_file: str) -> Tuple[Dict[str, List[float]], List[str]]:
    """
    Read participant covariate values and covariate names.
    
    Parameters
    ----------
    covariate_file : str
        Path to file containing covariate data
    
    Returns
    -------
    tuple
        (covariate_dict, covariate_names) where:
        - covariate_dict: Dict with participant IDs as keys, covariate values as lists
        - covariate_names: List of covariate names in order
    """
    pass


def load_configuration(config_file: str) -> Dict:
    """
    Load configuration from a file.
    
    Parameters
    ----------
    config_file : str
        Path to configuration file (JSON or YAML format)
    
    Returns
    -------
    dict
        Configuration dictionary
    """
    pass


def save_configuration(config: Dict, config_file: str) -> None:
    """
    Save configuration to a file for reproducibility.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    config_file : str
        Path where configuration should be saved
    """
    pass


def validate_configuration(config: Dict) -> bool:
    """
    Validate that all required configuration parameters are present and valid.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    
    Returns
    -------
    bool
        True if configuration is valid, raises exception otherwise
    """
    pass
