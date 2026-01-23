"""
Main execution script for whole-brain inter-subject functional connectivity analysis.

Orchestrates the complete pipeline:
1. Configuration reading
2. Participant-level FC change calculations
3. Inter-subject similarity analysis
4. Group-level statistical analysis
"""

from typing import Dict, List
from . import config
from . import participant_fc
from . import inter_subject


def run_analysis(config_file: str = None) -> None:
    """
    Execute the complete whole-brain ISFC analysis pipeline.
    
    Parameters
    ----------
    config_file : str, optional
        Path to configuration file. If not provided, prompts user for input.
    """
    pass


def validate_and_load_config(config_file: str = None) -> Dict:
    """
    Load and validate configuration, prompting user if needed.
    
    Parameters
    ----------
    config_file : str, optional
        Path to configuration file
    
    Returns
    -------
    dict
        Validated configuration dictionary
    """
    pass


def process_all_participants(analysis_config: Dict) -> Dict[str, any]:
    """
    Process all participants' data.
    
    Parameters
    ----------
    analysis_config : dict
        Configuration dictionary containing data paths and parameters
    
    Returns
    -------
    dict
        Results dictionary containing FC change matrices for all participants
    """
    pass


def execute_group_analysis(participant_results: Dict[str, any], 
                           analysis_config: Dict,
                           output_dir: str) -> Dict:
    """
    Execute group-level inter-subject analysis.
    
    Parameters
    ----------
    participant_results : dict
        Results from participant-level processing
    analysis_config : dict
        Configuration dictionary
    output_dir : str
        Directory for output files
    
    Returns
    -------
    dict
        Group analysis results
    """
    pass


def generate_summary_report(results: Dict, output_file: str) -> None:
    """
    Generate summary report of the analysis.
    
    Parameters
    ----------
    results : dict
        Complete analysis results
    output_file : str
        Path where summary report should be saved
    """
    pass
