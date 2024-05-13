import logging
from pathlib import Path
logger = logging.getLogger(__name__)


def cfg_mean_nsamples_resolution(cfg):
    """
    Resolves the number of samples based on the configuration.

    Args:
        cfg: The configuration object containing the parameters 'mean' and 'number_of_samples'.

    Returns:
        bool: True if the number of samples is 1, False otherwise.

    Side Effects:
        If 'mean' is True and 'number_of_samples' is more than 1, it logs an error and sets 'number_of_samples' to 1.
    """
    if cfg.mean and cfg.number_of_samples > 1:
        logger.error(
            "All the samples will be the mean.. cfg.number_of_samples=1 will be forced.")
        cfg.number_of_samples = 1

    return cfg.number_of_samples == 1


def get_path(sample_path: Path, is_amass: bool, gender: str, split: str, onesample: bool, mean: bool, fact: float):
    """
    Constructs a path based on the provided parameters.

    Args:
        sample_path (Path): The base path for the sample.
        is_amass (bool): A flag indicating whether the sample is from AMASS.
        gender (str): The gender of the sample.
        split (str): The split of the sample (e.g., 'train', 'test').
        onesample (bool): A flag indicating whether there is only one sample.
        mean (bool): A flag indicating whether the sample is a mean sample.
        fact (float): A factor to be included in the path.

    Returns:
        path (Path): The constructed path.
    """
    extra_str = ("_mean" if mean else "") if onesample else "_multi"
    fact_str = "" if fact == 1 else f"{fact}_"
    gender_str = gender + "_" if is_amass else ""
    path = sample_path / f"{fact_str}{gender_str}{split}{extra_str}"
    return path
