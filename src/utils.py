import argparse
import logging
import subprocess
import warnings
from distutils.util import strtobool
from typing import Tuple, Union

from accelerate.logging import get_logger


def configure_logging():
    """Configures the logging to prevent multiple equivalent warnings across all the available processes

    Returns:
            Logger: Object for logging to terminal
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = get_logger(__name__)

    # Ignore some warnings to keep the training logs cleaner
    warnings.filterwarnings(
        "ignore", category=FutureWarning, message=".*autocast.*deprecated.*"
    )
    warnings.filterwarnings("ignore", message=".*Profiler function.*ignored.*")

    return logger


def safely_eval_as_bool(x: str):
    """Evaluate the input str safely, as a boolean. Accepts inputs 0, 1, true and false."""
    return bool(strtobool(x))


def check_gpu_vendor():
    """
    Determines the GPU vendor by checking the availability of GPU management tools.
    Returns "cuda" if `nvidia-smi` is available, "rocm" if `rocm-smi` is available.
    If neither of them is available on the system, raises a RuntimeError.
    """
    try:
        subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        return "cuda"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    try:
        subprocess.run(
            ["rocm-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        return "rocm"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    raise RuntimeError("Unsupported GPU vendor.")


def parse_resolution(resolution: Union[str, int]) -> Tuple[int, int]:
    # Check if the input is a single integer
    if isinstance(resolution, str) and resolution.isdigit():
        resolution = int(resolution)

    if isinstance(resolution, int):
        return (int(resolution),) * 2

    # Check if the input is in the form of "width,height"
    try:
        width, height = map(int, resolution.split(","))
        return (width, height)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Resolution must be an integer or a tuple of two integers in the form 'width,height'."
        )
