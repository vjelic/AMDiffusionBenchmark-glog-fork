import logging
import subprocess
import warnings

from accelerate.logging import get_logger
from distutils.util import strtobool


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
    Returns "cuda" if NVIDIA's `nvidia-smi` is available, "rocm" if AMD's `rocm-smi` is available.
    If neither `nvidia-smi` nor `rocm-smi` is available on the system, raises a RuntimeError.
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

    raise RuntimeError("Neither NVIDIA nor AMD GPU tools are available.")
