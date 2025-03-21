"""This module defines various constants used throughout the project."""

# Default metrics
DEFAULT_METRICS = {
    "avg_loss": None,
    "std_loss": None,
    "avg_loss_tail": None,
    "std_loss_tail": None,
    "avg_time": None,
    "std_time": None,
    "avg_fps": None,
    "std_fps": None,
    "avg_tflops": None,
    "std_tflops": None,
    "num_samples": 0,
}

METRIC_COLUMNS = list(DEFAULT_METRICS.keys())

METADATA_COLUMNS = [
    "run_id",
    "timestamp",
    "status",
    "git_hash",
    "git_user",
]

# Status values
STATUS_SUCCESS = "success"
STATUS_OOM = "oom"
STATUS_OOM_SKIPPED = "oom-skipped"
STATUS_ERROR = "error"
STATUS_NOTRUN = "not_run"

# Run parameter identifiers
ACCELERATE_CONFIG = "accelerate_config"
CLI_ARGS = "train_args"
RUN_ID = "run_id"
STATUS = "status"
TRAIN_BATCH_SIZE = "train_batch_size"
NUM_FRAMES = "num_frames"
