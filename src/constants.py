"""This module defines various constants used throughout the project."""

# Default metrics
DEFAULT_METRICS = {
    "total_elapsed_time": None,
    "avg_loss": None,
    "std_loss": None,
    "avg_loss_tail": None,
    "std_loss_tail": None,
    "avg_step_time_total": None,
    "std_step_time_total": None,
    "avg_step_time_gpu": None,
    "std_step_time_gpu": None,
    "avg_fps_gpu": None,
    "std_fps_gpu": None,
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
