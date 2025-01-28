#!/usr/bin/env python3

"""
Log processing utilities for training runs.

Handles parsing log files, extracting metrics, and generating summary statistics.

Features:
- Parses training logs to extract key metrics
- Calculates averages and standard deviations
- Handles warmup period exclusion
- Detects OOM and error conditions
- Generates summary DataFrames
- Saves results in CSV format

Usage Example:
-------------
python process_logs.py outputs/run_outputs/sweep_001 --warmup 5

The script generates a runs_summary.csv with processed metrics and metadata.
"""

import argparse
import datetime
import logging
import os
import re
from distutils.util import strtobool
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from src.constants import (
    ACCELERATE_CONFIG,
    CLI_ARGS,
    DEFAULT_METRICS,
    RUN_ID,
    STATUS,
    STATUS_ERROR,
    STATUS_NOTRUN,
    STATUS_OOM,
    STATUS_SUCCESS,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_log_file(
    log_filename: str, warmup_steps: int, tail_steps: int = 10
) -> Dict[str, Any]:
    """Parses the log file to extract metrics and compute averages and standard deviations.

    Args:
        log_filename (str): Path to the log file.
        warmup_steps (int): Number of initial steps to exclude from averaging.
        tail_steps (int, optional): Number of last steps to include in tail averages. Defaults to 10.

    Returns:
        Dict[str, Any]: A dictionary containing averaged metrics, standard deviations, tail averages, and status.
    """
    step_losses = []
    step_times = []
    fps_values = []
    flops_values = []
    status = STATUS_SUCCESS

    # Regular expression patterns to extract metrics
    loss_pattern = re.compile(r"step_loss': ([0-9\.e+-]+)")
    time_pattern = re.compile(r"step_time': ([0-9\.e+-]+)")
    fps_pattern = re.compile(r"fps_gpu': ([0-9\.e+-]+)")
    flops_pattern = re.compile(r"tflops/s': ([0-9\.e+-]+)")
    oom_pattern = re.compile(
        r"out of memory|OutOfMemoryError|HSA_STATUS_ERROR_OUT_OF_RESOURCES"
    )

    try:
        with open(log_filename, "r") as f:
            log_content = f.read()
            # Check for OOM errors first
            if oom_pattern.search(log_content):
                status = STATUS_OOM
            elif "Traceback" in log_content or "ERROR" in log_content:
                status = STATUS_ERROR

            # Extract metrics
            for line in log_content.splitlines():
                if "INFO - Step" in line:
                    loss_match = loss_pattern.search(line)
                    if loss_match:
                        step_losses.append(float(loss_match.group(1)))

                    time_match = time_pattern.search(line)
                    if time_match:
                        step_times.append(float(time_match.group(1)))

                    fps_match = fps_pattern.search(line)
                    if fps_match:
                        fps_values.append(float(fps_match.group(1)))

                    flops_match = flops_pattern.search(line)
                    if fps_match:
                        flops_values.append(float(flops_match.group(1)))

        # Exclude warmup steps
        step_losses = step_losses[warmup_steps:]
        step_times = step_times[warmup_steps:]
        fps_values = fps_values[warmup_steps:]
        flops_values = flops_values[warmup_steps:]

        # Calculate tail metrics
        tail_losses = step_losses[-tail_steps:] if step_losses else []

        # Calculate statistics
        metrics = {
            **DEFAULT_METRICS,
            "avg_loss": np.mean(step_losses) if step_losses else None,
            "std_loss": np.std(step_losses) if step_losses else None,
            "avg_loss_tail": np.mean(tail_losses) if tail_losses else None,
            "std_loss_tail": np.std(tail_losses) if tail_losses else None,
            "avg_time": np.mean(step_times) if step_times else None,
            "std_time": np.std(step_times) if step_times else None,
            "avg_fps": np.mean(fps_values) if fps_values else None,
            "std_fps": np.std(fps_values) if fps_values else None,
            "avg_tflops": np.mean(flops_values) if flops_values else None,
            "std_tflops": np.std(flops_values) if flops_values else None,
            "num_samples": len(step_losses),
            STATUS: status,
        }

        return metrics

    except Exception as e:
        logging.error(f"Error parsing log file {log_filename}: {str(e)}")
        return {
            **DEFAULT_METRICS,
            STATUS: STATUS_ERROR,
        }


def save_dataframe(df: pd.DataFrame, output_dir: str) -> None:
    """Saves the DataFrame to CSV with consistent formatting.

    Args:
        df (pd.DataFrame): DataFrame to save
        output_dir (str): Directory where to save the CSV
    """

    if bool(strtobool(os.environ.get("DISABLE_RUNS_SUMMARY", "0"))):
        logging.info("DISABLE_RUNS_SUMMARY=1 --> Not saving runs_summary.csv.")
        return

    if df.empty:
        logging.info("DataFrame is empty. No file will be saved.")
        return

    dataframe_file = os.path.join(output_dir, "runs_summary.csv")
    df.to_csv(
        dataframe_file,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    logging.info(f"DataFrame saved to {dataframe_file}")


def generate_dataframe(
    output_dir: str,
    warmup_steps: int,
    git_info: Dict[str, Any],
    update_df: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Parses all available logs in the output directory and generates a DataFrame.

    Args:
        output_dir (str): Directory where logs and configs are saved.
        warmup_steps (int): Number of initial steps to exclude from average computations.
        git_info (Dict[str, Any]): Git information.
        update_df (bool): If True, update existing DataFrame if present.

    Returns:
        Optional[pd.DataFrame]: A DataFrame containing run information and metrics.
    """
    existing_df = load_existing_dataframe(output_dir) if update_df else None
    run_ids = get_run_ids(output_dir)

    if not run_ids:
        logging.info("No configuration files found. No DataFrame created.")
        return existing_df if existing_df is not None else pd.DataFrame()

    to_update_run_ids = get_runs_to_update(existing_df, run_ids)

    if not to_update_run_ids:
        return existing_df

    data_list = [
        process_run(run_id, output_dir, warmup_steps, git_info)
        for run_id in to_update_run_ids
    ]

    new_df = pd.DataFrame(data_list)
    df = combine_dataframes(existing_df, new_df)

    if not df.empty:
        save_dataframe(df, output_dir)

    return df


def load_existing_dataframe(output_dir: str) -> Optional[pd.DataFrame]:
    existing_df_path = os.path.join(output_dir, "runs_summary.csv")
    if os.path.exists(existing_df_path):
        try:
            return pd.read_csv(existing_df_path)
        except Exception as e:
            logging.error(f"Error reading existing DataFrame: {e}")


def get_run_ids(output_dir: str) -> List[int]:
    """Helper function to get run IDs from configuration files."""
    try:
        config_files = [
            f for f in os.listdir(output_dir) if re.match(r"\d+_config\.yaml", f)
        ]
        return sorted(
            int(re.match(r"(\d+)_config\.yaml", f).group(1)) for f in config_files
        )
    except Exception as e:
        logging.error(f"Error listing files in directory {output_dir}: {e}")
        return []


def get_runs_to_update(
    existing_df: Optional[pd.DataFrame], run_ids: List[int]
) -> List[int]:
    if existing_df is not None:
        existing_run_status = existing_df.set_index(RUN_ID)[STATUS]
        return [
            run_id
            for run_id in run_ids
            if run_id not in existing_run_status.index
            or existing_run_status.loc[run_id] in [STATUS_NOTRUN, STATUS_ERROR]
        ]
    return run_ids


def process_run(
    run_id: int,
    output_dir: str,
    warmup_steps: int,
    git_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Process a run from its config and log files.

    This function reads a run's configuration and log files from disk and returns a dictionary containing
    the run's parameters, metrics, and metadata.

    Args:
        run_id (int): Unique identifier for the run
        output_dir (str): Directory containing the run's files
        warmup_steps (int): Number of warmup steps to exclude from metric calculations
        git_info (Dict[str, Any]): Git repository information at the time of the run

    Returns:
        Dict[str, Any]: Dictionary containing:
            - run_id: The provided run ID
            - timestamp: Timestamp when the log file was last modified (None if not run)
            - accelerate configuration parameters
            - CLI arguments used for the run
            - parsed metrics from log file (or default metrics if not run)
            - git repository information
    """
    config_filename = os.path.join(output_dir, f"{run_id}_config.yaml")
    with open(config_filename, "r") as f:
        config = yaml.safe_load(f)
    accelerate_config_params = config.get(ACCELERATE_CONFIG, {})
    cli_args_params = config.get(CLI_ARGS, {})

    log_filename = os.path.join(output_dir, f"{run_id}_logs.txt")
    if os.path.exists(log_filename):
        metrics = parse_log_file(log_filename, warmup_steps)
        log_mtime = os.path.getmtime(log_filename)
        timestamp = datetime.datetime.fromtimestamp(log_mtime)
    else:
        metrics = {**DEFAULT_METRICS, STATUS: STATUS_NOTRUN}
        timestamp = None

    return {
        RUN_ID: run_id,
        "timestamp": timestamp,
        **accelerate_config_params,
        **cli_args_params,
        **metrics,
        **git_info,
    }


def combine_dataframes(
    existing_df: Optional[pd.DataFrame], new_df: pd.DataFrame
) -> pd.DataFrame:
    if existing_df is not None:
        existing_df = existing_df[~existing_df[RUN_ID].isin(new_df[RUN_ID])]
        df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        df = new_df
    return df.sort_values(RUN_ID).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments with values from environment variables or defaults.
    """
    parser = argparse.ArgumentParser(
        description="Process log files and generate summary dataframe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.getenv("LOG_PROCESSOR_OUTPUT_DIR", "outputs/runs"),
        help="Directory containing log files to process",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=int(os.getenv("LOG_PROCESSOR_WARMUP_STEPS", "5")),
        help="Number of initial steps to exclude from calculations",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        raise ValueError(f"Directory not found: {args.output_dir}")

    return args


def print_summary_statistics(df: pd.DataFrame) -> None:
    """Print summary statistics for the processed runs.

    Args:
        df (pd.DataFrame): DataFrame containing run results
    """
    if df.empty:
        logging.warning("No runs to summarize. DataFrame is empty.")
        return
    logging.info(f"Total runs processed: {len(df)}")
    # Status distribution with aligned counts
    status_counts = df[STATUS].value_counts()
    max_status_len = max(len(str(status)) for status in status_counts.index)
    logging.info("Status distribution:")
    for status, count in status_counts.items():
        logging.info(f"  {str(status):<{max_status_len}} : {count:>4d}")

    # Configure pandas display options for better readability
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.precision", 4)
    pd.set_option("display.expand_frame_repr", False)

    # Only include columns that have more than one unique value
    display_cols = [col for col in df.columns if df[col].nunique() > 1]

    logging.info("DataFrame Summary:")
    # Use to_string() for better formatting control
    logging.info(
        "\n"
        + df[display_cols].to_string(
            index=False, justify="left", col_space=12, na_rep="N/A"
        )
    )


def main() -> None:
    """Process logs and display summary statistics.

    Generates a summary DataFrame from log files and prints statistics about the runs.
    Handles empty results and failed runs appropriately.
    """
    args = parse_args()

    git_info = {"git_commit": "unknown", "git_branch": "unknown"}
    logging.info("Generating summary dataframe...")

    df = generate_dataframe(args.output_dir, args.warmup, git_info)

    if df.empty:
        logging.warning("No data found to process")
        return

    print_summary_statistics(df)


if __name__ == "__main__":
    main()
