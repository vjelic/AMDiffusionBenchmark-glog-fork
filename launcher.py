#!/usr/bin/env python3

"""
Launcher script for training configurations.

Automates the management and execution of multiple training runs with different configurations.

Features:
- Reads parameter configurations from YAML files
- Generates parameter combinations
- Manages distributed training configurations
- Tracks metrics and metadata
- Supports failure recovery
- Provides logging and analysis

Usage Example:
-------------
python launcher.py \
    --config_file config/flux_mini_benchmark.yaml \
    --output_dir outputs/runs/sweep_000 \
    --dry-run

The script generates a runs_summary.csv with metrics and metadata for analysis.
"""
import datetime
import itertools
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Iterable, List

import git
import hydra
import pandas as pd
import yaml
from dotenv import load_dotenv
from flatten_dict import flatten, unflatten
from omegaconf import DictConfig, OmegaConf

from src.constants import (
    ACCELERATE_CONFIG,
    CLI_ARGS,
    DEFAULT_METRICS,
    METADATA_COLUMNS,
    METRIC_COLUMNS,
    NUM_FRAMES,
    RUN_ID,
    STATUS,
    STATUS_OOM,
    STATUS_OOM_SKIPPED,
    STATUS_SUCCESS,
    TRAIN_BATCH_SIZE,
)
from utility_scripts.process_logs import (
    generate_dataframe,
    print_summary_statistics,
    save_dataframe,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load environment variables from a .env file
load_dotenv()


def generate_combinations(params: DictConfig) -> List[Dict[str, Any]]:
    """Generates all combinations of parameters.

    Args:
        params (Dict[str, Any]): Dictionary of parameters.

    Returns:
        List[Dict[str, Any]]: List of parameter combinations.
    """
    if params:
        # flatten the config e.g. {dynamo_config: {dynamo_mode: default}} -> {dynamo_config.dynamo_mode: default}
        flat_params = flatten(params, reducer="dot")
        # Split params dict into those iterable values and those with single values
        iterable_valued_dict: Dict[str, Any] = {}
        single_valued_dict: Dict[str, Any] = {}
        for key, value in flat_params.items():
            if isinstance(value, Iterable) and not isinstance(value, str) and value:
                iterable_valued_dict[key] = value
            else:
                single_valued_dict[key] = value

        # No combinations to generate
        if not iterable_valued_dict:
            return [unflatten(flat_params, splitter="dot")]

        # Generate all combinations of the list values
        keys = list(iterable_valued_dict.keys())
        product_values = itertools.product(*iterable_valued_dict.values())

        # Create a list of dictionaries for each combination
        combinations = []
        for combination in product_values:
            # Create a dictionary for the current combination
            current_combination = {**dict(zip(keys, combination)), **single_valued_dict}
            combinations.append(unflatten(current_combination, splitter="dot"))

        return combinations

    return [{}]


def build_command(
    accelerate_config_path: str, train_args_params: Dict[str, Any]
) -> List[str]:
    """Builds the command to execute the training script.

    Args:
        accelerate_config_path (str): Path to the Accelerate config file.
        train_args_params (Dict[str, Any]): Dictionary of CLI arguments.

    Returns:
        List[str]: The command to be executed as a list of strings.
    """
    cmd = [
        "accelerate",
        "launch",
        "--config_file",
        accelerate_config_path,
        "train.py",
    ]
    for param, value in train_args_params.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{param}")
        else:
            cmd.extend([f"--{param}", str(value)])
    return cmd


def save_configuration(
    config_filename: str,
    accelerate_config_params: Dict[str, Any],
    train_args_params: Dict[str, Any],
) -> None:
    """Saves the combined configuration to a YAML file.

    Args:
        config_filename (str): Path to the configuration file.
        accelerate_config_params (Dict[str, Any]): Accelerate configuration parameters.
        train_args_params (Dict[str, Any]): CLI arguments.
    """
    combined_config = {
        ACCELERATE_CONFIG: accelerate_config_params,
        CLI_ARGS: train_args_params,
    }
    with open(config_filename, "w") as f:
        yaml.dump(
            combined_config,
            f,
        )


def should_skip_already_done(dataframe: pd.DataFrame, run_id: int) -> bool:
    """Determines whether a run should be skipped based on the existing DataFrame (success/oom).

    Args:
        dataframe (pd.DataFrame): The existing DataFrame of runs.
        run_id (int): The run identifier.

    Returns:
        bool: True if the run should be skipped due to success/oom and resume, False otherwise.
    """
    if (
        dataframe is not None
        and not dataframe.empty
        and RUN_ID in dataframe.columns
        and run_id in dataframe[RUN_ID].values
    ):
        run_status = dataframe.loc[dataframe[RUN_ID] == run_id, STATUS].values[0]
        # Add oom-skipped to skipped statuses
        if run_status in [STATUS_SUCCESS, STATUS_OOM]:
            logging.info(
                f"{run_status.upper()}: Skipping run {run_id} with status '{run_status}' due to resume flag."
            )
            return True
    return False


def get_git_info() -> Dict[str, Any]:
    """Retrieves Git repository information.

    Returns:
        Dict[str, Any]: A dictionary containing Git hash and user name.
    """
    try:
        repo = git.Repo(search_parent_directories=True)
        git_hash = repo.head.object.hexsha

        config_reader = repo.config_reader()
        if config_reader.has_section("user") and config_reader.has_option(
            "user", "name"
        ):
            git_user = config_reader.get_value("user", "name")
        else:
            git_user = None

    except Exception as e:
        logging.warning(f"Could not retrieve Git info: {e}")
        git_hash = None
        git_user = None

    return {"git_hash": git_hash, "git_user": git_user}


def run_training(
    cmd: List[str], log_filename: str, idx: int, dry_run: bool = False
) -> None:
    """Runs the training script and captures the output.

    Args:
        cmd (List[str]): The command to execute.
        log_filename (str): Path to the log file.
        idx (int): Run index for logging purposes.
        dry_run (bool): If True, perform a dry run without executing the command.

    This function executes the training command and logs the output.
    In dry-run mode, it only logs the command that would be executed.
    Any exceptions during execution are logged.
    """
    if dry_run:
        logging.info(f"DRY RUN:  command for run {idx}:")
        logging.info(" ".join(cmd))
        logging.info(f"Log would be saved to: {log_filename}")
    else:
        logging.info(f"Executing command for run {idx}: {' '.join(cmd)}")
        with open(log_filename, "w") as log_file:
            try:
                # Use Popen to capture output in real-time for both console and file.
                # subprocess.run is blocking and returns after subprocess exits.
                # Thus cannot capture and process output in real-time.
                # subprocess.Popen returns immediately after starting the process.
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,  # buffer 1 line so it can be redirected immediately
                )

                # redirect output in real-time
                for line in process.stdout:  # 1 line buffered => only 1 line to read
                    sys.stdout.write(line)  # Print to console
                    log_file.write(line)  # Write to file
                    log_file.flush()  # Ensure immediate write

                process.wait()  # Wait for the process to finish
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, cmd)

                logging.info(f"Run {idx} completed successfully!")
            except subprocess.CalledProcessError as e:
                logging.error(
                    f"Run {idx} failed with return code {e.returncode}. Check {log_filename} for details."
                )
            except Exception as e:
                logging.error(f"Run {idx} encountered an unexpected error: {e}")
        # Wait before next run
        time.sleep(3)


def should_skip_obvious_ooms(
    df: pd.DataFrame,
    run_id: int,
    target_feature: str,
    accelerate_config_params: Dict[str, Any],
    train_args_params: Dict[str, Any],
    output_dir: str,
    git_info: Dict[str, Any],
) -> bool:
    """Determines whether to skip a run if a previous smaller configuration resulted in OOM.

    This logic checks if there's a configuration identical in all parameters except one feature
    (e.g. batch size, max num frames), which previously OOMed at a smaller configuration than the
    current run. If found, we skip the current run to save time.

    Args:
        df (pd.DataFrame): The DataFrame of all runs and metrics.
        run_id (int): The current run identifier.
        target_feature (str): The feature to compare (e.g. batch size).
        accelerate_config_params (Dict[str, Any]): Current run's accelerate configuration parameters.
        train_args_params (Dict[str, Any]): Current run's CLI arguments.
        output_dir (str): Directory where output logs and configs are saved.
        git_info (Dict[str, Any]): Git information.

    Returns:
        bool: True if the run should be skipped due to a prior smaller configuration OOM, False otherwise.
    """
    current_target_feature = train_args_params.get(target_feature)
    if current_target_feature is None:
        return False
    if df.empty or target_feature not in df.columns:
        return False

    # Exclude known non-config columns from comparison
    exclude_cols = METADATA_COLUMNS + METRIC_COLUMNS

    # Combine configuration parameters (accelerate + CLI) and filter out feature & non-config keys
    combined_params = {**accelerate_config_params, **train_args_params}
    config_filter = {
        k: v
        for k, v in combined_params.items()
        if k != target_feature and k not in exclude_cols
    }

    # If any required config key isn't in the DataFrame, no match can be found
    for k in config_filter:
        if k not in df.columns:
            return False

    # Filter the DataFrame for runs with OOM status and smaller config
    found_match = False
    for _, run in df.iterrows():
        # Skip if feature conditions aren't met
        if (
            not run[target_feature]
            or run[target_feature] >= current_target_feature
            or run[STATUS] != STATUS_OOM
        ):
            continue

        # Check if all config parameters match
        if all(
            str(run.get(param)) == str(value) for param, value in config_filter.items()
        ):
            found_match = True
            break

    if found_match:
        logging.info(
            f"{STATUS_OOM_SKIPPED.upper()}: Skipping run {run_id} because a smaller {target_feature} configuration OOMed previously."
        )
        # Create a new row for the skipped run with oom-skipped status
        skipped_data = pd.Series(
            {
                RUN_ID: run_id,
                "timestamp": datetime.datetime.now(),
                **accelerate_config_params,
                **train_args_params,
                **DEFAULT_METRICS,
                **git_info,
                STATUS: STATUS_OOM_SKIPPED,
            }
        )
        # Update the dataframe: if the row exists, replace it entirely; otherwise, append it.
        if run_id not in df[RUN_ID].values:
            df = pd.concat([df, pd.DataFrame([skipped_data])], ignore_index=True)
        else:
            row_index = df.index[df[RUN_ID] == run_id][0]
            df.loc[row_index] = skipped_data

        save_dataframe(df, output_dir)
        return True

    return False


def get_run_info(idx, accelerate_config, train_args):
    run_name = "_".join(
        [
            f"mdl={train_args.get('model', 'unknown')}",
            f"bs={train_args.get('train_batch_size', 'unknown')}",
            f"res={train_args.get('resolution', 'unknown')}",
            f"gckp={train_args.get('gradient_checkpointing', 'unknown')}",
            f"mp={accelerate_config.get('mixed_precision', 'unknown')}",
            f"cmpl={accelerate_config.get('dynamo_config', {}).get('dynamo_backend', 'unknown')}",
        ]
    )
    # replace all special characters including space with underscore
    run_name = re.sub(r"[^a-zA-Z0-9_=]", "_", run_name)
    return (idx, run_name)


def prepare_run_configurations(
    combinations: List[Dict[str, Dict[str, Any]]],
    output_dir: str,
) -> None:
    """Prepare configurations for all runs.

    Args:
        combinations: List of parameter combinations
        output_dir: Directory for output files
    """
    for idx, combination in enumerate(combinations, start=1):
        accelerate_config_params = combination[ACCELERATE_CONFIG]
        train_args_params = combination[CLI_ARGS]

        # Save combined configuration
        run_id, run_name = get_run_info(
            idx, accelerate_config_params, train_args_params
        )
        config_path = os.path.join(output_dir, f"{run_id}_config_{run_name}.yaml")
        save_configuration(config_path, accelerate_config_params, train_args_params)


def should_skip(
    idx: int,
    combination: Dict[str, Dict[str, Any]],
    cfg: DictConfig,
    dataframe: pd.DataFrame,
    git_info: Dict[str, Any],
) -> bool:
    """Check if run should be skipped based on various conditions.

    Args:
        idx: Run index
        combination: Parameter combination
        cfg: Configuration dictionary
        dataframe: Existing results dataframe
        git_info: Git repository info

    Returns:
        bool: True if run should be skipped
    """
    if cfg.launcher.resume and should_skip_already_done(dataframe, idx):
        return True

    if cfg.launcher.skip_larger_bs and dataframe is not None and not dataframe.empty:
        accelerate_config_params = combination[ACCELERATE_CONFIG]
        train_args_params = combination[CLI_ARGS]

        # check for OOMs for smaller batch size configs
        if should_skip_obvious_ooms(
            dataframe,
            idx,
            TRAIN_BATCH_SIZE,
            accelerate_config_params,
            train_args_params,
            cfg.train_args.logging_dir,
            git_info,
        ):
            return True

        # check for OOMs for smaller max num frames configs
        if should_skip_obvious_ooms(
            dataframe,
            idx,
            NUM_FRAMES,
            accelerate_config_params,
            train_args_params,
            cfg.train_args.logging_dir,
            git_info,
        ):
            return True

    return False


def execute_run(
    idx: int,
    total_runs: int,
    combination: Dict[str, Dict[str, Any]],
    cfg: DictConfig,
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Execute a single training run with specific configuration parameters.

    Args:
        idx (int): The index/ID number of the current run
        total_runs (int): Total number of runs to be executed
        combination (Dict[str, Dict[str, Any]]): Dictionary containing accelerate config and CLI parameters
            for this specific run
        cfg (DictConfig): Configuration dictionary
        dataframe (pd.DataFrame): DataFrame to store results and metrics

    Returns:
        pd.DataFrame: The input dataframe, potentially modified with run results

    """
    accelerate_config_params = combination[ACCELERATE_CONFIG]
    train_args_params = combination[CLI_ARGS]
    run_id, run_name = get_run_info(idx, accelerate_config_params, train_args_params)

    # Prepare output file
    log_filename = os.path.join(
        cfg.train_args.logging_dir, f"{run_id}_logs_{run_name}.txt"
    )

    # Setup trace folders for the current run
    if "profiling_step" in train_args_params:
        profile_dir = os.path.join(
            cfg.train_args.logging_dir, f"{run_id}_profile_{run_name}"
        )
        os.makedirs(profile_dir, exist_ok=True)
        train_args_params["profiling_logging_dir"] = profile_dir
        logging.info(
            f"Profiling enabled for run {run_id}. Logs will be saved to {profile_dir}"
        )

    # Log run info
    logging.info("=" * 50)
    logging.info(f"Starting run {idx}_{run_name}/{total_runs}")
    logging.info(f"Config:\n\033[1;31m{yaml.dump(combination, sort_keys=False)}\033[0m")
    logging.info(f"Logging to {log_filename}")

    temp_accelerate_config = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False,  # refrain from deleting the file automatically, so that subprocess can access
            suffix=".yaml",
            mode="w",
            encoding="utf-8",
        ) as temp_accelerate_config:
            temp_file_path = temp_accelerate_config.name
            # Dump the dictionary to the temporary file
            yaml.dump(accelerate_config_params, temp_accelerate_config)
            # Build command line arguments, run the training script and capture the output
            cmd = build_command(temp_file_path, train_args_params)
            run_training(cmd, log_filename, idx, dry_run=cfg.launcher.dry_run)
    finally:
        # Ensure the temporary config is removed if it exists
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    return dataframe


@hydra.main(version_base=None, config_path="config")
def main(cfg: DictConfig) -> None:
    """Main function to execute the launcher script."""

    if cfg.launcher.show_config:
        logging.info(
            f"Full config:\n\033[1;31m{OmegaConf.to_yaml(cfg, resolve=True)}\033[0m"
        )
    else:
        all_combinations = generate_combinations(cfg)
        total_runs = len(all_combinations)
        logging.info(f"Total runs to execute: {total_runs}")

        prepare_run_configurations(
            all_combinations,
            cfg.train_args.logging_dir,
        )

        git_info = get_git_info()
        dataframe = generate_dataframe(
            cfg.train_args.logging_dir, cfg.launcher.warmup_steps, git_info
        )

        for idx, combination in enumerate(all_combinations, start=1):
            if should_skip(idx, combination, cfg, dataframe, git_info):
                continue
            execute_run(idx, total_runs, combination, cfg, dataframe)
            dataframe = generate_dataframe(
                cfg.train_args.logging_dir, cfg.launcher.warmup_steps, git_info
            )

        logging.info("All runs completed.")
        print_summary_statistics(dataframe)


if __name__ == "__main__":
    main()
