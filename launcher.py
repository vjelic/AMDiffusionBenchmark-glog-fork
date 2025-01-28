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
    --param_config_file config/minimal_benchmark_config.yaml \
    --accelerate_config_template config/accelerate_fsdp_config.yaml \
    --output_dir outputs/runs/sweep_000 \
    --dry-run

The script generates a runs_summary.csv with metrics and metadata for analysis.
"""

import argparse
import copy
import datetime
import itertools
import logging
import os
import subprocess
import time
from typing import Any, Dict, List

import git
import pandas as pd
import yaml

from src.constants import (
    ACCELERATE_CONFIG,
    CLI_ARGS,
    DEFAULT_METRICS,
    METADATA_COLUMNS,
    METRIC_COLUMNS,
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


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments with parameters:
            param_config_file: Path to the YAML parameter configuration file
            accelerate_config_template: Path to the Accelerate config template
            output_dir: Directory for output logs and configs
            dry_run: Whether to do a dry run without execution
            warmup_steps: Number of initial steps to exclude from averages
            no-resume: Whether to skip completed or failed OOM runs
            no-skip_larger_bs: Whether to skip larger batch sizes after OOM
    """
    parser = argparse.ArgumentParser(
        description="Launcher script for training configurations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--param_config_file",
        type=str,
        default=os.getenv(
            "LAUNCHER_PARAM_CONFIG", "config/minimal_benchmark_config.yaml"
        ),
        help="Path to the YAML parameter configuration file containing parameter values.",
    )
    parser.add_argument(
        "--accelerate_config_template",
        type=str,
        default=os.getenv(
            "LAUNCHER_ACCELERATE_CONFIG", "config/accelerate_fsdp_config.yaml"
        ),
        help="Path to the Accelerate config file template.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.getenv("LAUNCHER_OUTPUT_DIR", "outputs/runs/sweep_000"),
        help="Directory where output logs and configs will be saved.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Whether to perform a dry run without executing training commands.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=int(os.getenv("LAUNCHER_WARMUP_STEPS", "5")),
        help="Number of initial steps to exclude from average computations.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="If specified, do not skip any runs regardless of previous status.",
    )
    parser.add_argument(
        "--no-skip-larger-bs",
        action="store_false",
        dest="skip_larger_bs",
        help="Do not skip runs with larger batch sizes if smaller ones OOMed.",
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.isfile(args.param_config_file):
        parser.error(f"Parameter config file not found: {args.param_config_file}")
    if not os.path.isfile(args.accelerate_config_template):
        parser.error(
            f"Accelerate config template not found: {args.accelerate_config_template}"
        )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def set_nested_value(d: Dict[str, Any], keys: List[str], value: Any) -> None:
    """Recursively sets a value in a nested dictionary.

    If intermediate keys do not exist, they will be created as empty dictionaries.

    Args:
        d (Dict[str, Any]): The dictionary to update.
        keys (List[str]): List of keys representing the path to the value.
        value (Any): The value to set.
    """
    if len(keys) == 1:
        d[keys[0]] = value
    else:
        if not isinstance(d.get(keys[0]), dict):
            d[keys[0]] = {}
        set_nested_value(d[keys[0]], keys[1:], value)


def update_accelerate_config(
    template_path: str, override_params: Dict[str, Any], output_path: str
) -> None:
    """Updates an Accelerate configuration file with overridden parameters.

    Args:
        template_path (str): Path to the template Accelerate config YAML file.
        override_params (Dict[str, Any]): Dictionary of parameters to override in the config.
            Keys can be nested keys separated by dots.
        output_path (str): Path where the updated config file will be saved.
    """
    # Load the template accelerate config
    with open(template_path, "r") as f:
        accelerate_config = yaml.safe_load(f)

    # Create a deep copy to avoid modifying the original template
    updated_config = copy.deepcopy(accelerate_config)

    # Override parameters using the helper function
    for key, value in override_params.items():
        keys = key.split(".")
        set_nested_value(updated_config, keys, value)

    # Save the updated accelerate config
    with open(output_path, "w") as f:
        yaml.dump(updated_config, f)


def load_configurations(param_config_file: str) -> List[Dict[str, Dict[str, Any]]]:
    """Loads configurations from a parameter configuration file and generates all combinations.

    Args:
        param_config_file (str): Path to the YAML parameter configuration file.

    Returns:
        List[Dict[str, Dict[str, Any]]]: A list of configurations, each containing
            'accelerate_config' and 'cli_args' dictionaries.
    """
    config = load_yaml_config(param_config_file)
    accelerate_combinations = generate_combinations(config.get(ACCELERATE_CONFIG, {}))
    cli_combinations = generate_combinations(config.get(CLI_ARGS, {}))
    return combine_configurations(accelerate_combinations, cli_combinations)


def load_yaml_config(param_config_file: str) -> Dict[str, Any]:
    """Loads a YAML configuration file.

    Args:
        param_config_file (str): Path to the YAML parameter configuration file.

    Returns:
        Dict[str, Any]: Parsed YAML configuration.
    """
    with open(param_config_file, "r") as f:
        return yaml.safe_load(f)


def generate_combinations(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generates all combinations of parameters.

    Args:
        params (Dict[str, Any]): Dictionary of parameters.

    Returns:
        List[Dict[str, Any]]: List of parameter combinations.
    """
    if params:
        param_names = list(params.keys())
        param_values = list(params.values())
        return [
            dict(zip(param_names, values))
            for values in itertools.product(*param_values)
        ]
    return [{}]


def combine_configurations(
    accelerate_combinations: List[Dict[str, Any]],
    cli_combinations: List[Dict[str, Any]],
) -> List[Dict[str, Dict[str, Any]]]:
    """Combines accelerate and CLI parameter combinations.

    Args:
        accelerate_combinations (List[Dict[str, Any]]): List of accelerate parameter combinations.
        cli_combinations (List[Dict[str, Any]]): List of CLI parameter combinations.

    Returns:
        List[Dict[str, Dict[str, Any]]]: Combined configurations.
    """
    all_combinations = []
    for cli_combination in cli_combinations:
        for accel_combination in accelerate_combinations:
            combined = {
                ACCELERATE_CONFIG: accel_combination,
                CLI_ARGS: cli_combination,
            }
            all_combinations.append(combined)
    return all_combinations


def build_command(
    unique_accelerate_config_path: str, cli_args_params: Dict[str, Any]
) -> List[str]:
    """Builds the command to execute the training script.

    Args:
        unique_accelerate_config_path (str): Path to the unique Accelerate config file.
        cli_args_params (Dict[str, Any]): Dictionary of CLI arguments.

    Returns:
        List[str]: The command to be executed as a list of strings.
    """
    cmd = [
        "accelerate",
        "launch",
        "--config_file",
        unique_accelerate_config_path,
        "train.py",
    ]
    for param, value in cli_args_params.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{param}")
        else:
            cmd.extend([f"--{param}", str(value)])
    return cmd


def save_configuration(
    config_filename: str,
    accelerate_config_params: Dict[str, Any],
    cli_args_params: Dict[str, Any],
) -> None:
    """Saves the combined configuration to a YAML file.

    Args:
        config_filename (str): Path to the configuration file.
        accelerate_config_params (Dict[str, Any]): Accelerate configuration parameters.
        cli_args_params (Dict[str, Any]): CLI arguments.
    """
    combined_config = {
        ACCELERATE_CONFIG: accelerate_config_params,
        CLI_ARGS: cli_args_params,
    }
    with open(config_filename, "w") as f:
        yaml.dump(combined_config, f)


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
                subprocess.run(
                    cmd, stdout=log_file, stderr=subprocess.STDOUT, check=True
                )
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
    accelerate_config_params: Dict[str, Any],
    cli_args_params: Dict[str, Any],
    output_dir: str,
    git_info: Dict[str, Any],
) -> bool:
    """Determines whether to skip a run if a previous smaller batch size configuration resulted in OOM.

    This logic checks if there's a configuration identical in all parameters except batch size,
    which previously OOMed at a smaller batch size than the current run's batch size. If found,
    we skip the current run to save time.

    Args:
        df (pd.DataFrame): The DataFrame of all runs and metrics.
        run_id (int): The current run identifier.
        accelerate_config_params (Dict[str, Any]): Current run's accelerate configuration parameters.
        cli_args_params (Dict[str, Any]): Current run's CLI arguments.
        output_dir (str): Directory where output logs and configs are saved.
        git_info (Dict[str, Any]): Git information.

    Returns:
        bool: True if the run should be skipped due to a prior smaller batch size OOM, False otherwise.
    """
    current_batch_size = cli_args_params.get(TRAIN_BATCH_SIZE)
    if current_batch_size is None:
        return False
    if df.empty or TRAIN_BATCH_SIZE not in df.columns:
        return False

    # Exclude known non-config columns from comparison
    exclude_cols = METADATA_COLUMNS + METRIC_COLUMNS

    # Combine configuration parameters (accelerate + CLI) and filter out batch size & non-config keys
    combined_params = {**accelerate_config_params, **cli_args_params}
    config_filter = {
        k: v
        for k, v in combined_params.items()
        if k != TRAIN_BATCH_SIZE and k not in exclude_cols
    }

    # If any required config key isn't in the DataFrame, no match can be found
    for k in config_filter:
        if k not in df.columns:
            return False

    # Filter the DataFrame for runs with OOM status and smaller batch size
    found_match = False
    for _, run in df.iterrows():
        # Skip if batch size conditions aren't met
        if (
            not run[TRAIN_BATCH_SIZE]
            or run[TRAIN_BATCH_SIZE] >= current_batch_size
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
            f"{STATUS_OOM_SKIPPED.upper()}: Skipping run {run_id} because a smaller batch size configuration OOMed previously."
        )
        # Create a new row for the skipped run with oom-skipped status
        skipped_data = pd.Series(
            {
                RUN_ID: run_id,
                "timestamp": datetime.datetime.now(),
                **accelerate_config_params,
                **cli_args_params,
                **DEFAULT_METRICS,
                **git_info,
                STATUS: STATUS_OOM_SKIPPED,
            }
        )
        # Update the dataframe
        if run_id not in df[RUN_ID].values:
            df = pd.concat([df, pd.DataFrame([skipped_data])], ignore_index=True)
        else:
            # Convert the data types before assignment
            for col in skipped_data.index:
                df.loc[df[RUN_ID] == run_id, col] = df[col].dtype.type(
                    skipped_data[col]
                )

        save_dataframe(df, output_dir)
        return True

    return False


def prepare_run_configurations(
    combinations: List[Dict[str, Dict[str, Any]]],
    output_dir: str,
    accelerate_template: str,
) -> None:
    """Prepare configurations for all runs.

    Args:
        combinations: List of parameter combinations
        output_dir: Directory for output files
        accelerate_template: Path to accelerate config template
    """
    for idx, combination in enumerate(combinations, start=1):
        accelerate_config_params = combination[ACCELERATE_CONFIG]
        cli_args_params = combination[CLI_ARGS]

        # Create unique accelerate config
        unique_accelerate_config_path = os.path.join(
            output_dir, f"{idx}_accelerate_config.yaml"
        )
        update_accelerate_config(
            accelerate_template,
            accelerate_config_params,
            unique_accelerate_config_path,
        )

        # Save combined configuration
        config_filename = os.path.join(output_dir, f"{idx}_config.yaml")
        save_configuration(config_filename, accelerate_config_params, cli_args_params)


def should_skip(
    idx: int,
    combination: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
    dataframe: pd.DataFrame,
    git_info: Dict[str, Any],
) -> bool:
    """Check if run should be skipped based on various conditions.

    Args:
        idx: Run index
        combination: Parameter combination
        args: Command line arguments
        dataframe: Existing results dataframe
        git_info: Git repository info

    Returns:
        bool: True if run should be skipped
    """
    if args.resume and should_skip_already_done(dataframe, idx):
        return True

    if args.skip_larger_bs and dataframe is not None and not dataframe.empty:
        accelerate_config_params = combination[ACCELERATE_CONFIG]
        cli_args_params = combination[CLI_ARGS]
        if should_skip_obvious_ooms(
            dataframe,
            idx,
            accelerate_config_params,
            cli_args_params,
            args.output_dir,
            git_info,
        ):
            return True

    return False


def execute_run(
    idx: int,
    total_runs: int,
    combination: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Execute a single training run with specific configuration parameters.

    Args:
        idx (int): The index/ID number of the current run
        total_runs (int): Total number of runs to be executed
        combination (Dict[str, Dict[str, Any]]): Dictionary containing accelerate config and CLI parameters
            for this specific run
        args (argparse.Namespace): Parsed command line arguments containing global settings
        dataframe (pd.DataFrame): DataFrame to store results and metrics

    Returns:
        pd.DataFrame: The input dataframe, potentially modified with run results

    """
    accelerate_config_params = combination[ACCELERATE_CONFIG]
    cli_args_params = combination[CLI_ARGS]

    # Setup config and logs
    unique_accelerate_config_path = os.path.join(
        args.output_dir, f"{idx}_accelerate_config.yaml"
    )
    # Prepare output file
    log_filename = os.path.join(args.output_dir, f"{idx}_logs.txt")

    # Log run info
    logging.info("=" * 50)
    logging.info(f"Starting run {idx}/{total_runs}")
    logging.info(f"Accelerate Config Params: {accelerate_config_params}")
    logging.info(f"CLI Args: {cli_args_params}")
    logging.info(f"Logging to {log_filename}")

    # Build command line arguments, run the training script and capture the output
    cmd = build_command(unique_accelerate_config_path, cli_args_params)
    run_training(cmd, log_filename, idx, dry_run=args.dry_run)

    return dataframe


def main() -> None:
    """Main function to execute the launcher script."""
    args = parse_args()
    all_combinations = load_configurations(args.param_config_file)
    total_runs = len(all_combinations)
    logging.info(f"Total runs to execute: {total_runs}")

    prepare_run_configurations(
        all_combinations, args.output_dir, args.accelerate_config_template
    )

    git_info = get_git_info()
    dataframe = generate_dataframe(args.output_dir, args.warmup_steps, git_info)

    for idx, combination in enumerate(all_combinations, start=1):
        if should_skip(idx, combination, args, dataframe, git_info):
            continue
        execute_run(idx, total_runs, combination, args, dataframe)
        dataframe = generate_dataframe(args.output_dir, args.warmup_steps, git_info)

    logging.info("All runs completed.")
    print_summary_statistics(dataframe)


if __name__ == "__main__":
    main()
