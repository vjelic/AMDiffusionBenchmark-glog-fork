import logging
import os
import re
import tempfile
from unittest.mock import patch

import pandas as pd
import yaml
from omegaconf import OmegaConf

import launcher

# Prevent CSV generation
os.environ["DISABLE_RUNS_SUMMARY"] = "1"


@patch("hydra.initialize")
@patch("hydra.compose")
def test_hydra_config_loading(mock_compose, mock_initialize, caplog):
    """
    Test that Hydra configuration is loaded correctly with default values
    when no overrides are provided.
    """
    # Create a simple mock config
    mock_config = OmegaConf.create(
        {
            "launcher": {
                "dry_run": False,
                "resume": False,
                "skip_larger_bs": False,
                "warmup_steps": 5,
                "show_config": True,  # keep this true to avoid executing the main logic
            },
            "train_args": {
                "logging_dir": "test_output_dir",
                "profiling_logging_dir": "test_output_dir",
            },
        }
    )

    caplog.set_level(logging.INFO)
    launcher.main(mock_config)
    assert OmegaConf.to_yaml(mock_config).strip() in caplog.text.strip()


def test_generate_combinations():
    """
    Test launcher.generate_combinations with a small parameter dictionary
    and confirm the correct cartesian product is generated.
    """
    params = {
        "param1": [1, 2],
        "param2": ["A", "B"],
    }
    combos = launcher.generate_combinations(params)
    # We expect 4 combos: (1,A), (1,B), (2,A), (2,B)
    expected = [
        {"param1": 1, "param2": "A"},
        {"param1": 1, "param2": "B"},
        {"param1": 2, "param2": "A"},
        {"param1": 2, "param2": "B"},
    ]
    assert combos == expected


def test_build_command():
    """
    Test that the launcher.build_command function constructs
    the accelerate CLI command correctly.
    """
    accelerate_config_path = "test_accelerate.yaml"
    train_args_params = {
        "model": "flux",
        "train_batch_size": 2,
        "use_cache": True,
    }
    cmd = launcher.build_command(accelerate_config_path, train_args_params)
    assert cmd[:5] == [
        "accelerate",
        "launch",
        "--config_file",
        "test_accelerate.yaml",
        "train.py",
    ]
    # Boolean argument -> flag only if True
    assert "--use_cache" in cmd
    assert "--model" in cmd
    assert "flux" in cmd
    assert "--train_batch_size" in cmd
    assert "2" in cmd


@patch("hydra.compose")
@patch("launcher.subprocess.run")
def test_main_dry_run(mock_subprocess_run, mock_compose):
    """
    Test launcher.main function in dry-run mode.
    Ensures no actual subprocess is called and we have at least
    one row in the DataFrame so 'status_counts' won't be empty.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock config with dry_run=True and logging_dir set to tmpdir
        mock_config = OmegaConf.create(
            {
                "launcher": {
                    "dry_run": True,
                    "resume": False,
                    "skip_larger_bs": False,
                    "warmup_steps": 5,
                    "show_config": False,
                },
                "train_args": {
                    "logging_dir": tmpdir,  # Use tmpdir as the logging directory
                    "profiling_logging_dir": tmpdir,
                },
            }
        )

        mock_compose.return_value = mock_config

        with (
            patch(
                "launcher.generate_combinations",
                return_value=[
                    {
                        "accelerate_config": {
                            "fsdp_config.fsdp_sharding_strategy": "FULL_SHARD"
                        },
                        "train_args": {"model": "flux"},
                    }
                ],
            ),
            patch("launcher.save_configuration"),
            patch("launcher.save_dataframe"),
            patch(
                "launcher.generate_dataframe",
                return_value=pd.DataFrame([{"status": "DRY_RUN"}]),
            ),
            patch("launcher.print_summary_statistics"),
        ):
            # Call main with no arguments (Hydra will use the mocked config)
            launcher.main(mock_config)

    # Dry-run => no real subprocess calls
    mock_subprocess_run.assert_not_called()


def test_save_configuration():
    """
    Ensure that launcher.save_configuration writes out the combined config
    with launcher.ACCELERATE_CONFIG + CLI_ARGS as YAML.
    """
    accel_params = {"fsdp_config.fsdp_sharding_strategy": "FULL_SHARD"}
    cli_params = {"model": "flux"}

    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, "test_config.yaml")
        launcher.save_configuration(config_file, accel_params, cli_params)
        assert os.path.isfile(config_file)

        with open(config_file, "r") as f:
            loaded_yaml = yaml.safe_load(f)
        assert loaded_yaml[launcher.ACCELERATE_CONFIG] == accel_params
        assert loaded_yaml[launcher.CLI_ARGS] == cli_params


def test_should_skip_already_done_skips_success():
    """
    If DataFrame indicates run #5 had launcher.STATUS_SUCCESS,
    and resume=True, then we skip that run.
    """
    df = pd.DataFrame(
        [
            {launcher.RUN_ID: 5, launcher.STATUS: launcher.STATUS_SUCCESS},
            {launcher.RUN_ID: 10, launcher.STATUS: launcher.STATUS_OOM},
        ]
    )
    should_skip = launcher.should_skip_already_done(df, 5)
    assert should_skip is True


def test_should_skip_already_done_skips_oom():
    """
    If DataFrame indicates run #10 had launcher.STATUS_OOM,
    and resume=True, then we skip that run as well.
    """
    df = pd.DataFrame(
        [
            {launcher.RUN_ID: 5, launcher.STATUS: launcher.STATUS_SUCCESS},
            {launcher.RUN_ID: 10, launcher.STATUS: launcher.STATUS_OOM},
        ]
    )
    should_skip = launcher.should_skip_already_done(df, 10)
    assert should_skip is True


def test_should_skip_already_done_no_skip_unknown():
    """
    If the run does not exist or has different status,
    launcher.should_skip_already_done should return False.
    """
    df = pd.DataFrame(
        [
            {launcher.RUN_ID: 5, launcher.STATUS: launcher.STATUS_SUCCESS},
            {launcher.RUN_ID: 10, launcher.STATUS: launcher.STATUS_OOM},
        ]
    )
    # Run #999 not in the DF
    should_skip = launcher.should_skip_already_done(df, 999)
    assert should_skip is False


@patch("launcher.save_dataframe")
def test_should_skip_obvious_ooms_true_batch_size(mock_save_dataframe):
    """
    If a smaller batch size OOMed with the same config,
    and current run uses bigger batch_size => skip.
    Additionally, verify that save_dataframe is called with the updated DF
    containing the new row with launcher.STATUS_OOM_SKIPPED.
    """
    # We'll store a row with launcher.train_batch_size=1, status=OOM
    # and matching launcher.accelerate_config + CLI except for the batch size.
    prior_data = {
        launcher.RUN_ID: 1,
        launcher.STATUS: launcher.STATUS_OOM,
        launcher.TRAIN_BATCH_SIZE: 1,
        "model": "flux",
        "fsdp_config.fsdp_sharding_strategy": "FULL_SHARD",
        "resolution": "960,544",
    }
    df = pd.DataFrame([prior_data])

    accelerate_config_params = {"fsdp_config.fsdp_sharding_strategy": "FULL_SHARD"}
    train_args_params = {
        "model": "flux",
        launcher.TRAIN_BATCH_SIZE: 2,  # bigger => should skip
        "resolution": "960,544",
    }
    # We'll call it launcher.run_id=2
    out = launcher.should_skip_obvious_ooms(
        df,
        2,
        launcher.TRAIN_BATCH_SIZE,
        accelerate_config_params,
        train_args_params,
        "outputs",
        {"git_hash": "abc", "git_user": "me"},
    )
    assert (
        out is True
    ), f"Expected to skip because smaller {launcher.TRAIN_BATCH_SIZE=} OOMed previously."

    # Verify that save_dataframe was called
    mock_save_dataframe.assert_called_once()
    # Verify that the saved DF contains a new row for run 2 with status OOM_SKIPPED
    saved_df = mock_save_dataframe.call_args[0][0]
    assert 2 in saved_df[launcher.RUN_ID].values
    row = saved_df[saved_df[launcher.RUN_ID] == 2].iloc[0]
    assert row[launcher.STATUS] == launcher.STATUS_OOM_SKIPPED


@patch("launcher.save_dataframe")
def test_should_skip_obvious_ooms_true_num_frames(mock_save_dataframe):
    """
    If a smaller num_frames configuration OOMed with the same config,
    and current run uses bigger num_frames => skip.
    Verify that save_dataframe is called with the updated DF.
    """
    prior_data = {
        launcher.RUN_ID: 1,
        launcher.STATUS: launcher.STATUS_OOM,
        launcher.NUM_FRAMES: 10,
        "model": "flux",
        "fsdp_config.fsdp_sharding_strategy": "FULL_SHARD",
        "resolution": "960,544",
    }
    df = pd.DataFrame([prior_data])

    accelerate_config_params = {"fsdp_config.fsdp_sharding_strategy": "FULL_SHARD"}
    train_args_params = {
        "model": "flux",
        launcher.NUM_FRAMES: 12,  # bigger => should skip
        "resolution": "960,544",
    }
    # We'll call it launcher.run_id=2
    out = launcher.should_skip_obvious_ooms(
        df,
        2,
        launcher.NUM_FRAMES,
        accelerate_config_params,
        train_args_params,
        "outputs",
        {"git_hash": "abc", "git_user": "me"},
    )
    assert (
        out is True
    ), f"Expected to skip because smaller {launcher.NUM_FRAMES=} OOMed previously."

    mock_save_dataframe.assert_called_once()
    saved_df = mock_save_dataframe.call_args[0][0]
    assert 2 in saved_df[launcher.RUN_ID].values
    row = saved_df[saved_df[launcher.RUN_ID] == 2].iloc[0]
    assert row[launcher.STATUS] == launcher.STATUS_OOM_SKIPPED


def test_should_skip_obvious_ooms_false_different_config():
    """
    If a smaller batch size OOMed with different config,
    and current run uses bigger batch_size => do not skip.
    """
    # We'll store a row with launcher.train_batch_size=1, status=OOM
    # and NON matching launcher.accelerate_config + CLI wuth larger batch size.
    prior_data = {
        launcher.RUN_ID: 1,
        launcher.STATUS: launcher.STATUS_OOM,
        launcher.TRAIN_BATCH_SIZE: 1,
        "model": "flux",
        # Hypothetical launcher.accelerate_config fields
        "fsdp_config.fsdp_sharding_strategy": "FULL_SHARD",
    }
    df = pd.DataFrame([prior_data])

    accelerate_config_params = {"fsdp_config.fsdp_sharding_strategy": "FULL_SHARD"}
    train_args_params = {
        "model": "stable-diffusion-xl",  # DIFFERENT!
        launcher.TRAIN_BATCH_SIZE: 2,
    }
    # We'll call it launcher.run_id=2
    out = launcher.should_skip_obvious_ooms(
        df,
        2,
        launcher.TRAIN_BATCH_SIZE,
        accelerate_config_params,
        train_args_params,
        "outputs",
        {"git_hash": "abc", "git_user": "me"},
    )
    assert out is False


def test_should_skip_obvious_ooms_false_no_smaller_config():
    """
    If no smaller batch size has OOMed with same config, do not skip.
    """
    df = pd.DataFrame(
        [
            {
                launcher.RUN_ID: 1,
                launcher.STATUS: launcher.STATUS_SUCCESS,
                launcher.TRAIN_BATCH_SIZE: 1,
            }
        ]
    )
    accelerate_config_params = {"fsdp_config.fsdp_sharding_strategy": "FULL_SHARD"}
    train_args_params = {
        "model": "flux",
        launcher.TRAIN_BATCH_SIZE: 2,
    }
    out = launcher.should_skip_obvious_ooms(
        df,
        2,
        launcher.TRAIN_BATCH_SIZE,
        accelerate_config_params,
        train_args_params,
        "outputs",
        {"git_hash": "abc", "git_user": "me"},
    )
    assert out is False


def test_should_skip_obvious_ooms_false_same_feature():
    """
    If the run that OOMed had the same (not smaller) batch size,
    should not skip the new run.
    """
    df = pd.DataFrame(
        [
            {
                launcher.RUN_ID: 1,
                launcher.STATUS: launcher.STATUS_OOM,
                launcher.TRAIN_BATCH_SIZE: 2,
            }
        ]
    )
    accelerate_config_params = {}
    train_args_params = {launcher.TRAIN_BATCH_SIZE: 2}
    out = launcher.should_skip_obvious_ooms(
        df,
        2,
        launcher.TRAIN_BATCH_SIZE,
        accelerate_config_params,
        train_args_params,
        "outputs",
        {},
    )
    assert out is False


def test_should_skip_obvious_ooms_false_empty_df():
    """
    If the DF is empty, we obviously do not skip.
    """
    df = pd.DataFrame()
    accelerate_config_params = {}
    train_args_params = {launcher.TRAIN_BATCH_SIZE: 2}
    out = launcher.should_skip_obvious_ooms(
        df,
        2,
        launcher.TRAIN_BATCH_SIZE,
        accelerate_config_params,
        train_args_params,
        "outputs",
        {},
    )
    assert out is False


@patch("launcher.save_configuration")
def test_prepare_run_configurations(mock_save_conf):
    """
    Validate that for each combination, we call:
      launcher.ACCELERATE_CONFIG and launcher.save_configuration
    with correct file names.
    """
    combos = [
        {
            launcher.ACCELERATE_CONFIG: {"accel_key": "valA"},
            launcher.CLI_ARGS: {"cli_key": "val1"},
        },
        {
            launcher.ACCELERATE_CONFIG: {"accel_key": "valB"},
            launcher.CLI_ARGS: {"cli_key": "val2"},
        },
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        # We just want to test calls, no actual file writes

        launcher.prepare_run_configurations(
            combos,
            output_dir=tmpdir,
        )

    # For each combination, save_configuration should be called once.
    assert mock_save_conf.call_count == 2


@patch("launcher.should_skip_already_done", return_value=True)
def test_should_skip_true_already_done(mock_done):
    """
    If the run was already done, we skip directly without checking OOM logic.
    """

    # Create a Hydra-style config
    cfg = OmegaConf.create(
        {
            "launcher": {"resume": True, "skip_larger_bs": True},
            "train_args": {
                "logging_dir": "test_output_dir",
                "profiling_logging_dir": "test_output_dir",
            },
        }
    )

    df = pd.DataFrame()
    combination = {}

    skip_result = launcher.should_skip(5, combination, cfg, df, {})
    assert skip_result is True
    mock_done.assert_called_once()


@patch("launcher.subprocess.Popen")
def test_run_training_normal(mock_subproc_run):
    """
    launcher.run_training should call subprocess.run with the specified command,
    capturing stdout/stderr to the log file.
    """
    cmd = ["accelerate", "launch", "--config_file", "foo.yaml", "train.py"]

    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "dummy_logs.txt")
        launcher.run_training(cmd, log_file, idx=1, dry_run=False)

    mock_subproc_run.assert_called_once()
    # Validate call signature
    called_args, called_kwargs = mock_subproc_run.call_args
    assert called_args[0] == cmd
    assert "stdout" in called_kwargs
    assert "stderr" in called_kwargs


@patch("launcher.subprocess.Popen", side_effect=Exception("test error"))
def test_run_training_exception(mock_subproc_run, caplog):
    """
    If launcher.run_training triggers an exception, it should be logged.
    """
    cmd = ["accelerate", "launch", "--config_file", "foo.yaml", "train.py"]

    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "dummy_logs.txt")
        launcher.run_training(cmd, log_file, idx=1, dry_run=False)
    assert "encountered an unexpected error: test error" in caplog.text


@patch("launcher.should_skip_already_done", return_value=False)
@patch("launcher.should_skip_obvious_ooms", return_value=False)
def test_should_skip_false(mock_ooms, mock_done):
    """If both sub-checks are False, final skip is False."""

    cfg = OmegaConf.create(
        {
            "launcher": {"resume": True, "skip_larger_bs": True},
            "train_args": {
                "logging_dir": "test_output_dir",
                "profiling_logging_dir": "test_output_dir",
            },
        }
    )
    df = pd.DataFrame([{launcher.RUN_ID: 1, launcher.STATUS: launcher.STATUS_SUCCESS}])
    combination = {launcher.ACCELERATE_CONFIG: {}, launcher.CLI_ARGS: {}}
    out = launcher.should_skip(1, combination, cfg, df, {})
    assert out is False
    mock_done.assert_called_once()
    mock_ooms.assert_called()


@patch("launcher.should_skip_already_done", return_value=False)
@patch("launcher.should_skip_obvious_ooms", return_value=True)
def test_should_skip_true_ooms(mock_ooms, mock_done):
    """If skip_already_done is False but skip_obvious_ooms is True, then skip."""

    cfg = OmegaConf.create(
        {
            "launcher": {"resume": True, "skip_larger_bs": True},
            "train_args": {
                "logging_dir": "test_output_dir",
                "profiling_logging_dir": "test_output_dir",
            },
        }
    )

    df = pd.DataFrame([{launcher.RUN_ID: 123}])
    combination = {launcher.ACCELERATE_CONFIG: {}, launcher.CLI_ARGS: {}}
    out = launcher.should_skip(2, combination, cfg, df, {})
    assert out is True
    mock_done.assert_called_once()
    mock_ooms.assert_called_once()


def test_run_training_dry_run(caplog):
    """
    In dry_run mode, launcher.run_training should NOT call subprocess.run
    and should log that it's a dry run.
    """
    caplog.set_level(logging.INFO)

    cmd = ["some", "cmd"]
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "dummy.txt")
        launcher.run_training(cmd, log_file, idx=2, dry_run=True)

    assert "DRY RUN:  command for run 2" in caplog.text


@patch("launcher.run_training")
def test_execute_run(mock_run_training, caplog):
    """
    launcher.execute_run calls launcher.run_training with the appropriate command
    and logs "Starting run 3/10" at INFO level.
    """
    caplog.set_level(logging.INFO)

    combo = {
        launcher.ACCELERATE_CONFIG: {
            "fsdp_config.fsdp_sharding_strategy": "FULL_SHARD"
        },
        launcher.CLI_ARGS: {"model": "flux"},
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = OmegaConf.create(
            {
                "launcher": {
                    "dry_run": False,
                    "warmup_steps": 5,
                    "resume": False,
                    "skip_larger_bs": False,
                },
                "train_args": {"logging_dir": tmpdir, "profiling_logging_dir": tmpdir},
            }
        )

        df = pd.DataFrame()
        updated_df = launcher.execute_run(3, 10, combo, cfg, df)
        assert updated_df is df
    mock_run_training.assert_called_once()
    assert re.search(r"Starting run 3_.*\/10", caplog.text)
