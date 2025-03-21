import argparse
import logging
import os
import tempfile
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

import launcher

# Prevent CSV generation
os.environ["DISABLE_RUNS_SUMMARY"] = "1"


def test_parse_args_no_args():
    """
    Test launcher.parse_args with no command-line arguments,
    which should raise a SystemExit if the default files
    do not exist (mocked via os.path.isfile=False).
    """
    test_argv = []
    with (
        patch("sys.argv", ["launcher.py"] + test_argv),
        patch("launcher.os.path.isfile", return_value=False),
        pytest.raises(SystemExit),
    ):
        launcher.parse_args()


def test_load_yaml_config():
    """
    Test loading a YAML config from a file.
    """
    sample_config = {
        launcher.ACCELERATE_CONFIG: {
            "fsdp_config.fsdp_sharding_strategy": ["FULL_SHARD"],
        },
        launcher.CLI_ARGS: {"model": ["flux", "stable-diffusion-xl"]},
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "params.yaml")
        with open(config_path, "w") as f:
            yaml.dump(sample_config, f)

        loaded = launcher.load_yaml_config(config_path)
        assert launcher.ACCELERATE_CONFIG in loaded
        assert launcher.CLI_ARGS in loaded
        assert (
            loaded[launcher.ACCELERATE_CONFIG]
            == sample_config[launcher.ACCELERATE_CONFIG]
        )
        assert loaded[launcher.CLI_ARGS] == sample_config[launcher.CLI_ARGS]


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


def test_combine_configurations():
    """
    Test that accelerate + CLI combos are combined properly
    without making assumptions about order.
    """
    accel_combos = [{"accel1": "a"}, {"accel1": "b"}]
    cli_combos = [{"cli1": 1}, {"cli1": 2}]
    combined = launcher.combine_configurations(accel_combos, cli_combos)

    # Check we get the expected number of combinations
    assert len(combined) == 4

    # Create set of expected combinations
    expected = {
        (("accel1", "a"), ("cli1", 1)),
        (("accel1", "a"), ("cli1", 2)),
        (("accel1", "b"), ("cli1", 1)),
        (("accel1", "b"), ("cli1", 2)),
    }

    # Convert actual results to comparable tuples
    actual = {
        tuple(
            sorted(c[launcher.ACCELERATE_CONFIG].items())
            + sorted(c[launcher.CLI_ARGS].items())
        )
        for c in combined
    }

    assert actual == expected


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


@patch("launcher.parse_args")
@patch("launcher.subprocess.run")
def test_main_dry_run(mock_subprocess_run, mock_parse_args):
    """
    Test launcher.main function in dry-run mode.
    Ensures no actual subprocess is called and we have at least
    one row in the DataFrame so 'status_counts' won't be empty.
    """
    # Mock launcher.parse_args to return minimal valid arguments
    mock_args = argparse.Namespace(
        config_file=None,
        output_dir="test_outputs",  # Will override with tmpdir below
        dry_run=True,
        resume=False,
        skip_larger_bs=False,
        warmup_steps=5,
    )
    mock_parse_args.return_value = mock_args

    with tempfile.TemporaryDirectory() as tmpdir:
        # Override the output_dir to ensure all artifacts go in tmpdir
        mock_args.output_dir = tmpdir

        with (
            patch("launcher.load_configurations") as mock_load_conf,
            patch("launcher.save_configuration"),
            patch("launcher.save_dataframe"),
            patch(
                "launcher.generate_dataframe",
                return_value=pd.DataFrame([{"status": "DRY_RUN"}]),
            ),
        ):

            # Fake combinations
            mock_conf = [
                {
                    "accelerate_config": {
                        "fsdp_config.fsdp_sharding_strategy": "FULL_SHARD"
                    },
                    "train_args": {"model": "flux"},
                }
            ]

            launcher.main(mock_args, mock_conf)

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
    args = argparse.Namespace(resume=True, skip_larger_bs=True)
    df = pd.DataFrame()
    combination = {}

    skip_result = launcher.should_skip(5, combination, args, df, {})
    assert skip_result is True
    mock_done.assert_called_once()


@patch("launcher.subprocess.run")
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


@patch("launcher.subprocess.run", side_effect=Exception("test error"))
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
    args = argparse.Namespace(
        resume=True,
        skip_larger_bs=True,
        output_dir="some_dir",
    )
    df = pd.DataFrame([{launcher.RUN_ID: 1, launcher.STATUS: launcher.STATUS_SUCCESS}])
    combination = {launcher.ACCELERATE_CONFIG: {}, launcher.CLI_ARGS: {}}
    out = launcher.should_skip(1, combination, args, df, {})
    assert out is False
    mock_done.assert_called_once()
    mock_ooms.assert_called()


@patch("launcher.should_skip_already_done", return_value=False)
@patch("launcher.should_skip_obvious_ooms", return_value=True)
def test_should_skip_true_ooms(mock_ooms, mock_done):
    """If skip_already_done is False but skip_obvious_ooms is True, then skip."""
    args = argparse.Namespace(
        resume=True,
        skip_larger_bs=True,
        output_dir="some_dir",
    )
    df = pd.DataFrame([{launcher.RUN_ID: 123}])
    combination = {launcher.ACCELERATE_CONFIG: {}, launcher.CLI_ARGS: {}}
    out = launcher.should_skip(2, combination, args, df, {})
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
        args = argparse.Namespace(
            output_dir=tmpdir,
            dry_run=False,
            warmup_steps=5,
            resume=False,
            skip_larger_bs=False,
        )
        df = pd.DataFrame()
        updated_df = launcher.execute_run(3, 10, combo, args, df)
        assert updated_df is df
    mock_run_training.assert_called_once()

    assert "Starting run 3/10" in caplog.text
