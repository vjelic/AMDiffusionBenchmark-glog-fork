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


def test_set_nested_value():
    """
    Test that launcher.set_nested_value correctly sets a nested key in a dictionary.
    """
    dictionary = {}
    keys = ["outer", "inner", "final"]
    value = "test_value"
    launcher.set_nested_value(dictionary, keys, value)
    assert dictionary == {"outer": {"inner": {"final": "test_value"}}}


def test_update_accelerate_config():
    """
    Test update_accelerate_config by creating a temporary template
    and verifying updates occur.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        template_path = os.path.join(tmpdir, "template.yaml")
        output_path = os.path.join(tmpdir, "updated.yaml")

        # Write a simple template
        with open(template_path, "w") as f:
            yaml.dump(
                {
                    "fsdp_config": {"fsdp_sharding_strategy": "SHARD_GRAD_OP"},
                    "dynamo_config": {
                        "dynamo_backend": "inductor",
                        "dynamo_mode": "default",
                    },
                },
                f,
            )

        override_params = {
            "fsdp_config.fsdp_sharding_strategy": "FULL_SHARD",
            "dynamo_config.dynamo_backend": "no",
        }
        launcher.update_accelerate_config(template_path, override_params, output_path)

        with open(output_path, "r") as f:
            updated = yaml.safe_load(f)

        assert updated["fsdp_config"]["fsdp_sharding_strategy"] == "FULL_SHARD"
        assert updated["dynamo_config"]["dynamo_backend"] == "no"
        assert updated["dynamo_config"]["dynamo_mode"] == "default"


def test_load_yaml_config():
    """
    Test loading a YAML config from a file.
    """
    sample_param_config = {
        launcher.ACCELERATE_CONFIG: {
            "fsdp_config.fsdp_sharding_strategy": ["FULL_SHARD"],
        },
        launcher.CLI_ARGS: {"mixed_precision": ["bf16", "fp16"]},
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "params.yaml")
        with open(config_path, "w") as f:
            yaml.dump(sample_param_config, f)

        loaded = launcher.load_yaml_config(config_path)
        assert launcher.ACCELERATE_CONFIG in loaded
        assert launcher.CLI_ARGS in loaded
        assert (
            loaded[launcher.ACCELERATE_CONFIG]
            == sample_param_config[launcher.ACCELERATE_CONFIG]
        )
        assert loaded[launcher.CLI_ARGS] == sample_param_config[launcher.CLI_ARGS]


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
    cli_args_params = {
        "mixed_precision": "fp16",
        "train_batch_size": 2,
        "use_cache": True,
    }
    cmd = launcher.build_command(accelerate_config_path, cli_args_params)
    assert cmd[:5] == [
        "accelerate",
        "launch",
        "--config_file",
        "test_accelerate.yaml",
        "train.py",
    ]
    # Boolean argument -> flag only if True
    assert "--use_cache" in cmd
    assert "--mixed_precision" in cmd
    assert "fp16" in cmd
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
        param_config_file=None,
        accelerate_config_template="fake_accel_template.yaml",
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
            patch("launcher.update_accelerate_config"),
            patch("launcher.save_configuration"),
            patch("launcher.save_dataframe"),
            patch(
                "launcher.generate_dataframe",
                return_value=pd.DataFrame([{"status": "DRY_RUN"}]),
            ),
        ):

            # Fake combinations
            mock_load_conf.return_value = [
                {
                    "accelerate_config": {
                        "fsdp_config.fsdp_sharding_strategy": "FULL_SHARD"
                    },
                    "cli_args": {"mixed_precision": "bf16"},
                }
            ]

            launcher.main()

    # Dry-run => no real subprocess calls
    mock_subprocess_run.assert_not_called()


def test_save_configuration():
    """
    Ensure that launcher.save_configuration writes out the combined config
    with launcher.accelerate_config + cli_args as YAML.
    """
    accel_params = {"fsdp_config.fsdp_sharding_strategy": "FULL_SHARD"}
    cli_params = {"mixed_precision": "bf16"}

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


def test_should_skip_obvious_ooms_true():
    """
    If a smaller batch size OOMed with the same config,
    and current run uses bigger batch_size => skip.
    """
    # We'll store a row with launcher.train_batch_size=1, status=OOM
    # and matching launcher.accelerate_config + CLI except for the batch size.
    prior_data = {
        launcher.RUN_ID: 1,
        launcher.STATUS: launcher.STATUS_OOM,
        launcher.TRAIN_BATCH_SIZE: 1,
        "mixed_precision": "bf16",
        # Hypothetical launcher.accelerate_config fields
        "fsdp_config.fsdp_sharding_strategy": "FULL_SHARD",
    }
    df = pd.DataFrame([prior_data])

    accelerate_config_params = {"fsdp_config.fsdp_sharding_strategy": "FULL_SHARD"}
    cli_args_params = {
        "mixed_precision": "bf16",
        launcher.TRAIN_BATCH_SIZE: 2,  # bigger => should skip
    }
    # We'll call it launcher.run_id=2
    out = launcher.should_skip_obvious_ooms(
        df,
        2,
        accelerate_config_params,
        cli_args_params,
        "outputs",
        {"git_hash": "abc", "git_user": "me"},
    )
    assert out is True, "Expected to skip because smaller BS 1 OOMed previously."


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
        "mixed_precision": "bf16",
        # Hypothetical launcher.accelerate_config fields
        "fsdp_config.fsdp_sharding_strategy": "FULL_SHARD",
    }
    df = pd.DataFrame([prior_data])

    accelerate_config_params = {"fsdp_config.fsdp_sharding_strategy": "FULL_SHARD"}
    cli_args_params = {
        "mixed_precision": "fp16",  # DIFFERENT!
        launcher.TRAIN_BATCH_SIZE: 2,
    }
    # We'll call it launcher.run_id=2
    out = launcher.should_skip_obvious_ooms(
        df,
        2,
        accelerate_config_params,
        cli_args_params,
        "outputs",
        {"git_hash": "abc", "git_user": "me"},
    )
    assert out is False


def test_should_skip_obvious_ooms_false_no_smaller_bs():
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
    cli_args_params = {
        "mixed_precision": "bf16",
        launcher.TRAIN_BATCH_SIZE: 2,
    }
    out = launcher.should_skip_obvious_ooms(
        df,
        2,
        accelerate_config_params,
        cli_args_params,
        "outputs",
        {"git_hash": "abc", "git_user": "me"},
    )
    assert out is False


def test_should_skip_obvious_ooms_false_same_bs():
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
    cli_args_params = {launcher.TRAIN_BATCH_SIZE: 2}
    out = launcher.should_skip_obvious_ooms(
        df,
        2,
        accelerate_config_params,
        cli_args_params,
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
    cli_args_params = {launcher.TRAIN_BATCH_SIZE: 2}
    out = launcher.should_skip_obvious_ooms(
        df,
        2,
        accelerate_config_params,
        cli_args_params,
        "outputs",
        {},
    )
    assert out is False


@patch("launcher.update_accelerate_config")
@patch("launcher.save_configuration")
def test_prepare_run_configurations(mock_save_conf, mock_update_accel):
    """
    Validate that for each combination, we call:
      launcher.accelerate_config
      launcher.save_configuration
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
            combos, output_dir=tmpdir, accelerate_template="my_template.yaml"
        )

    # For each combination, we call launcher.update_accelerate_config once
    # and launcher.save_configuration once
    assert mock_update_accel.call_count == 2
    assert mock_save_conf.call_count == 2

    # TODO
    # We can the calls more precisely if you want:
    # calls = [call(args...) ...]
    # assert mock_update_accel.mock_calls == calls


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


@patch("subprocess.run")
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


@patch("subprocess.run", side_effect=Exception("test error"))
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
    mock_ooms.assert_called_once()


@patch("launcher.should_skip_already_done", return_value=False)
@patch("launcher.should_skip_obvious_ooms", return_value=True)
def test_should_skip_true_ooms(mock_ooms, mock_done):
    """If skip_already_done=False but skip_obvious_ooms=True => True."""
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
    and logs that it's a dry run.
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
    'launcher.execute_run' calls launcher.run_training with the appropriate command
    and logs "Starting run 3/10" at INFO level.
    """
    caplog.set_level(logging.INFO)

    combo = {
        launcher.ACCELERATE_CONFIG: {
            "fsdp_config.fsdp_sharding_strategy": "FULL_SHARD"
        },
        launcher.CLI_ARGS: {"mixed_precision": "bf16"},
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
