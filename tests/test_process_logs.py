import argparse
import logging
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from utility_scripts import process_logs


@pytest.fixture
def caplog_info_level(caplog):
    """Ensure we capture logs at info level."""
    caplog.set_level(logging.INFO)
    return caplog


@pytest.fixture
def mock_git_info():
    """A simple fixture returning mock git info."""
    return {"git_commit": "dummy", "git_branch": "dummy_branch"}


def test_parse_log_file_complex_rocm_oom(tmp_path):
    """
    Confirm process_logs.parse_log_file detects ROCM OOM from a multi-rank stack trace
    containing torch.OutOfMemoryError or 'out of memory' lines.
    """
    log_content = r"""
        [rank5]: torch.OutOfMemoryError: HIP out of memory. Tried to allocate 198.00 MiB.
        [rank2]: Traceback (most recent call last):
        2025-01-09 11:22:40,246 - INFO - Step 1: {'step_loss': 0.41044700145721436, 'step_time': 7.534288167953491, 'fps_gpu': 1.0618123201110594, 'tflops/s': 52.04832654729589}
        """
    logfile = tmp_path / "oom.log"
    logfile.write_text(log_content)

    metrics = process_logs.parse_log_file(str(logfile), warmup_steps=0)
    assert metrics["status"] == process_logs.STATUS_OOM
    assert metrics["num_samples"] == 1
    assert np.isclose(metrics["avg_loss"], 0.410447, atol=1e-5)
    assert np.isclose(metrics["avg_time"], 7.534288, atol=1e-5)
    assert np.isclose(metrics["avg_fps"], 1.0618123, atol=1e-5)
    assert np.isclose(metrics["avg_tflops"], 52.0483265, atol=1e-5)


def test_parse_log_file_complex_cuda_oom(tmp_path):
    """
    Confirm process_logs.parse_log_file detects CUDA OOM from a multi-rank stack trace
    containing torch.OutOfMemoryError or 'out of memory' lines.
    """
    log_content = r"""
        [rank2]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 256.00 MiB.
        [rank2]: Traceback (most recent call last):
        2025-01-09 11:22:40,246 - INFO - Step 1: {'step_loss': 0.41044700145721436, 'step_time': 7.534288167953491, 'fps_gpu': 1.0618123201110594, 'tflops/s': 52.04832654729589}
        """
    logfile = tmp_path / "oom.log"
    logfile.write_text(log_content)

    metrics = process_logs.parse_log_file(str(logfile), warmup_steps=0)
    assert metrics["status"] == process_logs.STATUS_OOM
    assert metrics["num_samples"] == 1
    assert np.isclose(metrics["avg_loss"], 0.410447, atol=1e-5)
    assert np.isclose(metrics["avg_time"], 7.534288, atol=1e-5)
    assert np.isclose(metrics["avg_fps"], 1.0618123, atol=1e-5)
    assert np.isclose(metrics["avg_tflops"], 52.0483265, atol=1e-5)


def test_parse_log_file_traceback_error(tmp_path):
    """
    Confirm process_logs.parse_log_file sets status=ERROR if a 'Traceback' is found,
    even if logs contain valid steps.
    """
    log_content = r"""
        2025-01-09 11:22:42,233 - INFO - Step 2: {'step_loss': 0.4012545943260193, 'step_time': 1.8575315, 'fps_gpu': 4.30679, 'tflops/s': 52.04832654729589}
        [rank2]: Traceback (most recent call last):
            ValueError("some error")
        """
    logfile = tmp_path / "traceback_error.log"
    logfile.write_text(log_content)

    metrics = process_logs.parse_log_file(str(logfile), warmup_steps=0)
    assert metrics["status"] == process_logs.STATUS_ERROR
    assert metrics["num_samples"] == 1
    assert np.isclose(metrics["avg_loss"], 0.4012545943, atol=1e-5)


def test_parse_log_file_multiple_steps_no_error(tmp_path):
    """
    Verify we parse multiple steps, skip warmup, compute tail averages,
    and end up with status=SUCCESS if no OOM or errors exist.
    """
    content = r"""
        2025-01-09 11:22:40,246 - INFO - Step 1: {'step_loss': 0.4104, 'step_time': 7.5342, 'fps_gpu': 1.0618, 'tflops/s': 52.04832654729589}
        2025-01-09 11:22:42,233 - INFO - Step 2: {'step_loss': 0.4012, 'step_time': 1.8575, 'fps_gpu': 4.3067, 'tflops/s': 276.1793374846529}
        2025-01-09 11:22:44,219 - INFO - Step 3: {'step_loss': 0.4387, 'step_time': 1.8597, 'fps_gpu': 4.3016, 'tflops/s': 275.70271792899825}
        2025-01-09 11:22:46,209 - INFO - Step 4: {'step_loss': 0.4513, 'step_time': 1.8614, 'fps_gpu': 4.2977, 'tflops/s': 275.5930278356191}
        """
    logfile = tmp_path / "normal_steps.log"
    logfile.write_text(content)

    metrics = process_logs.parse_log_file(str(logfile), warmup_steps=1, tail_steps=2)
    assert metrics["status"] == process_logs.STATUS_SUCCESS
    assert metrics["num_samples"] == 3  # steps 2..4

    arr_losses = [0.4012, 0.4387, 0.4513]
    assert np.isclose(metrics["avg_loss"], np.mean(arr_losses), atol=1e-5)


def test_parse_log_file_unreadable(tmp_path, caplog):
    """
    If file doesn't exist or is unreadable => logs an error and returns process_logs.STATUS_ERROR.
    """
    metrics = process_logs.parse_log_file(str(tmp_path / "no_file.log"), warmup_steps=0)
    assert metrics["status"] == process_logs.STATUS_ERROR
    assert "Error parsing log file" in caplog.text


def test_save_dataframe_empty(tmp_path, caplog_info_level):
    """
    Test handling of saving an empty dataframe.
    """
    df = pd.DataFrame()
    with patch.dict(os.environ, {}, clear=True):
        process_logs.save_dataframe(df, str(tmp_path))
        assert "No file will be saved" in caplog_info_level.text
        assert not os.path.exists(tmp_path / "runs_summary.csv")


@patch.dict(os.environ, {"DISABLE_RUNS_SUMMARY": "1"})
def test_save_dataframe_disabled_env(tmp_path, caplog_info_level):
    """
    Test handling of saving a dataframe when DISABLE_RUNS_SUMMARY is set.
    """
    df = pd.DataFrame({"val": [10, 20]})
    process_logs.save_dataframe(df, str(tmp_path))
    assert "Not saving runs_summary.csv." in caplog_info_level.text
    assert not os.path.exists(tmp_path / "runs_summary.csv")


def test_save_dataframe_normal(tmp_path, caplog_info_level):
    """
    Test handling of saving a valid dataframe.
    """
    df = pd.DataFrame({"val": [10, 20]})
    with patch.dict(os.environ, {}, clear=True):
        process_logs.save_dataframe(df, str(tmp_path))
        csvfile = tmp_path / "runs_summary.csv"
        assert csvfile.exists()
        assert "DataFrame saved to" in caplog_info_level.text


def test_generate_dataframe_no_configs(tmp_path, caplog_info_level):
    """
    If no N_config.yaml files => returns an empty or existing DF with log 'No config found'.
    """
    df = process_logs.generate_dataframe(
        str(tmp_path), warmup_steps=1, git_info={}, update_df=False
    )
    assert df.empty
    assert (
        "No configuration files found. No DataFrame created." in caplog_info_level.text
    )


def test_generate_dataframe_new_runs(tmp_path, mock_git_info):
    """
    If we have config+log for runs not in existing DF => parse them, store new results, save CSV.
    """
    with patch.dict(os.environ, {}, clear=True):
        existing_csv = tmp_path / "runs_summary.csv"
        existing_csv.write_text("run_id,status,avg_loss\n5,success,0.22\n")

        (tmp_path / "1_config.yaml").write_text(
            "accelerate_config:\n  testA: 1\ntrain_args:\n  foo: 'bar'"
        )
        (tmp_path / "1_logs.txt").write_text("INFO - Step 1 => step_loss': 1.0\n")

        (tmp_path / "5_config.yaml").write_text(
            "accelerate_config:\n  testB: 2\ntrain_args:\n  bar: 10"
        )

        df = process_logs.generate_dataframe(
            str(tmp_path), warmup_steps=0, git_info=mock_git_info
        )
        assert df is not None
        csvfile = tmp_path / "runs_summary.csv"
        assert csvfile.exists()

        # Read the generated CSV file to verify its contents
        final = pd.read_csv(csvfile)
        # Verify that the DataFrame contains exactly 2 rows (run_id 1 and 5)
        assert len(final) == 2, f"Found: \n{final}"

        # Get the row for run_id=1 and verify its metrics
        r1 = final[final["run_id"] == 1].iloc[0]
        assert r1["status"] == "success"  # Should be marked as successful
        assert r1["avg_loss"] == 1.0  # Should have the expected loss value

        r5 = final[final["run_id"] == 5].iloc[0]
        # The status should be either 'success' or 'not_run', but the row must exist
        assert r5["status"] in ["success", "not_run"]


def test_generate_dataframe_existing_only_no_new(tmp_path):
    """
    If we have config+log for runs already in existing DF => skip them, save CSV.
    """
    existing_csv = tmp_path / "runs_summary.csv"
    existing_csv.write_text("run_id,status\n1,success\n")
    (tmp_path / "1_config.yaml").write_text("dummy config")
    df = process_logs.generate_dataframe(
        str(tmp_path), warmup_steps=0, git_info={}, update_df=True
    )
    assert df is not None


def test_load_existing_dataframe_not_found(tmp_path):
    """
    If existing CSV not found => logs an error and returns None.
    """
    df = process_logs.load_existing_dataframe(str(tmp_path))
    assert df is None


def test_get_runs_to_update_no_existing():
    """
    If no existing DF => all runs are new.
    """
    runs = process_logs.get_runs_to_update(None, [1, 2, 3])
    assert runs == [1, 2, 3]


def test_get_runs_to_update_partial_existing():
    """
    If run 1 is set as success => update run 2 ("not_run") and 3 (missing).
    """
    df = pd.DataFrame({"run_id": [1, 2], "status": ["success", "not_run"]})
    runs = process_logs.get_runs_to_update(df, [1, 2, 3])
    assert runs == [2, 3]


def test_process_run_no_logs(tmp_path, mock_git_info):
    """
    Test processing of run with valid configuration but no log file.
    """
    (tmp_path / "10_config.yaml").write_text(
        "accelerate_config:\n  x: 5\ntrain_args:\n  y: 10\n"
    )
    result = process_logs.process_run(
        10, str(tmp_path), warmup_steps=1, git_info=mock_git_info
    )
    assert result["run_id"] == 10
    assert result["status"] == process_logs.STATUS_NOTRUN
    assert result["x"] == 5
    assert result["y"] == 10


def test_process_run_with_logs(tmp_path, mock_git_info):
    """
    Test processing of run logs with valid configuration and log files.
    """
    (tmp_path / "11_config.yaml").write_text(
        "accelerate_config:\n  p: 1.23\ntrain_args:\n  q: 4.56\n"
    )
    log_content = "INFO - Step 1 => step_loss': 0.9\nINFO - Step 2 => step_loss': 0.7\n"
    (tmp_path / "11_logs.txt").write_text(log_content)
    out = process_logs.process_run(
        11, str(tmp_path), warmup_steps=0, git_info=mock_git_info
    )
    assert out["run_id"] == 11
    assert out["status"] == process_logs.STATUS_SUCCESS
    assert out["p"] == 1.23
    assert out["q"] == 4.56
    assert out["avg_loss"] == 0.8


def test_combine_dataframes_none_existing():
    """
    Test combining dataframes when existing is None.
    """
    new = pd.DataFrame({"run_id": [3, 4], "status": ["success", "success"]})
    combined = process_logs.combine_dataframes(None, new)
    assert list(combined["run_id"]) == [3, 4]


def test_combine_dataframes_merge():
    """
    Test combining dataframes when existing and new have overlapping run_ids.
    """
    existing = pd.DataFrame({"run_id": [1, 2], "status": ["success", "error"]})
    new = pd.DataFrame({"run_id": [2, 3], "status": ["not_run", "success"]})
    merged = process_logs.combine_dataframes(existing, new)
    assert list(merged["run_id"]) == [1, 2, 3]


def test_print_summary_statistics_empty(caplog_info_level):
    """
    Test handling of an empty dataframe.
    """
    df = pd.DataFrame()
    process_logs.print_summary_statistics(df)
    assert "No runs to summarize. DataFrame is empty." in caplog_info_level.text


def test_print_summary_statistics_nonempty(caplog_info_level):
    """
    Test handling of a non-empty dataframe.
    """
    df = pd.DataFrame(
        {"run_id": [1, 2], "status": ["success", "error"], "avg_loss": [0.1, 0.2]}
    )
    process_logs.print_summary_statistics(df)
    assert "Total runs processed: 2" in caplog_info_level.text
    assert "Status distribution:" in caplog_info_level.text
    assert "success" in caplog_info_level.text
    assert "error" in caplog_info_level.text
    assert "DataFrame Summary:" in caplog_info_level.text


@patch("utility_scripts.process_logs.parse_args")
@patch("utility_scripts.process_logs.generate_dataframe")
@patch("utility_scripts.process_logs.print_summary_statistics")
def test_main_no_data(mock_print, mock_gen_df, mock_parse_args, caplog_info_level):
    """
    Test process_logs.main function when no data is found to process.
    """
    mock_parse_args.return_value = argparse.Namespace(output_dir="test_dir", warmup=5)
    mock_gen_df.return_value = pd.DataFrame()
    process_logs.main()
    assert "No data found to process" in caplog_info_level.text
    mock_print.assert_not_called()


@patch("utility_scripts.process_logs.parse_args")
@patch("utility_scripts.process_logs.generate_dataframe")
@patch("utility_scripts.process_logs.print_summary_statistics")
def test_main_with_data(mock_print, mock_gen_df, mock_parse_args, caplog_info_level):
    """
    Test process_logs.main function when data is found to process.
    """
    mock_parse_args.return_value = argparse.Namespace(output_dir="test_dir", warmup=5)
    mock_gen_df.return_value = pd.DataFrame({"run_id": [1], "status": ["success"]})
    process_logs.main()
    assert "Generating summary dataframe..." in caplog_info_level.text
    mock_print.assert_called_once()
