import os
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch
from dotenv import load_dotenv

from src.trainer import ModelManager
from src.utils import safely_eval_as_bool

load_dotenv()

# Define config paths
BASE_CONFIG = str(Path("config", "single_run.yaml"))


def run_subprocess_and_check_success(command, success_tag="completed successfully!"):
    """
    Run a subprocess command, capture output to a file, and check for success.

    Args:
        command: List of command parts to execute
        success_tag: String to search for in output to confirm success

    Returns:
        None, but asserts if command fails or success message not found
    """
    with tempfile.NamedTemporaryFile(mode="w+") as output_file:
        result = subprocess.run(
            command, stdout=output_file, stderr=subprocess.STDOUT, text=True
        )
        # Read output and check for success message
        output_file.seek(0)
        output_text = output_file.read()
        output_text_excerpt = (
            output_text[-1000:] if len(output_text) > 1000 else output_text
        )
        # Check return code
        assert result.returncode == 0, (
            f"Command failed with code {result.returncode}: {command}."
            f"Output excerpt: {output_text_excerpt}"
        )

        assert success_tag in output_text, (
            f"Success message not found in output. Command: {command}\n"
            f"Output excerpt: {output_text_excerpt}"
        )


def get_training_run_command(
    model: str, tmp_dir: tempfile.TemporaryDirectory, base_config: str = BASE_CONFIG
):
    return [
        "python",
        "launcher.py",
        "--config_file",
        base_config,
        f"train_args={model}",
        "--output_dir",
        tmp_dir,
        "--no_resume",
        "++train_args.train_batch_size=1",
        "++train_args.resolution=128",
        "++train_args.num_iterations=1",
        "++train_args.max_train_samples=2",
        "++train_args.use_cache=false",  # too slow otherwise
    ]


# Dry run tests that can run without GPU
@pytest.mark.parametrize("model", ModelManager().get_available_models())
def test_launcher_with_dry_run(model):
    """Integration test of model configurations in dry-run mode."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        command = get_training_run_command(model=model, tmp_dir=tmp_dir)
        command.append("--dry_run")
        run_subprocess_and_check_success(command, "All runs completed.")


# Full run tests that require GPU
@pytest.mark.parametrize("model", ModelManager().get_available_models())
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU")
@pytest.mark.skipif(
    safely_eval_as_bool(os.getenv("SKIP_GPU_TESTS", "false")),
    reason="Skipping GPU tests as per environment variable.",
)
def test_launcher_with_real_run(model):
    """Integration test of model configurations with real runs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        command = get_training_run_command(model=model, tmp_dir=tmp_dir)
        run_subprocess_and_check_success(command, "completed successfully!")
