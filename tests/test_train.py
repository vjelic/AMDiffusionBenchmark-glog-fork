import sys
from unittest.mock import MagicMock, patch

import pytest

import train


@pytest.fixture
def minimal_args():
    """
    Minimal set of CLI arguments for parse_args.
    """
    return [
        "--model_path",
        "some/fake-model-path",
        "--train_data_path",
        "path/to/fake-data",
        "--num_iterations",
        "1",
        "--train_batch_size",
        "1",
        "--substitute_sdpa_with_flash_attn",
        "False",
    ]


def test_parse_args_minimal(monkeypatch: pytest.MonkeyPatch, minimal_args: list):
    """
    Check parse_args with minimal arguments.
    """
    # Mock the `check_gpu_vendor` function to always return 'rocm'
    with patch("train.check_gpu_vendor", return_value="rocm"):

        # Set up command line arguments
        argv = ["train.py"] + minimal_args
        monkeypatch.setattr(sys, "argv", argv)

        args = train.parse_args()

        assert args.model_path == "some/fake-model-path"
        assert args.train_data_path == "path/to/fake-data"
        assert args.num_iterations == 1
        assert args.train_batch_size == 1


def test_trainer_init(minimal_args: list, monkeypatch: pytest.MonkeyPatch):
    """
    Ensure trainer init doesn't fail (CPU friendly).
    """
    with (
        patch("train.check_gpu_vendor", return_value="rocm"),
        patch("src.models.flux.FluxModel", new=MagicMock()),
        patch("src.models.hunyuan.HunyuanVideoModel", new=MagicMock()),
    ):

        # Set up command line arguments
        argv = ["train.py"] + minimal_args
        monkeypatch.setattr(sys, "argv", argv)

        args = train.parse_args()

        # Create trainer instance and verify it initializes
        trainer = train.Trainer(args)
        assert trainer.args == args
