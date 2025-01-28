import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

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


def test_parse_args_minimal(monkeypatch, minimal_args):
    """
    Check parse_args with minimal arguments.
    """
    # Mock the check_gpu_vendor to return 'rocm'
    with patch("train.check_gpu_vendor", return_value="rocm"):

        # Set up command line arguments
        argv = ["train.py"] + minimal_args
        monkeypatch.setattr(sys, "argv", argv)

        args = train.parse_args()

        assert args.model_path == "some/fake-model-path"
        assert args.train_data_path == "path/to/fake-data"
        assert args.num_iterations == 1
        assert args.train_batch_size == 1

        # Test some defaults
        assert args.output_dir == "./outputs"
        assert args.resolution == 512
        assert args.shift == 3.0


def test_trainer_init(minimal_args, monkeypatch):
    """
    Ensure trainer init doesn't fail (CPU friendly).
    """
    with (
        patch("train.check_gpu_vendor", return_value="rocm"),
        patch("train.FlowMatchEulerDiscreteScheduler", new=MagicMock()),
        patch("train.FluxTransformer2DModel.from_pretrained", return_value=MagicMock()),
        patch("train.CLIPTextModel.from_pretrained", return_value=MagicMock()),
        patch("train.T5EncoderModel.from_pretrained", return_value=MagicMock()),
        patch("train.AutoencoderKL.from_pretrained", return_value=MagicMock()),
        patch("train.CLIPTokenizer.from_pretrained", return_value=MagicMock()),
        patch("train.T5TokenizerFast.from_pretrained", return_value=MagicMock()),
        patch("train.FluxPipeline.from_pretrained", return_value=MagicMock()),
    ):

        # Set up command line arguments
        argv = ["train.py"] + minimal_args
        monkeypatch.setattr(sys, "argv", argv)

        args = train.parse_args()

        # Create trainer instance and verify it initializes
        trainer = train.Trainer(args)
        assert trainer.args == args
