import argparse
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.trainer import ModelManager, Trainer


# Dummy Classes & Fixtures
class DummyAccelerator:
    """A dummy accelerator to replace Hugging Face's Accelerator during tests."""

    def __init__(self, **kwargs):
        self.is_local_main_process = True
        self.device = "cpu"
        self.process_index = 0
        self.num_processes = 1
        self.mixed_precision = kwargs.get("mixed_precision", "no")
        self.use_distributed = False
        self.state = type("State", (), {})()
        self.state.deepspeed_plugin = None

    def init_trackers(self, *args, **kwargs):
        pass

    def profile(self, profile_kwargs):
        class DummyContextManager:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        return DummyContextManager()

    def prepare(self, *args):
        return args

    def accumulate(self, model):
        class DummyContextManager:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        return DummyContextManager()

    def backward(self, loss):
        pass

    def clip_grad_norm_(self, params, max_norm):
        pass

    def wait_for_everyone(self):
        pass

    def log(self, logs, step):
        pass

    def clear(self):
        pass

    def end_training(self):
        pass


# Automatically patch Accelerator in trainer.py for all tests.
@pytest.fixture(autouse=True)
def patch_accelerator(monkeypatch):
    monkeypatch.setattr(
        "src.trainer.Accelerator", lambda **kwargs: DummyAccelerator(**kwargs)
    )


@pytest.fixture
def dummy_args():
    """Returns a dummy argparse.Namespace with minimal required attributes."""
    return argparse.Namespace(
        gradient_accumulation_steps=1,
        report_to="tensorboard",
        logging_dir="./logs",
        seed=42,
        model="flux-dev",
        shift=1.0,
        use_gradient_checkpointing=False,
        use_lora=False,
        resolution=64,
        center_crop=True,
        cache_dir="./cache",
        train_data_path="./data",
        use_cache=False,
        train_batch_size=1,
        dataloader_num_workers=0,
        validation_prompts="dummy prompt",
        validation_guidance_scale=7.5,
        validation_inference_steps=10,
        output_dir="./outputs",
        num_iterations=1,
        validation_iteration=1,
        adam_epsilon=1e-8,
        learning_rate=1e-4,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_weight_decay=0.01,
        lr_warmup_steps=0,
        lr_scheduler="linear",
        max_grad_norm=1.0,
        profiling_step=-1,  # Disable profiling by default
        profiling_logging_dir="./profile_logs",
    )


# Tests for get_model
def test_get_model_flux(monkeypatch):
    """Test that get_model returns proper tuple for 'flux-dev'."""
    dummy_flux = MagicMock(name="FluxModel")
    monkeypatch.setattr("src.models.flux.FluxModel", dummy_flux)

    cls, model_type = ModelManager().get_model("flux-dev")
    assert cls == dummy_flux
    assert model_type == "image"


def test_get_model_hunyuan(monkeypatch):
    """Test that get_model returns proper tuple for 'hunyuan-video'."""
    dummy_hunyuan = MagicMock(name="HunyuanVideoModel")
    monkeypatch.setattr("src.models.hunyuan.HunyuanVideoModel", dummy_hunyuan)

    cls, model_type = ModelManager().get_model("hunyuan-video")
    assert cls == dummy_hunyuan
    assert model_type == "video"


def test_get_model_invalid():
    """Ensure get_model raises NotImplementedError for unsupported model names."""
    with pytest.raises(NotImplementedError):
        ModelManager().get_model("unsupported-model")
