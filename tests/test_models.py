import argparse
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.models.base import BaseModel
from src.models.flux import FluxModel
from src.models.hunyuan import HunyuanVideoModel
from src.models.models_utils import compute_density_based_timestep_sampling


# Helper classes to simulate encoder outputs
class DummyEncodeResult:
    """Simulate the result of a VAE encode call."""

    class DummyLatentDist:
        def __init__(self, latent_tensor):
            self.latent_tensor = latent_tensor

        def sample(self):
            return self.latent_tensor

    def __init__(self, latent_tensor):
        self.latent_dist = DummyEncodeResult.DummyLatentDist(latent_tensor)


class DummyTextEncoderOutput:
    """Simulate a text encoder output with a pooler_output attribute."""

    def __init__(self, tensor):
        self.pooler_output = tensor


class DummyTextEncoderHidden:
    """Simulate a text encoder output for Hunyuan that has hidden_states."""

    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


@pytest.fixture
def dummy_args():
    """Returns an argparse.Namespace with dummy testing arguments."""
    return argparse.Namespace(
        model_path="some/fake-model-path",
        train_data_path="path/to/fake-data",
        num_iterations=1,
        train_batch_size=1,
        output_dir="./outputs",
        resolution=(512, 512),
        shift=3.0,
        substitute_sdpa_with_flash_attn=False,
        gradient_checkpointing=0.0,
    )


@pytest.fixture
def dummy_pipeline():
    """Returns a dummy pipeline object with required submodules."""
    dummy = MagicMock()
    # Create a dummy denoiser with an attn_processors method.
    dummy.denoiser = MagicMock()
    dummy.denoiser.attn_processors = {}
    dummy.vae = MagicMock()
    dummy.text_encoder = MagicMock()
    return dummy


@pytest.fixture(autouse=True)
def patch_diffusers_and_pipeline_map(dummy_pipeline):
    """
    Patches the diffusers pipelines and pipeline_map so that model lookups succeed.
    The keys are set to what the models expect:
      - FluxModel: "black-forest-labs/FLUX.1-dev"
      - HunyuanVideoModel: "hunyuanvideo-community/HunyuanVideo"
    """
    with (
        patch(
            "src.models.base.diffusers.FluxPipeline.from_pretrained",
            return_value=dummy_pipeline,
        ),
        patch(
            "src.models.base.diffusers.HunyuanVideoPipeline.from_pretrained",
            return_value=dummy_pipeline,
        ),
        patch(
            "src.models.base.pipeline_map",
            {
                "black-forest-labs/FLUX.1-dev": MagicMock(
                    from_pretrained=MagicMock(return_value=dummy_pipeline)
                ),
                "hunyuanvideo-community/HunyuanVideo": MagicMock(
                    from_pretrained=MagicMock(return_value=dummy_pipeline)
                ),
            },
        ),
        patch.object(
            FluxModel,
            "hf_id",
            "black-forest-labs/FLUX.1-dev",
            create=True,
        ),
        patch.object(
            HunyuanVideoModel,
            "hf_id",
            "hunyuanvideo-community/HunyuanVideo",
            create=True,
        ),
    ):
        yield


# -----------------------------------------------------------------------------
# BaseModel Tests
# -----------------------------------------------------------------------------
def test_base_model_init(dummy_args):
    """
    Tests BaseModel initialization by bypassing module initialization.
    We patch init_modules_from_pipeline so that a dummy denoiser (with attn_processors)
    is set and the call in the constructor doesn't fail.
    """
    # Create a mock scheduler with necessary attributes
    mock_scheduler = MagicMock()
    mock_scheduler.timesteps = MagicMock()
    mock_scheduler.config.num_train_timesteps = MagicMock()

    def init_modules_mock(self):
        self.denoiser = MagicMock(attn_processors={})
        self.submodules = {"scheduler": mock_scheduler}

    with patch.object(BaseModel, "init_modules_from_pipeline", new=init_modules_mock):
        model = BaseModel(dummy_args)

    assert model.args is dummy_args
    # Since we bypassed normal module initialization, pipe remains None.
    assert model.pipe is None


def test_base_model_init_modules(dummy_args, dummy_pipeline):
    """
    Tests module initialization in BaseModel.
    Here we let init_modules_from_pipeline run and then simulate extraction of submodules.
    """
    with patch.object(BaseModel, "hf_id", "black-forest-labs/FLUX.1-dev", create=True):
        model = BaseModel(dummy_args)
    # Simulate that init_modules_from_pipeline sets submodules.
    model.submodules["denoiser"] = model.pipe.denoiser
    assert "denoiser" in model.submodules


# -----------------------------------------------------------------------------
# FluxModel Tests
# -----------------------------------------------------------------------------
def test_flux_model_init(dummy_args):
    """
    Tests FluxModel initialization.
    """
    flux_model = FluxModel(dummy_args)
    assert flux_model.args is dummy_args


def test_flux_model_latent_ids(dummy_args):
    """
    Tests FluxModel latent ID preparation.
    """
    flux_model = FluxModel(dummy_args)

    # Set attributes needed for the view operation.
    flux_model.num_channels_latents = 3
    flux_model.height = 256
    flux_model.width = 256

    dummy_latent = torch.zeros(1, 3, 256, 256)
    dummy_vae = MagicMock()
    dummy_vae.dtype = torch.float32
    dummy_vae.encode.return_value = DummyEncodeResult(dummy_latent)

    # Set dummy VAE configuration with shift and scaling factors.
    dummy_config = type("DummyConfig", (), {})()
    dummy_config.shift_factor = 0.0
    dummy_config.scaling_factor = 1.0
    dummy_vae.config = dummy_config

    flux_model.submodules = {"vae": dummy_vae}
    latent_ids = flux_model.prepare_latent_image_ids(256, 256, "cpu", torch.float32)

    assert isinstance(latent_ids, torch.Tensor)
    # Flatten the spatial dimensions.
    assert latent_ids.shape == (256 * 256, 3)


def test_flux_model_encode_image(dummy_args):
    """
    Tests FluxModel image encoding.
    """
    flux_model = FluxModel(dummy_args)
    flux_model.num_channels_latents = 3
    flux_model.height = 256
    flux_model.width = 256

    dummy_latent = torch.zeros(1, 3, 256, 256)

    dummy_vae = MagicMock()
    dummy_vae.dtype = torch.float32
    dummy_vae.encode.return_value = DummyEncodeResult(dummy_latent)

    dummy_config = type("DummyConfig", (), {})()
    dummy_config.shift_factor = 0.0
    dummy_config.scaling_factor = 1.0

    dummy_vae.config = dummy_config
    flux_model.submodules = {"vae": dummy_vae}

    batch = {"pixel_values": torch.zeros(1, 3, 256, 256)}
    enc = flux_model.encode_image(batch)

    assert isinstance(enc, torch.Tensor)
    assert enc.shape == (1, (256 // 2) * (256 // 2), 2 * 2 * 3)


def test_flux_model_encode_text(dummy_args):
    """
    Tests FluxModel text encoding.
    """
    flux_model = FluxModel(dummy_args)

    # Create a dummy text encoder output (with a pooler_output attribute).
    dummy_text_encoder = MagicMock(
        return_value=DummyTextEncoderOutput(torch.zeros(1, 768))
    )
    dummy_text_encoder.dtype = torch.float32

    # Provide a dummy second text encoder (returns a tuple).
    dummy_text_encoder_2 = MagicMock(return_value=(torch.zeros(1, 768),))
    dummy_text_encoder_2.dtype = torch.float32
    flux_model.submodules = {
        "text_encoder": dummy_text_encoder,
        "text_encoder_2": dummy_text_encoder_2,
    }

    # Wrap the text encoders.
    batch = {
        "input_ids": torch.randint(0, 100, (1, 10)),
        "input_ids_2": torch.randint(0, 100, (1, 10)),
    }

    text_emb, pooled = flux_model.encode_text(batch)

    assert isinstance(text_emb, torch.Tensor)
    assert isinstance(pooled, torch.Tensor)
    assert text_emb.shape == (1, 768)
    assert pooled.shape == (1, 768)


# -----------------------------------------------------------------------------
# HunyuanVideoModel Tests
# -----------------------------------------------------------------------------
def test_hunyuan_video_model_init(dummy_args):
    """
    Tests HunyuanVideoModel initialization.
    """
    model = HunyuanVideoModel(dummy_args)
    assert model.args is dummy_args


def test_hunyuan_video_model_encode_image(dummy_args):
    """
    Tests video frame encoding in HunyuanVideoModel.
    """
    model = HunyuanVideoModel(dummy_args)
    dummy_latent = torch.zeros(1, 3, 256, 256)

    dummy_vae = MagicMock()
    dummy_vae.dtype = torch.float32
    dummy_vae.encode.return_value = DummyEncodeResult(dummy_latent)

    dummy_config = type("DummyConfig", (), {})()
    dummy_config.shift_factor = 0.0
    dummy_config.scaling_factor = 1.0
    dummy_vae.config = dummy_config

    model.submodules = {"vae": dummy_vae}

    batch = {"pixel_values": torch.zeros(1, 4, 3, 256, 256)}
    enc = model.encode_image(batch)

    assert isinstance(enc, torch.Tensor)
    assert enc.shape == (1, 3, 256, 256)


def test_hunyuan_video_model_encode_text(dummy_args):
    """
    Tests text encoding in HunyuanVideoModel.
    """
    model = HunyuanVideoModel(dummy_args)
    dummy_text_encoder = MagicMock()
    dummy_text_encoder.device = torch.device("cpu")
    dummy_text_encoder.dtype = torch.float32

    # First text encoder output with hidden_states.
    hidden = [torch.zeros(1, 100, 768) for _ in range(4)]
    dummy_output = DummyTextEncoderHidden(hidden)
    dummy_text_encoder.return_value = dummy_output

    # Second text encoder.
    dummy_text_encoder_2 = MagicMock(
        return_value=DummyTextEncoderOutput(torch.zeros(1, 768))
    )
    dummy_text_encoder_2.device = torch.device("cpu")
    dummy_text_encoder_2.dtype = torch.float32

    # Wrap the text encoders.
    model.submodules = {
        "text_encoder": dummy_text_encoder,
        "text_encoder_2": dummy_text_encoder_2,
    }
    batch = {
        "input_ids": torch.randint(0, 100, (1, 10)),
        "input_mask": torch.ones(1, 10, dtype=torch.int),
        "input_ids_2": torch.randint(0, 100, (1, 10)),
    }

    text_emb, pooled = model.encode_text(batch)
    assert isinstance(text_emb, torch.Tensor)
    assert isinstance(pooled, torch.Tensor)
    assert text_emb.shape == (1, 768)
    assert pooled.shape == (1, 5, 768)


class DummyScheduler:
    def __init__(self):
        # Create a dummy scheduler with a linear timesteps array.
        self.config = type("DummyConfig", (), {})()
        self.config.num_train_timesteps = 10_000
        # Linear schedule from 1.0 down to 0.0 over 10,000 steps.
        self.timesteps = torch.linspace(1.0, 0.0, steps=10_000)


class DummyEnv:
    # Create a dummy environment that acts as "self" for the function.
    submodules = {"scheduler": DummyScheduler()}


def test_compute_density_based_timestep_sampling():
    """Test that compute_density_based_timestep_sampling produces correctly noised latents, noise, and normalized timesteps."""
    num_timesteps = 10_000
    env = DummyEnv()

    # Create a strictly 1D sigma schedule from the dummy scheduler.
    sigmas = env.submodules["scheduler"].timesteps.clone().view(-1)

    # Create a dummy latents tensor (all zeros) with shape (batch_size, channels, height, width).
    latents = torch.zeros((2, 2, 4, 4))

    # Call the imported compute_density_based_timestep_sampling function,
    # passing env as self.
    noised_latents, noise, normalized_timesteps = (
        compute_density_based_timestep_sampling(
            env.submodules["scheduler"], latents, sigmas, logit_mean=0, logit_std=0
        )
    )

    # Basic shape checks.
    assert noised_latents.shape == latents.shape, "Noised latents shape mismatch"
    assert noise.shape == latents.shape, "Noise shape mismatch"
    assert (
        normalized_timesteps.shape[0] == latents.shape[0]
    ), "Timesteps batch size mismatch"

    # Expected sigma is the middle value of the scheduler.
    expected_index = int(0.5 * num_timesteps)
    expected_sigma = sigmas[expected_index]

    # For zero latents, noised_latents should equal sigma * noise (since (1-sigma)*0 = 0).
    assert torch.allclose(
        noised_latents, expected_sigma * noise, atol=1e-6
    ), "Noised latents do not match expected sigma-scaled noise."

    # Check normalized timesteps: raw is dummy_scheduler.timesteps[expected_index]
    # method returns timesteps / num_train_timesteps
    expected_timestep = (
        env.submodules["scheduler"].timesteps[expected_index] / num_timesteps
    )
    # Check the first sample's normalized timestep.
    assert torch.allclose(
        normalized_timesteps[0], expected_timestep.detach().clone(), atol=1e-4
    ), "Normalized timestep value is incorrect."
