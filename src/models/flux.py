import argparse
from typing import Dict, Tuple, Union

import torch

from src.models.base import BaseModel
from src.models.models_utils import compute_density_based_timestep_sampling


class FluxModel(BaseModel):
    """A specialized Diffusers model for FLUX.1 flow-based diffusion.

    This model extends BaseModel to handle specific configuration details and
    behaviors for the FLUX.1 diffusion workflow. It sets up a custom resolution,
    encodes images and text, and constructs latent image IDs suitable for flow matching.

    Attributes:
        hf_id (str): The Hfenugging Face model identifier (e.g., "black-forest-labs/FLUX.1-dev").
        loss_type (str): The type of loss function used for training or inference (e.g., "flow_match").
        guidance (float): Guidance strength for conditioning the model outputs.
        vae_scale_factor (int): Scale factor derived from the VAE configuration.
        height (int): Internal height setting, computed for FLUX.1 latents.
        width (int): Internal width setting, computed for FLUX.1 latents.
        num_channels_latents (int): Number of channels in the latents for this model.
    """

    def __init__(self, args: argparse.Namespace, is_training: bool = True) -> None:
        """Initializes the FluxModel with default FLUX.1-specific settings.

        Args:
            args (argparse.Namespace): Configuration namespace containing
                paths, hyperparameters, and other settings.
        """
        # Model details
        self.hf_id = "black-forest-labs/FLUX.1-dev"
        self.guidance = 1.0

        super(FluxModel, self).__init__(args, is_training)

        # VAE config
        self.vae_scale_factor = 2 ** (
            len(self.submodules["vae"].config.block_out_channels) - 1
        )
        width, height = self.args.resolution
        self.height = 2 * (height // (self.vae_scale_factor * 2))
        self.width = 2 * (width // (self.vae_scale_factor * 2))
        self.num_channels_latents = self.denoiser.config.in_channels // 4

        original_tokenizer = self.submodules["tokenizer"]
        original_tokenizer_2 = self.submodules["tokenizer_2"]

        self.submodules["tokenizer"] = lambda prompt: original_tokenizer(
            prompt,
            max_length=original_tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_overflowing_tokens=False,
            return_length=False,
        )

        self.submodules["tokenizer_2"] = lambda prompt: original_tokenizer_2(
            prompt,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )

    def prepare_latent_image_ids(
        self,
        height: int,
        width: int,
        device: Union[torch.device, str],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create image ID embeddings for the FLUX.1 model.

        Args:
            batch_size (int): The number of samples in the batch (unused in this method but may be required for consistency).
            height (int): The height of the image grid.
            width (int): The width of the image grid.
            device (torch.device): The device on which to place the tensor (e.g., 'cpu' or 'cuda').
            dtype (torch.dtype): The data type of the tensor (e.g., torch.float32).

        Returns:
            torch.Tensor: A tensor of shape (height * width, 3) containing the latent image IDs,
            moved to the specified device and cast to the specified data type.
        """
        latent_image_ids = torch.zeros(height, width, 3, device=device, dtype=dtype)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1]
            + torch.arange(height, device=device, dtype=dtype)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2]
            + torch.arange(width, device=device, dtype=dtype)[None, :]
        )

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = (
            latent_image_ids.shape
        )

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        ).contiguous()

        return latent_image_ids

    def encode_image(self, batch: dict) -> torch.Tensor:
        """Encode images into latents and generate corresponding image IDs.

        Args:
            batch (dict): Batch containing 'pixel_values' of images.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - latents: Encoded latent representations of the images.
                - image_ids: Image IDs associated with the latents.
        """
        # Encode image
        latents = (
            self.submodules["vae"]
            .encode(batch["pixel_values"].to(self.submodules["vae"].dtype))
            .latent_dist.sample()
        )
        latents = (
            latents - self.submodules["vae"].config.shift_factor
        ) * self.submodules["vae"].config.scaling_factor
        batch_size = latents.shape[0]

        # Pack the latents ...?
        latents = latents.view(
            batch_size,
            self.num_channels_latents,
            self.height // 2,
            2,
            self.width // 2,
            2,
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(
            batch_size,
            (self.height // 2) * (self.width // 2),
            self.num_channels_latents * 4,
        ).contiguous()

        return latents

    def encode_text(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode text inputs into embeddings.

        Args:
            batch (dict): A batch containing 'input_ids' and 'input_ids_2'.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - pooled_prompt_embeds: Embeddings from the first text encoder.
                - prompt_embeds: Embeddings from the second text encoder.
                - text_ids: Additional embeddings tensor.
        """
        # Compute text embeddings and hidden states
        pooled_prompt_embeds = self.submodules["text_encoder"](
            batch["input_ids"], output_hidden_states=False
        )
        pooled_prompt_embeds = pooled_prompt_embeds.pooler_output
        pooled_prompt_embeds = pooled_prompt_embeds.to(
            dtype=self.submodules["text_encoder"].dtype
        )

        prompt_embeds = self.submodules["text_encoder_2"](
            batch["input_ids_2"], output_hidden_states=False
        )[0]
        prompt_embeds = prompt_embeds.to(dtype=self.submodules["text_encoder_2"].dtype)

        return pooled_prompt_embeds, prompt_embeds

    def get_model_inputs(
        self,
        batch: Dict[str, torch.tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Creates any additional conditionals required for the FluxTransformer2DModel.forward:
        https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux.py#L389
        Args:
            batch (dict): Batch's data.

        Returns:
            - dict: A dictionary containing: "hidden_states", "encoder_hidden_states", "pooled_projections", "img_ids", and "txt_ids"
            - torch.Tensor: Timesteps
        """
        output = {}
        output["hidden_states"] = batch["noised_latents"]
        output["encoder_hidden_states"] = batch["prompt_embeds"]
        output["pooled_projections"] = batch["pooled_prompt_embeds"]
        output["img_ids"] = self.prepare_latent_image_ids(
            self.height // 2,
            self.width // 2,
            device=self.denoiser.device,
            dtype=self.denoiser.dtype,
        )
        output["txt_ids"] = torch.zeros(
            batch["prompt_embeds"].shape[1],
            3,
            device=self.denoiser.device,
            dtype=self.denoiser.dtype,
        )
        return output, batch["timestep"]

    def compute_loss(self, pred, noise, latents=None, timesteps=None):
        """Compute the loss for the model.

        Args:
            pred (torch.Tensor): Model output.
            noise (torch.Tensor): Sampled nise.
            latents (torch.Tensor): Model input.
            timesteps (torch.Tensor): Timesteps for the diffusion process.
        Returns:
            torch.Tensor: The computed loss.

        """
        return torch.nn.functional.mse_loss(
            pred.float(),
            (noise - latents).float(),
            reduction="mean",
        )

    def sample_timesteps_and_noise(
        self,
        latents: torch.Tensor,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample timesteps and apply noise using density-based timestep sampling."""
        return compute_density_based_timestep_sampling(
            self.submodules["scheduler"],
            latents,
            self.sigmas,
            logit_mean,
            logit_std,
        )
