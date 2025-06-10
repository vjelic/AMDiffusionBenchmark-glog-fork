import argparse
from typing import Dict, Tuple

import diffusers.training_utils
import torch

from src.models.base import BaseModel
from src.models.models_utils import compute_uniform_timestep_sampling


class StableDiffusionXLModel(BaseModel):
    """
    A specialized Diffusers model integration for Stable Diffusion XL flow-based diffusion.

    This model extends BaseModel to support SDXL-specific configuration details, including:
      - Custom latent image ID generation.
      - Image encoding via the VAE.
      - Dual text encoding using two text encoders.
      - Additional conditionals for the diffusion process.

    Attributes:
        hf_id (str): Hugging Face model identifier ("stabilityai/stable-diffusion-xl-base-1.0").
        loss_type (str): Loss function type (set to "denoising_score_").
        guidance (float): Guidance strength (set to 1.0).
        vae_scale_factor (int): Derived from the VAE configuration.
        height (int): Computed latent height.
        width (int): Computed latent width.
        num_channels_latents (int): Number of latent channels.
    """

    def __init__(self, args: argparse.Namespace, is_training: bool = True) -> None:
        """
        Initializes the StableDiffusionXLModel with SDXL-specific settings.

        Args:
            args (argparse.Namespace): Configuration namespace containing paths, hyperparameters,
                                       and other settings.
        """
        # Set model-specific parameters.
        self.hf_id = "stabilityai/stable-diffusion-xl-base-1.0"
        self.guidance = None

        super(StableDiffusionXLModel, self).__init__(args, is_training)

        # Calculate VAE scale factor and derive dimensions.
        self.vae_scale_factor = 2 ** (
            len(self.submodules["vae"].config.block_out_channels) - 1
        )
        self.height = 2 * (int(self.args.resolution[1]) // (self.vae_scale_factor * 2))
        self.width = 2 * (int(self.args.resolution[0]) // (self.vae_scale_factor * 2))

        # Override tokenizers to enforce consistent input formatting.
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
            max_length=original_tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
            return_overflowing_tokens=False,
            return_length=False,
        )

    def encode_image(self, batch: dict) -> torch.Tensor:
        """
        Encode input images into latent representations using the VAE.

        The process involves:
          - Passing the pixel values through the VAE encoder.
          - Adjusting by the VAE's shift and scaling factors.

        Args:
            batch (dict): Dictionary containing 'pixel_values' with the input images.

        Returns:
            torch.Tensor: Latent representations of shape (batch_size, channels, height, width).
        """
        vae = self.submodules["vae"]
        latents = vae.encode(batch["pixel_values"].to(vae.dtype)).latent_dist.sample()
        latents = (latents - (vae.config.shift_factor or 0)) * vae.config.scaling_factor

        return latents

    def encode_text(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text prompts using two text encoders.

        The first encoder produces pooled embeddings while the second produces token-level embeddings.

        Args:
            batch (dict): Dictionary containing:
                - "input_ids": Token IDs for the first text encoder.
                - "input_ids_2": Token IDs for the second text encoder.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Pooled prompt embeddings (first encoder).
                - Token-level prompt embeddings (second encoder).
        """
        # Token-level embeddings from the first text encoder.
        text_encoder = self.submodules["text_encoder"]
        prompt_embeds = text_encoder(batch["input_ids"], output_hidden_states=False)[0]

        # Pooled text embeddings from the second text encoder.
        text_encoder_2 = self.submodules["text_encoder_2"]
        pooled_prompt_embeds = text_encoder_2(
            batch["input_ids_2"], output_hidden_states=False
        )[0]

        prompt_embeds = torch.concat(
            [
                prompt_embeds,
                pooled_prompt_embeds[:, None, :].repeat(1, prompt_embeds.shape[1], 1),
            ],
            dim=-1,
        )

        prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=text_encoder_2.dtype)

        return pooled_prompt_embeds, prompt_embeds

    def get_model_inputs(
        self,
        batch: Dict[str, torch.tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Creates any additional conditionals required for the StableDiffusionXL model.
        https://github.com/huggingface/diffusers/blob/97fda1b75c70705b245a462044fedb47abb17e56/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L1213C39-L1213C50
        Args:
            batch (dict): Batch's data.

        Returns:
            - dict: A dictionary containing: "hidden_states", "encoder_hidden_states", "pooled_projections", "img_ids", and "txt_ids"
            - torch.Tensor: Timesteps
        """
        output = {}
        output["sample"] = batch["noised_latents"]
        output["encoder_hidden_states"] = batch["prompt_embeds"]
        pooled_prompt_embeds = batch["pooled_prompt_embeds"]
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        # TODO: These should come from the dataset's metadata
        time_ids = self.pipe._get_add_time_ids(
            (self.width, self.height),
            (0, 0),
            (self.width, self.height),
            dtype=pooled_prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

        batch_size = pooled_prompt_embeds.shape[0]
        time_ids = time_ids[None, :, :].repeat(batch_size, 1, 1)
        output["added_cond_kwargs"] = {
            "text_embeds": pooled_prompt_embeds,
            "time_ids": time_ids,
        }
        return output, batch["timestep"]

    def compute_loss(
        self, pred, noise, latents=None, timesteps=None, snr_gamma: float = 5.0
    ):
        """Compute the loss for the model.

        Args:
            pred (torch.Tensor): Model output.
            noise (torch.Tensor): Sampled nise.
            latents (torch.Tensor): Model input.
            timesteps (torch.Tensor): Timesteps for the diffusion process.
            snr_gamma (float): Min-SNR hyperparameter.
        Returns:
            torch.Tensor: The computed loss.

        """

        # Ref. https://github.com/huggingface/diffusers/blob/4e3ddd5afab3a4b0b6265f210d6710933dade660/examples/text_to_image/train_text_to_image.py#L1012
        snr = diffusers.training_utils.compute_snr(
            self.submodules["scheduler"], timesteps
        )
        mse_loss_weights = torch.stack(
            [snr, snr_gamma * torch.ones_like(timesteps)], dim=1
        ).min(dim=1)[0]

        prediction_type = self.submodules["scheduler"].config.prediction_type

        if prediction_type == "epsilon":
            mse_loss_weights = mse_loss_weights / snr
        elif prediction_type == "v_prediction":
            mse_loss_weights = mse_loss_weights / (snr + 1)
        else:
            raise ValueError(f"Unknown prediction_type {prediction_type}!")

        loss = torch.nn.functional.mse_loss(
            pred.float(), noise.float(), reduction="none"
        )
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
        loss = loss.mean()
        return loss

    def sample_timesteps_and_noise(
        self,
        latents: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample timesteps and apply noise to latents using the scheduler."""
        return compute_uniform_timestep_sampling(
            self.submodules["scheduler"], latents, self.sigmas
        )
