import argparse
import copy
import types
from typing import Dict, Tuple

import torch

from src.models.base import BaseModel
from src.models.models_utils import (
    compute_density_based_timestep_sampling,
    get_attention_processors,
    set_attention_processors,
)


class MochiModel(BaseModel):
    """An implementation of the Mochi-1 diffusion model for video generation.

    This model extends BaseModel and incorporates Mochi-1 (preview) specific settings.
    It includes methods for tokenizing text prompts according to a custom
    prompt template, as well as for encoding video frames and text prompts
    into latent representations.

    Attributes:
        hf_id (str): The Hugging Face model identifier (default: "genmo/mochi-1-preview").
        loss_type (str): The loss function type ("flow_match" for this implementation).
    """

    def __init__(self, args: argparse.Namespace, is_training: bool = True) -> None:
        """Initializes the Mochi-1 with appropriate submodule configurations.

        The constructor sets up the diffusion pipeline for video generation
        and defines a custom prompt template to process incoming text prompts.

        Args:
            args (argparse.Namespace): The argument namespace containing model,
                training, and configuration parameters.
        """
        # Initial implementation of Mochi-1 model
        self.hf_id = "genmo/mochi-1-preview"

        super(MochiModel, self).__init__(args, is_training)

        self.denoiser_config = copy.deepcopy(self.denoiser.config)
        original_tokenizer = self.submodules["tokenizer"]

        # Create a tokenizer that only takes prompt as argument
        self.submodules["tokenizer"] = lambda prompt: original_tokenizer(
            prompt,
            padding="max_length",
            max_length=256,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        self.latents_mean = torch.tensor(
            self.submodules["vae"].config.latents_mean
        ).view(1, 12, 1, 1, 1)
        self.latents_std = torch.tensor(self.submodules["vae"].config.latents_std).view(
            1, 12, 1, 1, 1
        )

    def init_attention(self):
        # Implement the attn_processors method if missing
        if not self.denoiser:
            raise RuntimeError("self.denoiser is not initialized.")

        # Check and setup attn_processors property
        if not hasattr(self.denoiser.__class__, "attn_processors"):

            def attn_processors(instance):
                # This allows the property to access the correct instance's method
                return get_attention_processors(instance)

            # Assign it as a property to the class
            setattr(
                self.denoiser.__class__, "attn_processors", property(attn_processors)
            )

        # Implement the set_attn_processor method if missing
        if not hasattr(self.denoiser, "set_attn_processor"):

            def set_attn_processor(self, processor_dict):
                # Use the helper to set all attention processors
                set_attention_processors(processor_dict, self)

            # Assign the method to the instance
            self.denoiser.set_attn_processor = types.MethodType(
                set_attn_processor, self.denoiser
            )

        return super().init_attention()

    def encode_image(self, batch: dict) -> torch.Tensor:
        """Encodes video frames into latent representations.

        The input frames are transposed to the shape (batch, channels, frames, height, width)
        before being processed by the VAE submodule.

        Args:
            batch (dict): A dictionary containing "pixel_values" of shape [B, F, C, H, W],
                where B is batch size, F is number of frames, C is number of channels,
                H and W are image height and width.

        Returns:
            torch.Tensor: The encoded latents after factoring in the scaling factor
            from the VAE configuration.
        """
        # Encode image, [B, F, C, H, W] -> [B, C, F, H, W]
        video = batch["pixel_values"].permute(0, 2, 1, 3, 4).contiguous()
        latents = (
            self.submodules["vae"]
            .encode(video.to(self.submodules["vae"].dtype))
            .latent_dist.sample()
        )
        return latents * self.submodules["vae"].config.scaling_factor

    def encode_text(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes text prompts into latent embeddings using two encoders.

        The first encoder (llama-like) uses a prompt template and an attention mask.
        The second encoder (CLIP-like) transforms text into pooled embeddings.

        Args:
            batch (dict): A dictionary containing:
                - "input_ids" (torch.Tensor): Input IDs for the first encoder.
                - "input_mask" (torch.Tensor): Attention mask for the first encoder.

        Returns:
            tuple:
                - torch.Tensor: A pooled embedding vector from the second text encoder (CLIP-like).
                - torch.Tensor: Hidden-state embeddings from the second-to-last layer
                of the first text encoder (llama-like), optionally cropped based on self.prompt_template["crop_start"].
        """
        # Create T5 embeds
        prompt_embeds = self.submodules["text_encoder"](
            batch["input_ids"], attention_mask=batch["input_mask"]
        )[0]
        prompt_embeds = prompt_embeds.to(dtype=self.submodules["text_encoder"].dtype)

        return None, prompt_embeds

    def encode_batch(self, batch: Dict[str, torch.Tensor]):
        batch = super().encode_batch(batch)
        batch.pop(
            "pooled_prompt_embeds"
        )  # this is None, because tokenizer2 doesn't exist
        return batch

    def get_model_inputs(
        self,
        batch: Dict[str, torch.tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Creates any additional conditionals required for the MochiTransformer3DModel.forward:
        https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_mochi.py#L407

        Args:
            batch (dict): Batch's data.

        Returns:
            - dict: A dictionary containing: "hidden_states", "encoder_hidden_states", and "encoder_attention_mask"
            - torch.Tensor: Timesteps
        """
        output = {}
        output["hidden_states"] = batch["noised_latents"]
        output["encoder_hidden_states"] = batch["prompt_embeds"]
        # Get the additional prompt attention mask conditional
        prompt_attention_mask = batch["input_mask"].to(device=self.denoiser.device)

        output["encoder_attention_mask"] = prompt_attention_mask
        return output, 1000 * (1 - batch["timestep"])

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
        denoised_latents = (latents - noise).float()
        return torch.nn.functional.mse_loss(
            pred.float(),
            (
                denoised_latents
                - self.latents_mean.to(denoised_latents.device, denoised_latents.dtype)
            )
            / self.latents_std.to(denoised_latents.device, denoised_latents.dtype),
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
