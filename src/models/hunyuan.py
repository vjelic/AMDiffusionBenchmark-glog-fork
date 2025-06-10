import argparse
import copy
from typing import Dict, Tuple

import torch

from src.models.base import BaseModel
from src.models.models_utils import compute_density_based_timestep_sampling


class HunyuanVideoModel(BaseModel):
    """An implementation of the Hunyuan flow-based diffusion model for video generation.

    This model extends BaseModel and incorporates HunyuanVideo-specific settings.
    It includes methods for tokenizing text prompts according to a custom
    prompt template, as well as for encoding video frames and text prompts
    into latent representations.

    Attributes:
        hf_id (str): The Hugging Face model identifier (default: "hunyuanvideo-community/HunyuanVideo").
        loss_type (str): The loss function type ("flow_match" for this implementation).
        guidance (float): Guidance value for conditional generation, set to a high default (1000.0).
        prompt_template (dict): Template for prompt formatting, including a truncation (`crop_start`) parameter.
    """

    def __init__(self, args: argparse.Namespace, is_training: bool = True) -> None:
        """Initializes the HunyuanVideoModel with appropriate submodule configurations.

        The constructor sets up the diffusion pipeline for video generation
        and defines a custom prompt template to process incoming text prompts.

        Args:
            args (argparse.Namespace): The argument namespace containing model,
                training, and configuration parameters.
        """
        # Initial implementation of Hunyuan-video model
        self.hf_id = "hunyuanvideo-community/HunyuanVideo"
        self.guidance = 1000.0
        super(HunyuanVideoModel, self).__init__(args, is_training)

        self.denoiser_config = copy.deepcopy(self.denoiser.config)

        self.prompt_template = {
            "template": (
                "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
                "1. The main content and theme of the video."
                "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
                "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
                "4. background environment, light, style and atmosphere."
                "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
            ),
            "crop_start": 95,
        }

        original_tokenizer = self.submodules["tokenizer"]
        original_tokenizer_2 = self.submodules["tokenizer_2"]

        # Create a tokenizer that only takes prompt as argument
        self.submodules["tokenizer"] = lambda prompt: original_tokenizer(
            [self.prompt_template["template"].format(p) for p in prompt],
            max_length=256 + self.prompt_template["crop_start"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_length=False,
            return_overflowing_tokens=False,
            return_attention_mask=True,
        )

        self.submodules["tokenizer_2"] = lambda prompt: original_tokenizer_2(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

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
        latents = (latents) * self.submodules["vae"].config.scaling_factor
        return latents

    def encode_text(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes text prompts into latent embeddings using two encoders.

        The first encoder (llama-like) uses a prompt template and an attention mask.
        The second encoder (CLIP-like) transforms text into pooled embeddings.

        Args:
            batch (dict): A dictionary containing:
                - "input_ids" (torch.Tensor): Input IDs for the first encoder.
                - "input_mask" (torch.Tensor): Attention mask for the first encoder.
                - "input_ids_2" (torch.Tensor): Input IDs for the second encoder.

        Returns:
            tuple:
                - torch.Tensor: A pooled embedding vector from the second text encoder (CLIP-like).
                - torch.Tensor: Hidden-state embeddings from the second-to-last layer
                of the first text encoder (llama-like), optionally cropped based on self.prompt_template["crop_start"].
        """
        # Create llama embeds
        prompt_embeds = self.submodules["text_encoder"](
            input_ids=batch["input_ids"].to(self.submodules["text_encoder"].device),
            attention_mask=batch["input_mask"],
            output_hidden_states=True,
        ).hidden_states[-3]
        prompt_embeds = prompt_embeds.to(dtype=self.submodules["text_encoder"].dtype)

        crop_start = self.prompt_template["crop_start"]
        if crop_start is not None and crop_start > 0:
            prompt_embeds = prompt_embeds[:, crop_start:]

        # Create CLIP embeds
        pooled_prompt_embeds = self.submodules["text_encoder_2"](
            batch["input_ids_2"].to(self.submodules["text_encoder_2"].device),
            output_hidden_states=False,
        ).pooler_output
        pooled_prompt_embeds = pooled_prompt_embeds.to(
            dtype=self.submodules["text_encoder_2"].dtype
        )
        return pooled_prompt_embeds, prompt_embeds

    def get_model_inputs(
        self,
        batch: Dict[str, torch.tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Creates any additional conditionals required for the HunyuanVideoTransformer3DModel.forward:
        https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_hunyuan_video.py#L1025
        Args:
            batch (dict): Contains "input_mask" which may be cropped based on the
                prompt_template["crop_start"] value.

        Returns:
            - dict: A dictionary containing: "hidden_states", "encoder_hidden_states", "pooled_projections", and "encoder_attention_mask"
            - torch.Tensor: Modified timesteps, multiplied by 1000.0 and cast to long.
        """
        output = {}
        output["hidden_states"] = batch["noised_latents"]
        output["encoder_hidden_states"] = batch["prompt_embeds"]
        output["pooled_projections"] = batch["pooled_prompt_embeds"]
        # Get the additional prompt attention mask conditional
        prompt_attention_mask = batch["input_mask"].to(device=self.denoiser.device)
        crop_start = self.prompt_template["crop_start"]
        if crop_start is not None and crop_start > 0:
            prompt_attention_mask = prompt_attention_mask[:, crop_start:]

        output["encoder_attention_mask"] = prompt_attention_mask
        return output, (batch["timestep"] * 1000.0).long()

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
