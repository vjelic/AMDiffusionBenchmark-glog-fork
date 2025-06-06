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

MAX_SEQUENCE_LENGTH = 512


class Wan2_1_I2V_Model(BaseModel):
    """An implementation of the Wan2.1 diffusion model for video generation.

    This model extends BaseModel and incorporates Wan2.1 specific settings.
    Currently only supports image-to-video.

    Attributes:
        hf_id (str): The Hugging Face model identifier (default: "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers").
        loss_type (str): The loss function type ("flow_match" for this implementation).
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """Initializes the Wan2.1 img2vid with appropriate submodule configurations.

        The constructor sets up the diffusion pipeline for video generation
        and defines a custom prompt template to process incoming text prompts.

        Args:
            args (argparse.Namespace): The argument namespace containing model,
                training, and configuration parameters.
        """
        # Initial implementation of Wan2.1 model
        self.hf_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"

        super(Wan2_1_I2V_Model, self).__init__(
            args,
        )

        self.denoiser_config = copy.deepcopy(self.denoiser.config)

        original_tokenizer = self.submodules["tokenizer"]
        self.submodules["tokenizer"] = lambda prompt: original_tokenizer(
            prompt,
            max_length=MAX_SEQUENCE_LENGTH,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

    def init_attention(self):
        """Implement the missing attn_processors method"""
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

    def init_checkpointing(
        self,
        checkpointing_pct: float = 0.0,
        transformer_block_name: str = "blocks",
    ):
        """
        WanTransformer3DModel has a different `transformer_block_name = "blocks"`:
        usually it's called "transformer_blocks" in Diffusers
        """
        return super().init_checkpointing(
            checkpointing_pct=checkpointing_pct,
            transformer_block_name=transformer_block_name,
        )

    def encode_image(self, batch: dict) -> torch.Tensor:
        dtype = self.submodules["image_encoder"].dtype
        device = self.submodules["image_encoder"].device
        # Prepare image embeds
        # Ref: https://github.com/huggingface/diffusers/blob/d8c617ccb08a7d0d4127c0628b29de404133eda7/src/diffusers/pipelines/wan/pipeline_wan_i2v.py#L223-L231
        image = batch["processed_image"].to(device)  # first frame of the video
        image_embeds = self.submodules["image_encoder"](
            pixel_values=image, output_hidden_states=True
        )
        image_embeds = image_embeds.hidden_states[-2]

        # Prepare latents
        # Ref. https://github.com/huggingface/diffusers/blob/d8c617ccb08a7d0d4127c0628b29de404133eda7/src/diffusers/pipelines/wan/pipeline_wan_i2v.py#L610
        video = batch["pixel_values"].to(device=device, dtype=dtype)
        video = video.permute(
            0, 2, 1, 3, 4
        ).contiguous()  # [B, F, C, H, W] -> [B, C, F, H, W]
        batch_size = video.shape[0]
        width, height = self.args.resolution
        num_channels_latents = self.submodules["vae"].config.z_dim
        latents = self.submodules["vae"].encode(video).latent_dist.sample()
        latents, video_condition = self.pipe.prepare_latents(
            image=video[:, :, 0],  # first frame of the video
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=self.args.num_frames,
            dtype=dtype,
            device=device,
            latents=latents,
        )
        return (image_embeds, latents, video_condition)

    def encode_text(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = self.submodules["text_encoder"].dtype
        device = self.submodules["text_encoder"].device
        # Create T5 embeds
        # Ref. https://github.com/huggingface/diffusers/blob/723dbdd36300cd5a14000b828aaef87ba7e1fa68/src/diffusers/pipelines/wan/pipeline_wan_i2v.py#L182
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["input_mask"].to(device)
        seq_lens = attention_mask.gt(0).sum(dim=1).long()
        prompt_embeds = self.submodules["text_encoder"](
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(MAX_SEQUENCE_LENGTH - u.size(0), u.size(1))])
                for u in prompt_embeds
            ],
            dim=0,
        )
        return prompt_embeds.to(dtype)

    def encode_batch(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Encodings of input batch that are shared between models.

        This method processes the input batch by encoding images and obtaining
        prompt embeddings from text data. The output dictionary contains these
        encoded representations and retains any included input masks.
        """
        output: Dict[str, torch.Tensor] = {}
        (
            output["encoder_hidden_states_image"],
            output["latents"],
            output["condition"],
        ) = self.encode_image(batch)
        output["prompt_embeds"] = self.encode_text(batch)
        return output

    def get_model_inputs(
        self,
        batch: Dict[str, torch.tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Creates any additional conditionals required for the WanTransformer3DModel.forward:
        Ref. https://github.com/huggingface/diffusers/blob/b8093e6665859265d42a77878930fc0f54126219/src/diffusers/models/transformers/transformer_wan.py#L390-L398
        Args:
            batch (dict): Batch's data.

        Returns:
            - dict: A dictionary containing: 'hidden_states', 'encoder_hidden_states', and 'encoder_hidden_states_image'
            - torch.Tensor: Timesteps
        """
        output = {}
        output["hidden_states"] = torch.cat(
            [batch["latents"], batch["condition"]], dim=1
        )
        output["encoder_hidden_states"] = batch["prompt_embeds"]
        output["encoder_hidden_states_image"] = batch["encoder_hidden_states_image"]
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
