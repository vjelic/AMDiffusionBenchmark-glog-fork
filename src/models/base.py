import argparse
import copy
from typing import Any, Dict, Tuple, Union

import diffusers
import torch
from peft import LoraConfig

from src.models.attention import replacement_processors
from src.utils import configure_logging

logger = configure_logging()

# A map from hf_id to appropriate huggingface pipeline
pipeline_map = {
    "black-forest-labs/FLUX.1-dev": diffusers.FluxPipeline,
    "hunyuanvideo-community/HunyuanVideo": diffusers.HunyuanVideoPipeline,
    "stabilityai/stable-diffusion-xl-base-1.0": diffusers.StableDiffusionXLPipeline,
    "genmo/mochi-1-preview": diffusers.MochiPipeline,
    "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers": diffusers.WanImageToVideoPipeline,
}


class BaseModel:
    """A base model class for image and video diffusion using the Diffusers library.

    **Any model that inherits this class must implement the model-specific encodings in:**
    - `encode_image`: image encodings
    - `encode_text`: text encodings
    - `get_model_inputs`: final postprocessing applied on the batch


    This class provides:
    - Initialization of the main diffusion pipeline
    - Loading of model submodules (VAE, tokenizers, text encoders, scheduler)
    - Overriding default attention processors with custom, FAv3-compatible ones
    - Methods for LoRA integration, unloading submodules to CPU, and moving
    submodules to the specified device and dtype

    Attributes:
        args (argparse.Namespace): Command line or programmatic configuration.
        hf_id (str): Identifier for the Hugging Face model repository to be used.
        loss_type (str): Defines the loss function type to be used for training or inference.
        guidance (Any): Configuration or settings related to guidance or conditioning.
        pipe (DiffusionPipeline): Diffusers pipeline containing the main modules.
        denoiser (nn.Module): The primary denoising or transformer module from the pipeline.
        submodules (dict): Collection of submodules to be loaded from the pipeline.
    """

    def __init__(self, args: argparse.Namespace, is_training: bool = True) -> None:
        """Initializes the BaseModel.

        Args:
            args (argparse.Namespace): Namespace containing all user-defined arguments,
                including paths, hyperparameters, and configuration options.
        """
        # Basic model initialization
        self.args = args
        self.hf_id = self.hf_id if hasattr(self, "hf_id") else None
        self.loss_type = self.loss_type if hasattr(self, "loss_type") else None
        self.guidance = self.guidance if hasattr(self, "guidance") else None
        self.pipe = None
        self.denoiser = None
        if not hasattr(self, "submodules"):
            self.submodules = {
                "vae": None,
                "text_encoder": None,
                "text_encoder_2": None,
                "tokenizer": None,
                "tokenizer_2": None,
                "image_processor": None,
                "image_encoder": None,
                "scheduler": None,
            }

        self.init_modules_from_pipeline()
        self.init_attention()
        if is_training:
            self.init_training()

    def init_modules_from_pipeline(self) -> None:
        """Initializes the modules from the Diffusers pipeline.

        The pipeline is loaded from the Hugging Face repository or from
        a local path specified in self.args.model_path. The corresponding
        submodules are then extracted and stored in self.submodules.
        """
        # Download all parts of the model
        self.pipe = pipeline_map[self.hf_id].from_pretrained(
            self.args.model_path
            if getattr(self.args, "model_path", None)
            else self.hf_id
        )

        # Get all the individual modules
        for name in self.submodules.keys():
            if hasattr(self.pipe, name):
                self.submodules[name] = getattr(self.pipe, name)

        self.denoiser = (
            self.pipe.transformer
            if hasattr(self.pipe, "transformer")
            else self.pipe.unet
        )

    def init_attention(
        self,
    ) -> None:
        # Overwrite the diffusers attention processors with processors that support FAv3
        attn_dict = self.denoiser.attn_processors
        new_attn_dict = {}
        for key, value in attn_dict.items():
            new_attn_dict[key] = None
            for original_name, replacement in replacement_processors.items():
                if type(value).__name__ == original_name:
                    new_attn_dict[key] = replacement(
                        getattr(self.args, "substitute_sdpa_with_flash_attn", False)
                    )
                    continue
            # If there is not matching attention processor raise error
            if new_attn_dict[key] == None:
                raise NotImplementedError(
                    f"The attention processor{type(value)} has not been implemented with FAv3 compatability."
                )

        self.denoiser.set_attn_processor(new_attn_dict)

    def init_training(
        self,
    ):
        self.init_checkpointing(
            checkpointing_pct=getattr(self.args, "gradient_checkpointing", 0.0),
        )
        self.init_sigmas()

    def init_checkpointing(
        self,
        checkpointing_pct: float = 0.0,
        transformer_block_name: str = "transformer_blocks",
    ) -> None:
        """
        Initialize (partial) gradient checkpointing for the transformer blocks in the model.

        Args:
            checkpointing_pct (float): Percentage of transformer blocks to apply gradient checkpointing to.
                                       Must be between 0.0 and 1.0, where 1.0 applies full checkpointing.
            transformer_block_name (str): Name of the attribute in `self.denoiser`
                                                    that holds the transformer blocks.
        """
        if not (0.0 <= checkpointing_pct <= 1.0):
            raise ValueError(
                f"`checkpointing_pct` must be between [0., 1.], got {checkpointing_pct}"
            )

        elif checkpointing_pct in {0.0, 1.0}:
            # No checkpointing if percentage is 0%
            # Full checkpointing is handled separately in the trainer code
            return

        # Confirm presence of transformer block attribute
        if transformer_block_name is None:
            raise ValueError(
                f"`transformer_block_name = None` is not allowed when `checkpointing_pct` is not equal to 0. or 1."
            )

        if not hasattr(self.denoiser, transformer_block_name):
            raise ValueError(
                f"{self.denoiser.__class__.__name__} doesn't have attribute `{transformer_block_name}`: "
                "review the model source code in Diffusers."
            )

        # Retrieve blocks and determine which ones will use checkpointing
        blocks = getattr(self.denoiser, transformer_block_name)
        cutout_idx = int(len(blocks) * checkpointing_pct)

        # If the percentage calculation results in no blocks, do nothing
        if cutout_idx == 0:
            logger.warning(
                f"Provided non-zero `checkpointing_pct={checkpointing_pct}` didn't yield any blocks to be checkpointed."
                " Try increasing the value."
            )
            return

        # Tag the checkpointed blocks for partial checkpointing
        checkpointed_blocks_names = []
        for idx, (name, block) in enumerate(blocks.named_children()):
            if idx >= cutout_idx:
                break  # No need to continue after reaching the desired percentage

            block.use_in_partial_checkpointing = True
            checkpointed_blocks_names.append(name)

        actual_checkpointing_pct = len(checkpointed_blocks_names) / len(blocks)
        logger.info(
            f"Applying partial gradient checkpointing on transformer blocks "
            f"{checkpointed_blocks_names} (`actualized_pct={actual_checkpointing_pct:.2f}`)"
        )

    def init_sigmas(self) -> None:
        sigmas = (
            self.submodules["scheduler"].timesteps
            / self.submodules["scheduler"].config.num_train_timesteps
        )
        self.sigmas = self.args.shift * sigmas / (1 + (self.args.shift - 1) * sigmas)
        self.submodules["scheduler"].timesteps = (
            self.sigmas * self.submodules["scheduler"].config.num_train_timesteps
        )

    def init_lora(self) -> None:
        """Initializes LoRA (Low-Rank Adaptation) for the denoiser module.

        This method:
        - Freezes parameters of the denoiser
        - Creates a LoRA configuration with the specified rank and alpha
        - Attaches the LoRA adapter to the denoiser module
        - Stores the denoiser's config for reference
        """
        lora_target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]
        self.denoiser.requires_grad_(False)
        model_lora_config = LoraConfig(
            r=self.args.lora_rank,
            lora_alpha=self.args.lora_rank,
            init_lora_weights="gaussian",
            target_modules=lora_target_modules,
        )
        self.denoiser.add_adapter(model_lora_config)
        self.denoiser_config = copy.deepcopy(self.denoiser.config)

    def get_attention_processors(self, *args, **kwargs):
        raise NotImplementedError()

    def set_attention_processors(self, *args, **kwargs):
        raise NotImplementedError()

    def unload_submodules(self) -> None:
        """Offloads all submodules to the CPU.

        Iterates over each submodule in self.submodules and moves it to the CPU,
        freeing up GPU memory.
        """
        for module in self.submodules.values():
            if hasattr(module, "to"):
                module.to("cpu")

    def submodules_to(
        self, device: Union[torch.device, str], dtype: torch.dtype
    ) -> None:
        """Moves all submodules to the specified device and dtype.

        Args:
            device (torch.device or str): The device to which the submodules should be moved
                (e.g., 'cpu' or 'cuda').
            dtype (torch.dtype): The data type for the submodules (e.g., torch.float32).
        """
        for module in self.submodules.values():
            if hasattr(module, "to"):
                module.to(device, dtype)

    def encode_image(self) -> None:
        """Encodes an image into a latent representation.

        This method must be implemented by subclasses that handle image encoding.
        """

    def encode_text(self) -> None:
        """Encodes text into a latent representation.

        This method must be implemented by subclasses that handle text encoding.
        """

    def get_model_inputs(
        self,
        _: Any,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Computes and returns any additional conditionals required for the model.

        Args:
            _ (Any): Placeholder for potential inputs not currently used by this method.

        Returns:
            tuple(dict, torch.Tensor): A dictionary of additional conditionals, and the input timesteps.
        """
        return {}, None

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
        output["latents"] = self.encode_image(batch)
        output["pooled_prompt_embeds"], output["prompt_embeds"] = self.encode_text(
            batch
        )
        if batch.get("input_mask") is not None:
            output["input_mask"] = batch["input_mask"]
        if batch.get("input_mask2") is not None:
            output["input_mask2"] = batch["input_mask2"]

        return output

    def compute_loss(self):
        """Compute the loss for the model."""

    def sample_timesteps_and_noise(self, latents: torch.Tensor, **kwargs) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Sample timesteps and apply noise."""
