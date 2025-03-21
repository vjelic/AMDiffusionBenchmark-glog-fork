import argparse
import copy
from typing import Union

import diffusers
import torch
from peft import LoraConfig

from src.models.attention import replacement_processors

# A map from hf_id to appropriate huggingface pipeline
pipeline_map = {
    "black-forest-labs/FLUX.1-dev": diffusers.FluxPipeline,
    "hunyuanvideo-community/HunyuanVideo": diffusers.HunyuanVideoPipeline,
    "stabilityai/stable-diffusion-xl-base-1.0": diffusers.StableDiffusionXLPipeline,
}


class BaseModel:
    """A base model class for image and video diffusion using the Diffusers library.

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

    def __init__(self, args: argparse.Namespace) -> None:
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
                "scheduler": None,
            }
        self.init_modules_from_pipeline()

        # Overwrite the diffusers attention processors with processors that support FAv3
        attn_dict = self.denoiser.attn_processors
        new_attn_dict = {}
        for key, value in attn_dict.items():
            new_attn_dict[key] = None
            for original_name, replacement in replacement_processors.items():
                if type(value).__name__ == original_name:
                    new_attn_dict[key] = replacement(
                        self.args.substitute_sdpa_with_flash_attn
                    )
                    continue
            # If there is not matching attention processor raise error
            if new_attn_dict[key] == None:
                raise NotImplementedError(
                    f"The attention processor{type(value)} has not been implemented with FAv3 compatability."
                )

        self.denoiser.set_attn_processor(new_attn_dict)

    def init_modules_from_pipeline(self) -> None:
        """Initializes the modules from the Diffusers pipeline.

        The pipeline is loaded from the Hugging Face repository or from
        a local path specified in self.args.model_path. The corresponding
        submodules are then extracted and stored in self.submodules.
        """
        # Download all parts of the model
        self.pipe = pipeline_map[self.hf_id].from_pretrained(
            self.args.model_path if self.args.model_path else self.hf_id
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

    def additional_conditionals(
        self, _: dict, timesteps: torch.Tensor
    ) -> tuple[dict, torch.Tensor]:
        """Computes and returns any additional conditionals required for the model.

        Args:
            _ (Any): Placeholder for potential inputs not currently used by this method.
            timesteps (torch.Tensor): Denoising timesteps for which conditionals are calculated.

        Returns:
            tuple(dict, torch.Tensor): A dictionary of additional conditionals, and the input timesteps.
        """
        return {}, timesteps

    def encode_image(self) -> None:
        """Encodes an image into a latent representation.

        This method must be implemented by subclasses that handle image encoding.
        """
        pass

    def encode_text(self) -> None:
        """Encodes text into a latent representation.

        This method must be implemented by subclasses that handle text encoding.
        """
        pass

    def compute_loss(self):
        """Compute the loss for the model."""
        pass

    def sample_timesteps_and_noise(self):
        """Sample timesteps and apply noise."""
        pass
