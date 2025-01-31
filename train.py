import argparse
import contextlib
import copy
import os
import time
from datetime import datetime
from tqdm import tqdm

import accelerate
import datasets
import diffusers
import numpy as np
import torch
import torch.distributed
import torch.nn.functional as F
import transformers
from accelerate import Accelerator, ProfileKwargs
from accelerate.utils import DummyOptim, DummyScheduler, DynamoBackend, set_seed
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling
from dotenv import load_dotenv
from peft import LoraConfig
from torch.utils.flop_counter import FlopCounterMode
from torchvision import transforms
from datasets import load_dataset
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from transformers import logging as hf_logging
from transformers.integrations.deepspeed import unset_hf_deepspeed_config

from src.attention import FAFluxAttnProcessor2_0
from src.datasets import CacheDataset, cache_collate_fn, collate_fn, preprocess_train
from src.utils import check_gpu_vendor, configure_logging, safely_eval_as_bool

# Load environment variables from a .env file
load_dotenv()

# Configure the logger for outputs to terminal
logger = configure_logging()

# Adjust torch.dynamo.cache_limit if needed
torch._dynamo.config.cache_size_limit = int(
    os.getenv("TORCH_DYNAMO_CACHE_SIZE_LIMIT", torch._dynamo.config.cache_size_limit)
)


class Trainer:
    """Trainer class for model training with multi-GPU support and mixed precision.

    This class encapsulates the training logic for a diffusion model using the Hugging Face
    Accelerate library. It handles model initialization, data loading, preprocessing,
    training loop execution, and validation image generation.

    Attributes:
        args (Namespace): Training configurations and command-line arguments.
        accelerator (Accelerator): Manages multi-GPU training and mixed precision.
        noise_scheduler: Schedules noise levels for the diffusion process.
        vae (AutoencoderKL): Variational Autoencoder for encoding and decoding images.
        transformer (FluxTransformer2DModel): Transformer model for the diffusion process.
        text_encoder (CLIPTextModel): First text encoder for processing captions.
        text_encoder_2 (T5EncoderModel): Second text encoder for processing captions.
        tokenizer (CLIPTokenizer): Tokenizer corresponding to the first text encoder.
        tokenizer_2 (T5TokenizerFast): Tokenizer corresponding to the second text encoder.
        vae_scale_factor (int): Scale factor derived from the VAE configuration.
        height (int): Height of the latent images after scaling.
        width (int): Width of the latent images after scaling.
        num_channels_latents (int): Number of channels in the latent representations.
        weight_dtype (torch.dtype): Data type used for mixed precision training.

    """

    def __init__(self, args) -> None:
        """Initialize the Trainer with configurations and models.

        Args:
            args (Namespace): Training configurations and arguments.
        """
        # Arguments
        self.args = args
        self.is_cached = False

        # Multi-gpu with accelerate
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_dir=args.logging_dir,
        )
        self.accelerator.init_trackers(
            os.path.join(args.report_to, datetime.now().strftime("%Y%m%d_%H%M%S"))
        )
        # Keep warning messages to the main process only. No progress bar spam.
        if self.accelerator.is_local_main_process:
            hf_logging.set_verbosity_warning()
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
        else:
            diffusers.utils.logging.disable_progress_bar()
            hf_logging.set_verbosity_error()
            datasets.disable_progress_bars()
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            transformers.utils.logging.disable_progress_bar()

        if args.seed is not None:
            set_seed(args.seed, device_specific=True)

        # Load all the parts of the model seperately
        self.transformer = FluxTransformer2DModel.from_pretrained(
            args.model_path,
            subfolder="transformer",
        )
        self.transformer_config = copy.deepcopy(self.transformer.config)

        # Overwrite the FluxAttnProcessor with an implementation that supports flash-attn
        self.transformer.set_attn_processor(
            FAFluxAttnProcessor2_0(
                substitute_sdpa_with_flash_attn=self.args.substitute_sdpa_with_flash_attn
            )
        )

        # Hacky way of getting deepspeed zero stage 3 working
        # (init on multiple models seems to not be supported in accelerate)
        unset_hf_deepspeed_config()

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.model_path, subfolder="scheduler"
        )

        # From https://github.com/huggingface/diffusers/blob/edb8c1bce67e81f0de90a7e4c16b2f6537d39f2d/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L114
        self.sigmas = (
            self.noise_scheduler.timesteps
            / self.noise_scheduler.config.num_train_timesteps
        ).to(self.accelerator.device)
        self.sigmas = (
            self.args.shift * self.sigmas / (1 + (self.args.shift - 1) * self.sigmas)
        )
        self.noise_scheduler.timesteps = (
            self.sigmas * self.noise_scheduler.config.num_train_timesteps
        )

        self.vae = AutoencoderKL.from_pretrained(
            args.model_path,
            subfolder="vae",
        )

        self.text_encoder = CLIPTextModel.from_pretrained(
            args.model_path,
            subfolder="text_encoder",
        )

        self.text_encoder_2 = T5EncoderModel.from_pretrained(
            args.model_path,
            subfolder="text_encoder_2",
        )

        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.args.model_path,
            subfolder="tokenizer",
        )
        self.tokenizer_2 = T5TokenizerFast.from_pretrained(
            self.args.model_path,
            subfolder="tokenizer_2",
        )

        self.pipe = FluxPipeline.from_pretrained(
            self.args.model_path,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            text_encoder_2=self.text_encoder_2,
            tokenizer_2=self.tokenizer_2,
            transformer=self.transformer,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.height = 2 * (int(args.resolution) // (self.vae_scale_factor * 2))
        self.width = 2 * (int(args.resolution) // (self.vae_scale_factor * 2))
        self.num_channels_latents = self.transformer.config.in_channels // 4

        # For mixed precision training
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
            args.mixed_precision = self.accelerator.mixed_precision
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16
            args.mixed_precision = self.accelerator.mixed_precision

        if self.args.use_gradient_checkpointing:
            # Gradient checkpointing to save memory at the cost of computatational complexity
            self.transformer.enable_gradient_checkpointing()
        if self.args.use_lora:
            # Modules to apply the lora matrix to
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
            self.init_lora(lora_target_modules)

    def init_lora(self, lora_target_modules):
        self.transformer.requires_grad_(False)
        self.accelerator.wait_for_everyone()
        transformer_lora_config = LoraConfig(
            r=self.args.lora_rank,
            lora_alpha=self.args.lora_rank,
            init_lora_weights="gaussian",
            target_modules=lora_target_modules,
        )
        self.transformer.add_adapter(transformer_lora_config)
        self.transformer.to(dtype=torch.float32)

    def generate_val_image(self, iter=0):
        """Generate and save a validation image using the FLUX.1 Hugging Face pipeline.

        Args:
            iter (int, optional): The current iteration number, used in the output filename.
                Defaults to 0.
        """
        # Load all necessary components into VRAM
        self.pipe.vae.to(self.accelerator.device, self.weight_dtype)
        self.pipe.text_encoder.to(self.accelerator.device, self.weight_dtype)
        self.pipe.text_encoder_2.to(self.accelerator.device, self.weight_dtype)

        # Insert the initial config (required for deepspeed support)
        temp_copy = self.transformer.config
        self.transformer.config = self.transformer_config
        self.pipe.transformer = self.transformer

        self.pipe.set_progress_bar_config(
            disable=not self.accelerator.is_local_main_process
        )
        prompt = self.args.validation_prompts
        generator = None
        if self.args.seed is not None:
            generator = torch.Generator(device=self.accelerator.device).manual_seed(
                self.args.seed
            )

        # Known issue where latents are cast to torch.float32 before being fed into the VAE.
        # This forces the latents to be cast to float16/bfloat16 to avoid precision issues.
        def convertfp16(flux_pipeline, i, t, callback_kwargs):
            latents = callback_kwargs["latents"].to(self.weight_dtype)
            return {"latents": latents}

        image = self.pipe(
            prompt,
            guidance_scale=self.args.validation_guidance_scale,
            num_inference_steps=self.args.validation_inference_steps,
            max_sequence_length=256,
            height=self.args.resolution,
            width=self.args.resolution,
            callback_on_step_end=(
                convertfp16 if self.weight_dtype != torch.float32 else None
            ),
            callback_on_step_end_tensor_inputs=(
                ["latents"] if self.weight_dtype != torch.float32 else None
            ),
            generator=generator,
        ).images[0]

        # Return the deepspeed configuration
        self.transformer.config = temp_copy

        # Unload to CPU
        if self.args.use_cache:
            self.pipe.vae.to("cpu")
            self.pipe.text_encoder.to("cpu")
            self.pipe.text_encoder_2.to("cpu")
        if self.accelerator.is_local_main_process:
            if not os.path.exists("./outputs"):
                os.makedirs("./outputs")

            image.save(
                os.path.join(
                    self.args.output_dir, f"validation_images/val_image_iter{iter}.png"
                )
            )

    def get_dataloader(self):
        """Create and return the training DataLoader.

        Returns:
            DataLoader: A PyTorch DataLoader instance for the training dataset.
        """
        # The tokenizers need to be loaded separately to avoid issues
        # with multiprocessing and pickling in PyTorch DataLoader.
        tokenizer = CLIPTokenizer.from_pretrained(
            self.args.model_path,
            subfolder="tokenizer",
        )
        tokenizer_2 = T5TokenizerFast.from_pretrained(
            self.args.model_path,
            subfolder="tokenizer_2",
        )

        if not os.path.exists(self.args.cache_dir):
            os.makedirs(self.args.cache_dir)

        # Check whether all three files for the latents and text encodings have been cached.
        # Otherwise they need to be computed.
        if (
            os.path.exists(os.path.join(self.args.cache_dir, "latents.npy"))
            and os.path.exists(os.path.join(self.args.cache_dir, "prompt_embeds.npy"))
            and os.path.exists(
                os.path.join(self.args.cache_dir, "pooled_prompt_embeds.npy")
            )
            and self.args.use_cache
        ) or self.is_cached:
            self.is_cached = True
            # Load cached dataset
            train_dataset = CacheDataset(self.args.cache_dir)

            # Create the dataloader
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                shuffle=True,
                collate_fn=cache_collate_fn,
                batch_size=self.args.train_batch_size,
                num_workers=self.args.dataloader_num_workers,
            )
        else:
            # Load the dataset
            dataset = load_dataset(
                self.args.train_data_path,
                split="train",
            )

            # Create the image transform pipeline
            train_transforms = transforms.Compose(
                [
                    transforms.Resize(
                        self.args.resolution,
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    ),
                    (
                        transforms.CenterCrop(self.args.resolution)
                        if self.args.center_crop
                        else transforms.RandomCrop(self.args.resolution)
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )

            # Process the data with the image transform
            with self.accelerator.main_process_first():
                train_dataset = dataset.with_transform(
                    lambda examples: preprocess_train(
                        examples, tokenizer, tokenizer_2, train_transforms
                    ),
                )

            # Create the dataloader
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                shuffle=True,
                collate_fn=collate_fn,
                batch_size=self.args.train_batch_size,
                num_workers=self.args.dataloader_num_workers,
            )

        return train_dataloader

    def cache_latents_and_text(self, dataloader):
        """Calculate and cache the latents

        Args:
            DataLoader: A PyTorch DataLoader instance for the training dataset to be cached.
        """
        progress_bar = tqdm(
            range(
                0,
                len(dataloader),
            ),
            desc="Caching latents",
            # Only show the progress bar once on each machine.
            disable=not self.accelerator.is_local_main_process,
            leave=False,
        )
        latents_list = []
        pooled_prompt_list = []
        prompt_list = []
        # Calculate the latents for all images
        for step, batch in enumerate(dataloader):
            with torch.inference_mode():
                # Generate latents
                latents, img_ids = self.encode_image(batch)

                # Generate text encodings
                pooled_prompt_embeds, prompt_embeds, text_ids = self.encode_text(batch)

                gathered_latents = self.accelerator.gather(latents)
                gathered_pooled_embeds = self.accelerator.gather(pooled_prompt_embeds)
                gathered_embeds = self.accelerator.gather(prompt_embeds)

                if self.accelerator.is_main_process:
                    latents_list.append(gathered_latents.to("cpu"))
                    pooled_prompt_list.append(gathered_pooled_embeds.to("cpu"))
                    prompt_list.append(gathered_embeds.to("cpu"))

                if self.accelerator.is_local_main_process:
                    progress_bar.update(1)

        if self.accelerator.is_main_process:
            latents_list = torch.cat(latents_list)
            pooled_prompt_list = torch.cat(pooled_prompt_list)
            prompt_list = torch.cat(prompt_list)

        if self.accelerator.is_main_process:
            np.save(
                os.path.join(self.args.cache_dir, "latents.npy"),
                latents_list.numpy(),
            )
            np.save(
                os.path.join(self.args.cache_dir, "pooled_prompt_embeds.npy"),
                pooled_prompt_list.numpy(),
            )
            np.save(
                os.path.join(self.args.cache_dir, "prompt_embeds.npy"),
                prompt_list.numpy(),
            )

        self.accelerator.wait_for_everyone()
        self.is_cached = True

    def prepare_latent_image_ids(self, batch_size, height, width, device, dtype):
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
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width)[None, :]
        )

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = (
            latent_image_ids.shape
        )

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    def prepare_training(self, optimizer, train_dataloader, lr_scheduler):
        """Prepare the model and optimizer for training, especially for multi-GPU setups.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer initialized with model parameters.
            train_dataloader (torch.utils.data.DataLoader): The DataLoader for training data.
            lr_scheduler (_LRScheduler): The learning rate scheduler.

        Returns:
            Tuple[torch.optim.Optimizer, DataLoader, _LRScheduler]:
                - optimizer: The prepared optimizer.
                - train_dataloader: The prepared DataLoader.
                - lr_scheduler: The prepared learning rate scheduler.
        """
        # Only train the diffusion transformer
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.transformer.train()

        # Prepare multi-gpu training
        self.transformer, optimizer, train_dataloader, lr_scheduler = (
            self.accelerator.prepare(
                self.transformer, optimizer, train_dataloader, lr_scheduler
            )
        )

        # Move to correct gpu/dtype, offload vae to cpu if latents are cached
        if not self.args.use_cache:
            self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
            self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
            self.text_encoder_2.to(self.accelerator.device, dtype=self.weight_dtype)
        elif not self.is_cached:
            self.vae.to(self.accelerator.device, dtype=torch.float32)
            self.text_encoder.to(self.accelerator.device, dtype=torch.float32)
            self.text_encoder_2.to(self.accelerator.device, dtype=torch.float32)
        else:
            self.vae.to("cpu", dtype=self.weight_dtype)
            self.text_encoder.to("cpu", dtype=self.weight_dtype)
            self.text_encoder_2.to("cpu", dtype=self.weight_dtype)
        self.transformer.to(self.accelerator.device, dtype=self.weight_dtype)

        return optimizer, train_dataloader, lr_scheduler

    def encode_image(self, batch):
        """Encode images into latents and generate corresponding image IDs.

        Args:
            batch (dict): Batch containing 'pixel_values' of images.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - latents: Encoded latent representations of the images.
                - image_ids: Image IDs associated with the latents.
        """
        # If not cached
        if not self.is_cached:
            # Encode image
            latents = self.vae.encode(
                batch["pixel_values"].to(self.vae.dtype)
            ).latent_dist.sample()
            latents = (
                latents - self.vae.config.shift_factor
            ) * self.vae.config.scaling_factor
            bsz = latents.shape[0]

            # Pack the latents ...?
            latents = latents.view(
                bsz,
                self.num_channels_latents,
                self.height // 2,
                2,
                self.width // 2,
                2,
            )
            latents = latents.permute(0, 2, 4, 1, 3, 5)
            latents = latents.reshape(
                bsz,
                (self.height // 2) * (self.width // 2),
                self.num_channels_latents * 4,
            )
        else:
            # Else use cached latents
            latents = batch["latents"].to(self.weight_dtype)
            bsz = latents.shape[0]

        # Create the image ids
        image_ids = self.prepare_latent_image_ids(
            bsz,
            self.height // 2,
            self.width // 2,
            self.accelerator.device,
            self.weight_dtype,
        )
        return latents, image_ids

    def encode_text(self, batch):
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
        if not self.is_cached:
            pooled_prompt_embeds = self.text_encoder(
                batch["input_ids"], output_hidden_states=False
            )
            pooled_prompt_embeds = pooled_prompt_embeds.pooler_output
            pooled_prompt_embeds = pooled_prompt_embeds.to(
                dtype=self.text_encoder.dtype
            )

            prompt_embeds = self.text_encoder_2(
                batch["input_ids_2"], output_hidden_states=False
            )[0]
            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype)
        else:
            pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(self.vae.dtype)
            prompt_embeds = batch["prompt_embeds"].to(self.vae.dtype)

        # Additional embeddings?
        text_ids = torch.zeros(
            prompt_embeds.shape[1],
            3,
            dtype=self.weight_dtype,
            device=self.accelerator.device,
        )
        return pooled_prompt_embeds, prompt_embeds, text_ids

    def noise_latent(self, latents, logit_mean=0, logit_std=1):
        """Apply noise to latents and sample timesteps.

        Args:
            latents (torch.Tensor): The latent representations to be noised.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - noised_latents: Latents with noise applied.
                - noise: The noise added to the latents.
                - timesteps: Normalized timesteps used for noise scaling.
        """
        # Sample timesteps and create noise - density timestep sampling as used in sd3
        noise = torch.randn_like(latents)
        u = compute_density_for_timestep_sampling(
            weighting_scheme="logit_normal",
            batch_size=latents.shape[0],
            logit_mean=logit_mean,
            logit_std=logit_std,
        )
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices].to(self.accelerator.device)

        noised_latents = (
            self.sigmas[indices, None, None] * noise
            + (1.0 - self.sigmas[indices, None, None]) * latents
        )
        return (
            noised_latents,
            noise,
            timesteps / self.noise_scheduler.config.num_train_timesteps,
        )

    def profiling_ctx(self, step):

        if self.args.profiling_step == step:
            profile_kwargs = ProfileKwargs(
                activities=["cpu", "cuda"],
                record_shapes=True,
                with_stack=False,
                profile_memory=True,
                on_trace_ready=lambda prof: prof.export_chrome_trace(
                    os.path.join(
                        self.args.profiling_logging_dir,
                        f"pytorch_profile_gpu{self.accelerator.process_index}.json",
                    )
                ),
            )
            return self.accelerator.profile(profile_kwargs)

        return contextlib.nullcontext()

    def is_deepspeed_optimizer_configured(self) -> bool:
        """
        Checks if the DeepSpeed plugin is configured with an optimizer.
        """
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        return (
            deepspeed_plugin is not None
            and "optimizer" in deepspeed_plugin.deepspeed_config
        )

    def run_training(self):
        epsilon = self.args.adam_epsilon
        if self.args.adam_epsilon <= 1e-08 and self.weight_dtype == torch.float16:
            epsilon = 1e-03
            logger.warning(
                f"WARNING: Epsilon value too low for fp16. Increasing it to {epsilon}"
            )

        # Prepare model before optimizer is supposed to be more efficient for fsdp
        if self.accelerator.distributed_type == accelerate.DistributedType.FSDP:
            self.transformer = self.accelerator.prepare(self.transformer)

        trainable_params = list(
            filter(lambda p: p.requires_grad, self.transformer.parameters())
        )

        # Load the dataset
        train_dataloader = self.get_dataloader()

        # Optimizer settings
        # Only optimize parameters that require grad, e.g. lora layers
        optimizer_params = dict(
            params=trainable_params,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=epsilon,
            fused=True,
        )

        if self.is_deepspeed_optimizer_configured():
            optimizer = DummyOptim(**optimizer_params)
            lr_scheduler = DummyScheduler(
                optimizer,
                total_num_steps=self.args.num_iterations
                * self.accelerator.num_processes,
                num_warmup_steps=self.args.lr_warmup_steps
                * self.accelerator.num_processes,
            )
        else:
            optimizer = torch.optim.AdamW(**optimizer_params)
            # Calculate the amount of steps for the scheduling
            lr_scheduler = get_scheduler(
                self.args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=self.args.lr_warmup_steps
                * self.accelerator.num_processes,
                num_training_steps=self.args.num_iterations
                * self.accelerator.num_processes,
            )

        # Prepare model for training
        optimizer, train_dataloader, lr_scheduler = self.prepare_training(
            optimizer, train_dataloader, lr_scheduler
        )

        # Cache the latents and reload the dataloader
        if self.args.use_cache and not self.is_cached:
            self.cache_latents_and_text(train_dataloader)
            train_dataloader = self.get_dataloader()
            optimizer, train_dataloader, lr_scheduler = self.prepare_training(
                optimizer, train_dataloader, lr_scheduler
            )

        progress_bar = tqdm(
            range(
                0,
                int(self.args.num_iterations),
            ),
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not self.args.show_progress_bar
            or not self.accelerator.is_local_main_process,
        )

        # Calculate the flops for the training loop
        flop_counter = FlopCounterMode(display=False, depth=None)
        total_flops = 0.0

        # Disable flop counting when using torch.compile
        if self.accelerator.state.dynamo_plugin.backend != DynamoBackend.NO:
            flop_counter = contextlib.nullcontext()
            logger.warning(
                "WARNING: Disabling TFlops/s calculations due to incompatability with torch.compile"
            )

        # Training loop
        global_step = 0
        train_loss = 0.0

        while global_step < self.args.num_iterations:
            for step, batch in enumerate(train_dataloader):
                # Generate validation image
                if (
                    global_step % self.args.validation_iteration == 0
                    and self.args.validation_iteration != -1
                ):
                    with torch.no_grad():
                        self.generate_val_image(iter=global_step)

                # Train step
                if self.accelerator.use_distributed:
                    # We wait for every process to reach this,
                    # so that below timing would only contain current training step,
                    # not wait times of previous steps
                    # NB! This is strictly for benchmarking purposes
                    # and should be removed in real use
                    self.accelerator.wait_for_everyone()

                with self.accelerator.accumulate(self.transformer), self.profiling_ctx(
                    step
                ), flop_counter:
                    start_time = time.time()
                    # Encode the images
                    latents, image_ids = self.encode_image(batch)

                    # Sample timesteps and create noise - density timestep sampling as used in sd3
                    noised_latents, noise, timesteps = self.noise_latent(latents)

                    # Encode the prompts
                    pooled_prompt_embeds, prompt_embeds, text_ids = self.encode_text(
                        batch
                    )

                    if self.transformer_config.guidance_embeds:
                        # No CFG during training i.e. guidance of 1
                        guidance = torch.full(
                            [1],
                            1.0,
                            dtype=self.weight_dtype,
                            device=self.accelerator.device,
                        ).expand(latents.shape[0])
                    else:
                        guidance = None

                    # Inference
                    pred = self.transformer(
                        hidden_states=noised_latents,
                        timestep=timesteps,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=image_ids,
                        return_dict=False,
                    )[0]

                    # Flow matching loss and backward pass
                    target = noise - latents
                    loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
                    self.accelerator.backward(loss)

                    # Clip gradients
                    if (
                        self.accelerator.sync_gradients
                        and self.args.max_grad_norm > 0.0
                    ):
                        self.accelerator.clip_grad_norm_(
                            trainable_params,
                            self.args.max_grad_norm,
                        )

                    # Optimizer step
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                # Wait for all processes to finish before recording the time
                if self.accelerator.use_distributed:
                    self.accelerator.wait_for_everyone()
                step_time = time.time() - start_time
                fps_gpu = self.args.train_batch_size / step_time

                # Calculate loss across all processes
                # This is only for logging purposes; loss.backward() has already been done above
                avg_loss = self.accelerator.gather(
                    loss.repeat(self.args.train_batch_size)
                ).mean()
                train_loss += avg_loss.item() / self.args.gradient_accumulation_steps

                # Remove flop counter since we only need to compute the flops once.
                if isinstance(flop_counter, FlopCounterMode):
                    total_flops = flop_counter.get_total_flops()
                    flop_counter = contextlib.nullcontext()
                logs = {
                    "step_loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step_time": step_time,
                    "fps_gpu": fps_gpu,
                    "tflops/s": total_flops * 1e-12 / step_time,
                }

                # Logging
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    logs["train_loss"] = train_loss
                    train_loss = 0.0

                progress_bar.set_postfix(**logs)
                if self.accelerator.is_local_main_process:
                    logger.info(f"Step {global_step}: {logs}")
                    self.accelerator.log(logs, step=global_step)

                # Break if the max number of iterations are exceeded
                if global_step >= self.args.num_iterations:
                    break

            # Generate last validation image
            if global_step >= self.args.num_iterations:
                if (
                    global_step % self.args.validation_iteration == 0
                    and self.args.validation_iteration != -1
                ):
                    with torch.no_grad():
                        self.generate_val_image(iter=global_step)
        with (
            self.transformer.summon_full_params(self.transformer, writeback=False)
            if self.accelerator.distributed_type == accelerate.DistributedType.FSDP
            else contextlib.nullcontext()
        ):
            if self.accelerator.is_local_main_process:
                self.accelerator.unwrap_model(self.transformer).save_pretrained(
                    os.path.join(self.args.output_dir, "checkpoint")
                )

        # Empty the cache
        self.accelerator.end_training()
        self.accelerator.clear()


def parse_args():
    """Parse the arguments necessary for training:

    Returns:
        args: The parsed arguments containing the parameters necessary for training.
    """
    parser = argparse.ArgumentParser(
        description="Simple training script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.getenv("MODEL_PATH", "black-forest-labs/FLUX.1-dev"),
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.getenv("OUTPUT_DIR", "./outputs"),
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        default=os.getenv("TRAIN_DATA_PATH", "bghira/pseudo-camera-10k"),
        help="The directory where the training data is stored or dataset identifier from huggingface.co/datasets.",
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=os.getenv("VALIDATION_PROMPTS", "high-res picture of a cat"),
        help="A set of prompts evaluated every `--validation_iteration` and logged to `--report_to`.",
    )
    parser.add_argument(
        "--validation_iteration",
        type=int,
        default=int(os.getenv("VALIDATION_ITERATION", -1)),
        help="Number of iterations between validations.",
    )
    parser.add_argument(
        "--validation_guidance_scale",
        type=float,
        default=float(os.getenv("VALIDATION_GUIDANCE_SCALE", 0.0)),
        help="Number of iterations between validations.",
    )
    parser.add_argument(
        "--validation_inference_steps",
        type=int,
        default=int(os.getenv("VALIDATION_INFERENCE_STEPS", 20)),
        help="Number of iterations between validations.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.getenv("CACHE_DIR", "./cache"),
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--use_cache",
        default=os.getenv("USE_CACHE", "1"),
        type=lambda x: safely_eval_as_bool(x),
        help="Whether to use the cached latents or generate new ones. Accepts 'true', 'false', '1', or '0'.",
    )
    parser.add_argument(
        "--substitute_sdpa_with_flash_attn",
        default=os.getenv(
            "SUBSTITUTE_SDPA_WITH_FLASH_ATTN", check_gpu_vendor() == "rocm"
        ),
        type=lambda x: safely_eval_as_bool(x),
        help=(
            "Whether to use Flash-Attention as opposed to PyTorch's native SDPA as the attention backend. "
            "Accepts 'true', 'false', '1', or '0'. Note that Flash-Attention is not available for fp32; "
            "a warning will be issued if a datatype other than bf16 or fp16 is used, but execution will continue using SDPA. "
            "Consider using bf16 or fp16 for compatibility and potential speed-ups."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.getenv("SEED", 42)),
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=int(os.getenv("RESOLUTION", 512)),
        help="The resolution for input images, all the images in the train/validation dataset will be resized to this resolution.",
    )
    parser.add_argument(
        "--center_crop",
        default=safely_eval_as_bool(os.getenv("CENTER_CROP", "false")),
        action="store_true",
        help="Whether to center crop the input images to the resolution. If not set, the images will be randomly cropped.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=int(os.getenv("TRAIN_BATCH_SIZE", 1)),
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=int(os.getenv("GRADIENT_ACCUMULATION_STEPS", 1)),
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=int(os.getenv("NUM_ITERATIONS", 100)),
        help="Total number of training iterations.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=float(os.getenv("LEARNING_RATE", 1e-5)),
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=float(os.getenv("SHIFT", 3.0)),
        help="Flow matching shift parameter. Common values lie between 0.0-4.0.",
    )
    parser.add_argument(
        "--scale_lr",
        default=safely_eval_as_bool(os.getenv("SCALE_LR", "false")),
        action="store_true",
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default=os.getenv("LR_SCHEDULER", "constant"),
        help="The scheduler type to use. Choose between"
        '["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"].',
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=int(os.getenv("LR_WARMUP_STEPS", 500)),
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=int(os.getenv("DATALOADER_NUM_WORKERS", 0)),
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=float(os.getenv("ADAM_BETA1", 0.9)),
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=float(os.getenv("ADAM_BETA2", 0.999)),
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=float(os.getenv("ADAM_WEIGHT_DECAY", 0.0)),
        help="Weight decay to use.",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=float(os.getenv("ADAM_EPSILON", 1e-08)),
        help="Epsilon value for the Adam optimizer.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=float(os.getenv("MAX_GRAD_NORM", 0.0)),
        help="Max gradient norm.",
    )
    parser.add_argument(
        "--show_progress_bar",
        default=os.getenv("SHOW_PROGRESS_BAR", "false"),
        type=lambda x: safely_eval_as_bool(x),
        help="Whether to show TQDM progress bar or not for training (only in main process). Accepts 'true', 'false', '1', or '0'.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=os.getenv("MIXED_PRECISION", None),
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--use_lora",
        default=os.getenv("USE_LORA", "false"),
        type=lambda x: safely_eval_as_bool(x),
        help="Whether to use lora for fine-tuning purposes. Accepts 'true', 'false', '1', or '0'.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=int(os.getenv("LORA_RANK", 8)),
        help="The LoRA rank to use while training.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=os.getenv("USE_GRADIENT_CHECKPOINTING", "false"),
        type=lambda x: safely_eval_as_bool(x),
        help="Whether to use gradient checkpointing to save vram during training. Accepts 'true', 'false', '1', or '0'.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=int(os.getenv("LOCAL_RANK", -1)),
        help="For distributed training: local_rank.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=os.getenv("REPORT_TO", "tensorboard"),
        help="The integration to report the results and logs to."
        'Supported platforms are "tensorboard" (default), "wandb" and "comet_ml". Use "all" to report to all integrations.',
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default=os.getenv("LOGGING_DIR", "logs"),
        help="Experiment tracker (see args.report_to) log directory. Defaults to `logs`.",
    )
    parser.add_argument(
        "--profiling_step",
        type=int,
        default=int(os.getenv("PROFILING_STEP", -1)),
        help="Determines on which training iter to perform the profiling. Negative numbers disable the profiling (default).",
    )
    parser.add_argument(
        "--profiling_logging_dir",
        type=str,
        default=os.getenv("PROFILING_LOGGING_DIR", "logs/traces"),
        help="The logging directory where the profiling trace-files (JSON) will be written.",
    )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.train_data_path is None:
        raise ValueError("Need to specify a training folder.")

    return args


def main():
    """
    Main function to execute the training process.

    This function parses the command-line arguments, initializes the Trainer
    with the provided arguments, and starts the training loop.

    """
    # Get arguments
    args = parse_args()

    # Run training loop
    trainer = Trainer(args)
    trainer.run_training()


if __name__ == "__main__":
    main()
