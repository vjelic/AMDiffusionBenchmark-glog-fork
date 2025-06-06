import torch
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.training_utils import compute_density_for_timestep_sampling

from src.utils import configure_logging

# Configure the logger for outputs to terminal
logger = configure_logging()


def compute_density_based_timestep_sampling(
    scheduler: SchedulerMixin,
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample timesteps and apply noise using density-based timestep sampling.

    Args:
        scheduler (SchedulerMixin): Scheduler to be used for the sampling.
        latents (torch.Tensor): The latent representations.
        sigmas (torch.Tensor): Precomputed sigma values passed from the Trainer.
        logit_mean (float): Mean for density sampling.
        logit_std (float): Std for density sampling.

    Returns:
        tuple:
            - noised_latents (torch.Tensor): Latents after applying noise.
            - noise (torch.Tensor): The sampled noise.
            - normalized_timesteps (torch.Tensor): Timesteps normalized to [0, 1].
    """

    # Compute the density for each latent representation
    u = compute_density_for_timestep_sampling(
        weighting_scheme="logit_normal",
        batch_size=latents.shape[0],
        logit_mean=logit_mean,
        logit_std=logit_std,
    )
    num_steps = scheduler.config.num_train_timesteps
    indices = (u * num_steps).long()
    timesteps = scheduler.timesteps[indices].to(latents.device)
    normalized_timesteps = timesteps / num_steps

    # Sample noise and apply it to the latents
    noise = torch.randn_like(latents)
    sigmas_expanded = sigmas.to(latents.device)[indices].view(
        -1, *([1] * (noise.ndim - 1))
    )
    noised_latents = (sigmas_expanded * noise + (1.0 - sigmas_expanded) * latents).to(
        latents.dtype
    )

    return noised_latents, noise, normalized_timesteps


def compute_uniform_timestep_sampling(
    scheduler: SchedulerMixin,
    latents: torch.Tensor,
    sigmas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample timesteps and apply noise to latents using the scheduler.

    Timesteps are sampled uniformly as integers in [0, num_steps).

    Args:
        scheduler (SchedulerMixin): Scheduler to be used for the sampling.
        latents (torch.Tensor): The latent representations to be noised.
        sigmas (torch.Tensor): Precomputed sigma values passed from the Trainer.
        logit_mean (float): (Unused) Mean for density sampling; kept for interface consistency.
        logit_std (float): (Unused) Std for density sampling; kept for interface consistency.

    Returns:
        tuple:
            - noised_latents (torch.Tensor): Latents after noise has been applied.
            - noise (torch.Tensor): The sampled noise tensor.
            - timesteps (torch.Tensor): Timestep integers from [0, num_steps).
    """
    # Sample integer timesteps uniformly
    num_steps = scheduler.config.num_train_timesteps
    timesteps = torch.randint(0, num_steps, (latents.shape[0],), device=latents.device)
    # Use the passed sigmas: select sigma values based on sampled timesteps
    sigmas_expanded = sigmas.to(latents.device)[timesteps].view(
        -1, *([1] * (latents.ndim - 1))
    )

    # Sample noise and apply it to the latents
    noise = torch.randn_like(latents)
    noised_latents = latents + noise * sigmas_expanded
    return noised_latents, noise, timesteps


def get_attention_processors(module, processors=None):
    """
    Recursively get the attention processors.

    Args:
        module (Any): The module from which to start retrieval.
        processors (dict): Dictionary to store processors.

    Returns:
        dict: All attention processors throughout the model.
    """

    if processors is None:
        processors = {}

    for name, sub_module in module.named_children():
        if hasattr(sub_module, "processor"):
            processors[name] = sub_module.processor
        else:
            # Recurse into child modules
            get_attention_processors(sub_module, processors)

    return processors


def set_attention_processors(processor_dict, module):
    """
    Recursively set the attention processors.

    Args:
        processor_dict (dict): A dictionary of processors to set.
        module (Any): The module in which to start setting.
    """

    for name, sub_module in module.named_children():
        if name in processor_dict:
            processor = processor_dict[name]
            if (
                hasattr(sub_module, "processor")
                and isinstance(sub_module.processor, torch.nn.Module)
                and not isinstance(processor, torch.nn.Module)
            ):
                logger.info(
                    f"You are removing possibly trained weights of {sub_module.processor} with {processor}"
                )
                sub_module._modules.pop("processor")

            # Set the new processor
            sub_module.processor = processor
        else:
            # Recurse into child modules
            set_attention_processors(processor_dict, sub_module)
