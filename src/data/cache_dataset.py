import os

import numpy as np
import torch
from torch.utils.data import Dataset


class CacheDataset(Dataset):
    """Dataset for loading the cached latents and encodings."""

    def __init__(self, cache_dir, dtype):
        """Loads the encodings in read mode for faster initialization.

        Args:
            cache_dir (str): The path to the cached latents and text encodings
        """
        self.latents = np.load(os.path.join(cache_dir, "latents.npy"), mmap_mode="r")
        self.pooled_prompt_embeds = np.load(
            os.path.join(cache_dir, "pooled_prompt_embeds.npy"), mmap_mode="r"
        )
        self.prompt_embeds = np.load(
            os.path.join(cache_dir, "prompt_embeds.npy"), mmap_mode="r"
        )
        self.input_mask = np.load(
            os.path.join(cache_dir, "input_mask.npy"), mmap_mode="r"
        )
        self.input_mask_2 = np.load(
            os.path.join(cache_dir, "input_mask_2.npy"), mmap_mode="r"
        )
        self.dtype = dtype
        # Check that all the embeds and latents have the same length
        assert len(self.latents) == len(self.pooled_prompt_embeds) and len(
            self.pooled_prompt_embeds
        ) == len(
            self.prompt_embeds
        ), f"{self.latents.shape=}, {self.pooled_prompt_embeds.shape=}, {self.prompt_embeds.shape=}"

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        # Access individual samples directly by index
        latent = torch.from_numpy(np.copy(self.latents[idx])).to(self.dtype)
        pooled_prompt_embeds = torch.from_numpy(
            np.copy(self.pooled_prompt_embeds[idx])
        ).to(self.dtype)
        prompt_embeds = torch.from_numpy(np.copy(self.prompt_embeds[idx])).to(
            self.dtype
        )
        input_mask = torch.from_numpy(np.copy(self.input_mask[idx])).to(self.dtype)
        input_mask_2 = torch.from_numpy(np.copy(self.input_mask_2[idx])).to(self.dtype)
        return latent, pooled_prompt_embeds, prompt_embeds, input_mask, input_mask_2
