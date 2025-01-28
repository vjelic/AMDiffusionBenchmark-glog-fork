import os
import numpy as np
import torch
from torch.utils.data import Dataset


def tokenize_captions(examples, tokenizer, tokenizer_2):
    """Tokenize captions extracted from image filenames using two tokenizers.

    Args:
        examples (dict): A dictionary containing image data under the key "image".
            Each item in examples["image"] should have a 'filename' attribute.
        tokenizer (Tokenizer): The first tokenizer to process the captions.
        tokenizer_2 (Tokenizer): The second tokenizer to process the captions.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - inputs (torch.Tensor): Tokenized captions from the first tokenizer.
            - inputs_2 (torch.Tensor): Tokenized captions from the second tokenizer.
    """
    captions = []
    for example in examples["image"]:
        path = example.filename
        filename = os.path.splitext(os.path.basename(path))[0]
        caption = filename.replace("_", " ")
        captions.append(caption)

    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_overflowing_tokens=False,
        return_length=False,
    ).input_ids

    inputs_2 = tokenizer_2(
        captions,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    ).input_ids
    return inputs, inputs_2


def preprocess_train(examples, tokenizer, tokenizer_2, train_transforms=None):
    """Preprocess training examples by transforming images and tokenizing captions.

    Args:
        examples (dict): A dictionary containing image data under the key "image".
            Each item in examples["image"] should be a PIL image with a 'filename' attribute.
        tokenizer (Tokenizer): The first tokenizer to process the captions.
        tokenizer_2 (Tokenizer): The second tokenizer to process the captions.
        train_transforms (callable): A function or transform to apply to each image.

    Returns:
        dict: The updated examples dictionary with the following keys added:
            - "pixel_values": List of transformed image tensors.
            - "input_ids": Tensor of tokenized captions from the first tokenizer.
            - "input_ids_2": Tensor of tokenized captions from the second tokenizer.
    """
    images = [image.convert("RGB") for image in examples["image"]]
    if train_transforms:
        images = [train_transforms(image) for image in images]
    examples["pixel_values"] = images

    examples["input_ids"], examples["input_ids_2"] = tokenize_captions(
        examples, tokenizer, tokenizer_2
    )
    return examples


def collate_fn(examples):
    """Collate a list of examples into a batch for DataLoader.

    Args:
        examples (list): A list of examples, where each example is a dictionary containing:
            - "pixel_values": Tensor of transformed image data.
            - "input_ids": Tensor of tokenized captions from the first tokenizer.
            - "input_ids_2": Tensor of tokenized captions from the second tokenizer.

    Returns:
        dict: A dictionary containing batched tensors:
            - "pixel_values" (torch.Tensor): Batched image tensors of shape (batch_size, C, H, W).
            - "input_ids" (torch.Tensor): Batched token IDs from the first tokenizer.
            - "input_ids_2" (torch.Tensor): Batched token IDs from the second tokenizer.
    """
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    input_ids_2 = torch.stack([example["input_ids_2"] for example in examples])
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "input_ids_2": input_ids_2,
    }


def cache_collate_fn(batch):
    # `batch` is a list of items returned by `__getitem__`.
    # If `__getitem__` returns a tuple of (latent, text_encoding), like before:
    latents, pooled_prompt_embeds, prompt_embeds = zip(*batch)

    # Convert to tensors and stack them
    latents = torch.stack(latents, dim=0)
    pooled_prompt_embeds = torch.stack(pooled_prompt_embeds, dim=0)
    prompt_embeds = torch.stack(prompt_embeds, dim=0)

    # Return a dictionary
    return {
        "latents": latents,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "prompt_embeds": prompt_embeds,
    }


class CacheDataset(Dataset):
    """Dataset for loading the cached latents and encodings."""

    def __init__(self, cache_dir):
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

        # Check that all the embeds and latents have the same length
        assert len(self.latents) == len(self.pooled_prompt_embeds) and len(
            self.pooled_prompt_embeds
        ) == len(self.prompt_embeds)

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        # Access individual samples directly by index
        latent = torch.from_numpy(np.copy(self.latents[idx]))
        pooled_prompt_embeds = torch.from_numpy(np.copy(self.pooled_prompt_embeds[idx]))
        prompt_embeds = torch.from_numpy(np.copy(self.prompt_embeds[idx]))
        return latent, pooled_prompt_embeds, prompt_embeds
