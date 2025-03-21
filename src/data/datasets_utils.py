import os
from typing import Any, Iterable

import numpy as np
import torch
import torchvision

from src.utils import safely_eval_as_bool


def tokenize_captions(
    examples: dict, tokenizer, tokenizer_2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    if "prompt" in examples.keys():
        captions = examples["prompt"]
    else:
        for example in examples["image"]:
            path = example.filename
            filename = os.path.splitext(os.path.basename(path))[0]
            caption = filename.replace("_", " ")
            captions.append(caption)

    inputs = tokenizer(captions)
    input_ids = inputs.input_ids
    if hasattr(inputs, "attention_mask"):
        input_mask = inputs.attention_mask
    else:
        input_mask = None

    inputs_2 = tokenizer_2(captions)
    input_ids_2 = inputs_2.input_ids
    if hasattr(inputs_2, "attention_mask"):
        input_mask_2 = inputs_2.attention_mask
    else:
        input_mask_2 = None

    return input_ids, input_mask, input_ids_2, input_mask_2


def preprocess_train(
    examples: dict[str, torch.Tensor],
    tokenizer,
    tokenizer_2,
    train_transforms=None,
    num_frames: int = 30,
) -> dict[str, torch.Tensor]:
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
    if "image" in examples.keys():
        dataset = [image.convert("RGB") for image in examples["image"]]
        if train_transforms:
            dataset = [train_transforms(data) for data in dataset]
    elif "video" in examples.keys():
        dataset = []
        for video in examples["video"]:
            current_length = len(video)

            if num_frames > current_length:
                if safely_eval_as_bool(os.getenv("PAD_VIDEOS_TO_NUM_FRAMES", "false")):
                    # If num_frames is greater than the current length, pad with zeros
                    pad_length = num_frames - current_length
                    zero_frames = np.zeros_like(video.get_batch([0]).asnumpy())
                    video = np.concatenate(
                        [
                            video.get_batch(list(range(current_length))).asnumpy(),
                            np.tile(zero_frames, (pad_length, 1, 1, 1)),
                        ],
                        axis=0,
                    )
                else:
                    raise ValueError(
                        f"num_frames={num_frames} is longer than input video length {current_length}"
                    )
            else:
                video = video.get_batch(list(range(num_frames))).asnumpy()

            if train_transforms:
                video = torch.stack(
                    [
                        train_transforms(torchvision.transforms.ToPILImage()(frame))
                        for frame in video
                    ]
                )

            dataset.append(video)

    examples["pixel_values"] = dataset
    (
        examples["input_ids"],
        examples["input_mask"],
        examples["input_ids_2"],
        examples["input_mask_2"],
    ) = tokenize_captions(examples, tokenizer, tokenizer_2)
    return examples


def collate_fn(examples: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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
    input_mask = torch.stack([example["input_mask"] for example in examples])
    input_mask_2 = torch.stack([example["input_mask_2"] for example in examples])
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "input_mask": input_mask,
        "input_ids_2": input_ids_2,
        "input_mask_2": input_mask_2,
    }


def cache_collate_fn(batch: Iterable[Any]) -> dict[str, torch.Tensor]:
    # `batch` is a list of items returned by `__getitem__`.
    # If `__getitem__` returns a tuple of (latent, text_encoding), like before:
    latents, pooled_prompt_embeds, prompt_embeds, input_mask, input_mask_2 = zip(*batch)
    # Convert to tensors and stack them
    latents = torch.stack(latents, dim=0)
    pooled_prompt_embeds = torch.stack(pooled_prompt_embeds, dim=0)
    prompt_embeds = torch.stack(prompt_embeds, dim=0)
    input_mask = torch.stack(input_mask, dim=0)
    input_mask_2 = torch.stack(input_mask_2, dim=0)
    # Return a dictionary
    return {
        "latents": latents,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "prompt_embeds": prompt_embeds,
        "input_mask": input_mask,
        "input_mask_2": input_mask_2,
    }
