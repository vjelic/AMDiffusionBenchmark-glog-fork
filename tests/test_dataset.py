import argparse
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch

# Import the functions and classes to be tested
from src.data.cache_dataset import CacheDataset, create_or_append_to_xr_dataset
from src.data.datasets_utils import (
    collate_fn,
    preprocess_train,
    tokenize_captions,
)


def test_tokenize_captions():
    """
    Verify that the captions are properly transformed before being tokenized.
    Check that the output ids are properly extracted through the input_ids attribute.
    """
    # Create two mock images with captions as filenames
    image1 = MagicMock()
    image1.filename = "example_caption.png"
    image2 = MagicMock()
    image2.filename = "example_caption_two.png"
    batch = {"image": (image1, image2)}

    # Create 2 mock tokenizers that create the text embedding
    tokenizer_1 = MagicMock()
    output_object_1 = lambda x: None
    output_object_1.input_ids = torch.randn(2, 48)
    tokenizer_1.return_value = output_object_1

    tokenizer_2 = MagicMock()
    output_object_2 = lambda x: None
    output_object_2.input_ids = torch.randn(2, 16, 48)
    tokenizer_2.return_value = output_object_2

    # Check that the code executes correctly
    tokenize_captions(batch, tokenizer_1, tokenizer_2)

    # Check that the tokenizers was called with the correct captions
    assert tokenizer_1.call_args[0][0] == ["example caption", "example caption two"]
    assert tokenizer_2.call_args[0][0] == ["example caption", "example caption two"]


def test_preprocess_train():
    """
    Check that the images and captions are properly transformed according to the input transform.
    Check that the expected output shapes align with the inputs.
    """
    # Create dummy args
    dummy_args = argparse.Namespace(num_frames=9)
    # Create two mock images with captions as filenames and a RGB method that creates an image
    image1 = MagicMock()
    image1.filename = "example_caption.png"
    image1.convert = lambda x: torch.randn(
        1, 3, 16, 16
    )  # Additonal dimension to account for the use of interpolate instead of torchvision

    image2 = MagicMock()
    image2.filename = "example_caption_two.png"
    image2.convert = lambda x: torch.randn(1, 3, 16, 16)
    batch = {"image": (image1, image2)}

    # Create 2 mock tokenizers that create the text embedding
    tokenizer_1 = MagicMock()
    output_object_1 = lambda x: None
    output_object_1.input_ids = torch.randn(2, 48)
    tokenizer_1.return_value = output_object_1

    tokenizer_2 = MagicMock()
    output_object_2 = lambda x: None
    output_object_2.input_ids = torch.randn(2, 16, 48)
    tokenizer_2.return_value = output_object_2

    # Create a resize transform
    transform = lambda x: torch.nn.functional.interpolate(
        x, size=(4, 4), mode="bilinear", align_corners=False
    )

    # Check that the code executes correctly
    outputs = preprocess_train(
        examples=batch,
        args=dummy_args,
        tokenizer=tokenizer_1,
        tokenizer_2=tokenizer_2,
        train_transforms=transform,
    )

    # Check that the tokenizers was called with the correct captions
    assert tokenizer_1.call_args[0][0] == ["example caption", "example caption two"]
    assert tokenizer_2.call_args[0][0] == ["example caption", "example caption two"]

    # Check that the putputs are the correct shape
    assert outputs["input_ids"].shape == (2, 48)
    assert outputs["input_ids_2"].shape == (2, 16, 48)
    for i in range(2):
        assert outputs["pixel_values"][i].shape == (
            1,
            3,
            4,
            4,
        ), f"Image {i} not transformed correctly"

    # Check that no transforms yields expected outputs
    outputs = preprocess_train(
        examples=batch,
        args=dummy_args,
        tokenizer=tokenizer_1,
        tokenizer_2=tokenizer_2,
    )

    for _output in outputs["pixel_values"]:
        assert _output.shape == (1, 3, 16, 16)

    # Test on videos
    # Create two mock videos with captions as filenames and a RGB method that creates an image
    video1 = MagicMock()
    video1.get_batch.return_value.asnumpy.return_value = torch.randn(
        9, 3, 16, 16
    ).numpy()
    video1.__len__ = lambda _: 9
    video2 = MagicMock()
    video2.get_batch.return_value.asnumpy.return_value = torch.randn(
        9, 3, 16, 16
    ).numpy()
    video2.__len__ = lambda _: 9
    batch = {"video": (video1, video2), "prompt": ("video 1", "video 2")}
    image_processor = MagicMock()
    image_processor.return_value = {"pixel_values": torch.randn(1, 16, 4, 4)}

    outputs = preprocess_train(
        examples=batch,
        args=dummy_args,
        tokenizer=tokenizer_1,
        tokenizer_2=tokenizer_2,
        image_processor=image_processor,
    )
    for _output in outputs["pixel_values"]:
        assert _output.shape == (9, 3, 16, 16)
    for _output in outputs["processed_image"]:
        assert _output.shape == (16, 4, 4)


def test_collate_fn():
    """
    Verify that the inputs are properly collated by checking instance type and shapes.
    """
    # Create test data
    examples = []

    for _ in range(3):
        sample_dict = {
            "input_ids": torch.randn(48),
            "input_ids_2": torch.randn(16, 48),
            "input_mask": torch.randn(48),
            "input_mask_2": torch.randn(16, 48),
            "pixel_values": torch.randn(3, 4, 4),
        }
        examples.append(sample_dict)

    # Collate outputs
    collated_outputs = collate_fn(examples)

    # Check output shapes and types
    assert collated_outputs["input_ids"].shape == (3, 48)
    assert collated_outputs["input_ids_2"].shape == (3, 16, 48)
    assert collated_outputs["input_mask"].shape == (3, 48)
    assert collated_outputs["input_mask_2"].shape == (3, 16, 48)
    assert collated_outputs["pixel_values"].shape == (3, 3, 4, 4)

    # Check outputs
    assert isinstance(collated_outputs["input_ids"], torch.Tensor)
    assert isinstance(collated_outputs["input_ids_2"], torch.Tensor)
    assert isinstance(collated_outputs["pixel_values"], torch.Tensor)


def test_cache_dataset(tmp_path: Path):
    """
    Verify that the dataset class is capable of streaming the information in saved python files.
    Check that the saved data aligns with the loaded data.
    """
    # Mock data
    cached_data = dict(
        latents=np.random.normal(size=(3, 3, 16, 16)),
        text_enc=np.random.normal(size=(3, 48)),
        text_enc2=np.random.normal(size=(3, 16, 48)),
        mask=np.random.normal(size=(3, 48)),
        mask_2=np.random.normal(size=(3, 16, 48)),
    )
    # Save data for dataset to load
    cache_file = tmp_path / "cached_data.zarr"
    create_or_append_to_xr_dataset(cached_data, cache_file=cache_file)
    # Check if data was successfully loaded
    dataset = CacheDataset(cache_file, torch.float32)
    assert len(dataset) == 3

    # Check if data can be properly accessed
    for i, batch in enumerate(dataset):
        for key, value in cached_data.items():
            assert key in batch.keys()
            assert value[i].shape == batch[key].shape
