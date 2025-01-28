from unittest.mock import patch, MagicMock
import torch
import numpy as np


# Import the functions and classes to be tested
from src.datasets import (
    CacheDataset,
    cache_collate_fn,
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
    outputs = preprocess_train(batch, tokenizer_1, tokenizer_2, transform)

    # Check that the tokenizers was called with the correct captions
    assert tokenizer_1.call_args[0][0] == ["example caption", "example caption two"]
    assert tokenizer_2.call_args[0][0] == ["example caption", "example caption two"]

    # Check that the putputs are the correct shape
    assert outputs["input_ids"].shape == (2, 48)
    assert outputs["input_ids_2"].shape == (2, 16, 48)
    assert outputs["pixel_values"][0].shape == (1, 3, 4, 4)
    assert outputs["pixel_values"][1].shape == (1, 3, 4, 4)

    # Check that no transforms yields expected outputs
    outputs = preprocess_train(batch, tokenizer_1, tokenizer_2)

    assert outputs["pixel_values"][0].shape == (1, 3, 16, 16)
    assert outputs["pixel_values"][1].shape == (1, 3, 16, 16)


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
            "pixel_values": torch.randn(3, 4, 4),
        }
        examples.append(sample_dict)

    # Collate outputs
    collated_outputs = collate_fn(examples)

    # Check output shapes and types
    assert collated_outputs["input_ids"].shape == (3, 48)
    assert collated_outputs["input_ids_2"].shape == (3, 16, 48)
    assert collated_outputs["pixel_values"].shape == (3, 3, 4, 4)

    # Check outputs
    assert isinstance(collated_outputs["input_ids"], torch.Tensor)
    assert isinstance(collated_outputs["input_ids_2"], torch.Tensor)
    assert isinstance(collated_outputs["pixel_values"], torch.Tensor)


def test_collate_cache_fn():
    """
    Verify that the inputs are properly collated by checking shapes.
    """
    # Create batch input
    batch = []
    for _ in range(3):
        latent = torch.randn(3, 16, 16)
        text_enc = torch.randn(48)
        text_enc2 = torch.randn(16, 48)
        batch.append([latent, text_enc, text_enc2])

    # Collate input
    outputs = cache_collate_fn(batch)

    # Check shapes
    assert outputs["latents"].shape == (3, 3, 16, 16)
    assert outputs["pooled_prompt_embeds"].shape == (3, 48)
    assert outputs["prompt_embeds"].shape == (3, 16, 48)


def test_cache_dataset(tmp_path):
    """
    Verify that the dataset class is capable of streaming the information in saved python files.
    Check that the saved data aligns with the loaded data.
    """
    # Mock data
    latents = torch.randn(3, 3, 16, 16)
    text_enc = torch.randn(3, 48)
    text_enc2 = torch.randn(3, 16, 48)

    # Save data for dataset to load
    np.save(tmp_path / "latents.npy", latents)
    np.save(tmp_path / "pooled_prompt_embeds.npy", text_enc)
    np.save(tmp_path / "prompt_embeds.npy", text_enc2)

    # Check if data was successfully loaded
    dataset = CacheDataset(tmp_path)
    assert len(dataset) == 3

    # Check if data can be properly accessed
    for i, data in enumerate(dataset):
        assert data[0].shape == (3, 16, 16)
        assert (latents[i] == data[0]).all()
        assert data[1].shape == (48,)
        assert (text_enc[i] == data[1]).all()
        assert data[2].shape == (16, 48)
        assert (text_enc2[i] == data[2]).all()
