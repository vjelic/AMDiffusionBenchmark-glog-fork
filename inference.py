import argparse
import os

import torch
from diffusers import FluxPipeline, FluxTransformer2DModel


def parse_args():
    """Parse the arguments necessary for performing inference/image generation.


    Returns:
        args: The parsed arguments containing the parameters necessary for inference.
    """
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Simple inference script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.getenv("MODEL_PATH", "black-forest-labs/FLUX.1-dev"),
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--transformer_ckpt",
        type=str,
        default=os.getenv("TRANSFORMER_CKPT", None),
        required=False,
        help="Path to fine-tuned model checkpoint.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=os.getenv("PROMPTS", "A realistic image of a cat"),
        required=False,
        help="A single prompt or multiple prompts seperated via :",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=int(os.getenv("RESOLUTION", 512)),
        required=False,
        help="The resolution at which to generate the image.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=int(os.getenv("NUM_STEPS", 28)),
        required=False,
        help="The number of steps to use when generating the image.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=float(os.getenv("GUIDANCE_SCALE", 3.5)),
        required=False,
        help="The guidance scale to use when generating the image.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.getenv("SEED", 42)),
        required=False,
        help="The seed to use when generating images.",
    )

    args = parser.parse_args()
    return args


def main():
    """
    Main function to execute the inference process.

    This function parses the command-line arguments, initializes the a HuggingFace pipeline
    with the provided arguments and generates images based on the provided prompts.

    """
    # Argument parsing
    args = parse_args()

    # Load the model
    pipeline = FluxPipeline.from_pretrained(args.model_path).to("cuda")

    # Load the checkpoint if available
    if args.transformer_ckpt:
        pipeline.transformer = FluxTransformer2DModel.from_pretrained(
            args.transformer_ckpt
        ).to("cuda")

    # Extract prompts seperated primarily via prompt1:prompt2:prompt3
    prompts = args.prompts.split(":")

    # Set the seed
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    # Generate an image for each prompt
    for prompt in prompts:

        image = pipeline(
            prompt,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_steps,
            max_sequence_length=256,
            height=args.resolution,
            width=args.resolution,
            generator=generator,
        ).images[0]

        image.save(f"outputs/inference_images/{prompt}.png")


if __name__ == "__main__":
    main()
