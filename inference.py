import argparse
import os
import re
from pathlib import Path

import torch
from diffusers.utils import export_to_video, load_image

from src.models import ModelManager
from src.utils import check_gpu_vendor, parse_resolution, safely_eval_as_bool


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
        "--model",
        type=str,
        default=os.getenv("MODEL", "flux-dev"),
        required=False,
        help="Model to use, either flux-dev, hunyuan-video or stable-diffusion-xl.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.getenv("MODEL_PATH", None),
        help=(
            "Path to pretrained model or model identifier from huggingface.co/models. "
            "If None, uses the default model path for the specified model."
        ),
    )
    parser.add_argument(
        "--denoiser_ckpt",
        type=str,
        default=os.getenv("DENOISER_CKPT", None),
        required=False,
        help="Path to fine-tuned (denoiser) model checkpoint.",
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
        "--prompts",
        type=str,
        default=os.getenv("PROMPTS", "A realistic image of a cat"),
        required=False,
        help=(
            "Either a text prompt or a path/url to an image (image-to-asset models). "
            "If a path is provided, the filename stem will be used as the text prompt. "
            "A single prompt/path or multiple prompts/paths seperated via '|' (vertical line)."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=parse_resolution,
        default=parse_resolution(os.getenv("RESOLUTION", "512")),
        help="The resolution (width, height) of the outputs",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="Data type to use in the inference. Options are 'fp32' (default), 'fp16', or 'bf16'.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=int(os.getenv("NUM_STEPS", 28)),
        required=False,
        help="The number of steps to use when generating the output.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=int(os.getenv("NUM_FRAMES", 65)),
        help="Length of the video in number of frames. Only relevant for video models.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.getenv("SEED", 42)),
        required=False,
        help="The seed to use when generating outputs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/inference",
        required=False,
        help="Output path for generated outputs.",
    )

    args = parser.parse_args()
    return args


def preprocess_prompt(prompt: str) -> str:
    """Simply replace all non-alphanumerical characters with whitespace"""
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9,.?!]", " ", prompt))


def main():
    """
    Main function to execute the inference process.

    This function parses the command-line arguments, initializes the a HuggingFace pipeline
    with the provided arguments and generates images based on the provided prompts.

    """
    # Argument parsing
    args = parse_args()
    # Map model tag to model path using ModelManager
    model_class, model_input_type, model_output_type = ModelManager().get_model(
        args.model
    )

    # Precision handling
    dtype = torch.float32
    if args.precision == "fp16":
        dtype = torch.float16
    elif args.precision == "bf16":
        dtype = torch.bfloat16

    # Load the model
    model = model_class(
        args, is_training=False
    )  # Instantiate the model in inference mode
    pipeline = model.pipe.to("cuda", dtype=dtype)

    if args.denoiser_ckpt:  # Load the checkpoint if provided
        finetuned_denoiser = model.denoiser.from_pretrained(args.denoiser_ckpt).to(
            "cuda", dtype=dtype
        )
        if hasattr(pipeline, "transformer"):
            pipeline.transformer = finetuned_denoiser
        else:  # stable-diffusion-xl
            pipeline.unet = finetuned_denoiser

    # Extract prompts seperated via prompt1|prompt2|prompt3|... -> [prompt1, prompt2, prompt3, ...]
    prompts = [p.strip() for p in args.prompts.split("|") if p.strip()]

    # Set the seed
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    # Generate an image for each prompt
    output_dir = Path(args.output_dir, args.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    for prompt in prompts:
        width, height = args.resolution
        pipeline_kwargs = {
            "prompt": preprocess_prompt(prompt),
            "width": width,
            "height": height,
            "num_inference_steps": args.num_steps,
            "generator": generator,
        }
        if model_output_type == "video":
            pipeline_kwargs["num_frames"] = args.num_frames
        if model_input_type == "image":  # wan2.1-i2v
            pipeline_kwargs["image"] = load_image(prompt)
            prompt = pipeline_kwargs["prompt"] = preprocess_prompt(Path(prompt).stem)

        with torch.autocast("cuda", dtype):
            pred = pipeline(**pipeline_kwargs)

        if model_output_type == "image":
            image = pred.images[0]
            image.save(str(output_dir / f"{prompt}.png"))

        elif model_output_type == "video":
            video = pred.frames[0]
            export_to_video(
                video,
                str(output_dir / f"{prompt}.mp4"),
                fps=15,
            )


if __name__ == "__main__":
    main()
