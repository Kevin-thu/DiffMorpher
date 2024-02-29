import os
import torch
import numpy as np
import cv2
from PIL import Image
from argparse import ArgumentParser
from model import DiffMorpherPipeline

parser = ArgumentParser()
parser.add_argument(
    "--model_path", type=str, default="stabilityai/stable-diffusion-2-1-base",
    help="Pretrained model to use (default: %(default)s)"
)
parser.add_argument(
    "--image_path_0", type=str, default="",
    help="Path of the first image (default: %(default)s)")
parser.add_argument(
    "--prompt_0", type=str, default="",
    help="Prompt of the second image (default: %(default)s)")
parser.add_argument(
    "--image_path_1", type=str, default="",
    help="Path of the first image (default: %(default)s)")
parser.add_argument(
    "--prompt_1", type=str, default="",
    help="Prompt of the second image (default: %(default)s)")
parser.add_argument(
    "--output_path", type=str, default="./results",
    help="Path of the output image (default: %(default)s)"
)
parser.add_argument(
    "--save_lora_dir", type=str, default="./lora",
    help="Path of the output lora directory (default: %(default)s)"
)
parser.add_argument(
    "--load_lora_path_0", type=str, default="",
    help="Path of the lora directory of the first image (default: %(default)s)"
)
parser.add_argument(
    "--load_lora_path_1", type=str, default="",
    help="Path of the lora directory of the second image (default: %(default)s)"
)
parser.add_argument(
    "--use_adain", action="store_true",
    help="Use AdaIN (default: %(default)s)"
)
parser.add_argument(
    "--use_reschedule",  action="store_true",
    help="Use reschedule sampling (default: %(default)s)"
)
parser.add_argument(
    "--lamb",  type=float, default=0.6,
    help="Lambda for self-attention replacement (default: %(default)s)"
)
parser.add_argument(
    "--fix_lora_value", type=float, default=None,
    help="Fix lora value (default: LoRA Interp., not fixed)"
)
parser.add_argument(
    "--save_inter", action="store_true",
    help="Save intermediate results (default: %(default)s)"
)
parser.add_argument(
    "--num_frames", type=int, default=16,
    help="Number of frames to generate (default: %(default)s)"
)
parser.add_argument(
    "--duration", type=int, default=100,
    help="Duration of each frame (default: %(default)s ms)"
)
parser.add_argument(
    "--no_lora", action="store_true"
)

args = parser.parse_args()

os.makedirs(args.output_path, exist_ok=True)
pipeline = DiffMorpherPipeline.from_pretrained(
    args.model_path, torch_dtype=torch.float32)
pipeline.to("cuda")
images = pipeline(
    img_path_0=args.image_path_0,
    img_path_1=args.image_path_1,
    prompt_0=args.prompt_0,
    prompt_1=args.prompt_1,
    save_lora_dir=args.save_lora_dir,
    load_lora_path_0=args.load_lora_path_0,
    load_lora_path_1=args.load_lora_path_1,
    use_adain=args.use_adain,
    use_reschedule=args.use_reschedule,
    lamd=args.lamb,
    output_path=args.output_path,
    num_frames=args.num_frames,
    fix_lora=args.fix_lora_value,
    save_intermediates=args.save_inter,
    use_lora=not args.no_lora
)
images[0].save(f"{args.output_path}/output.gif", save_all=True,
               append_images=images[1:], duration=args.duration, loop=0)
