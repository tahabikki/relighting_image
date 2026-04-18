import argparse
import logging
import os
import sys

import torch
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from lbm.inference import evaluate, get_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Relighting inference using LBM model")
parser.add_argument("--source_image", type=str, required=True, help="Path to source image")
parser.add_argument("--output_path", type=str, required=True, help="Output directory path")
parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps (default: 50)")
parser.add_argument("--model_path", type=str, default=None, help="Path to model weights (default: ./models/relighting)")


def main():
    args = parser.parse_args()
    
    # Set model path
    if args.model_path is None:
        args.model_path = os.path.join(os.path.dirname(__file__), "models", "relighting")
    
    if not os.path.exists(args.model_path):
        logger.error(f"Model not found at {args.model_path}")
        return
    
    # Load model
    logger.info(f"Loading relighting model from {args.model_path}...")
    model = get_model(args.model_path, torch_dtype=torch.bfloat16, device="cuda")
    
    # Load source image
    if not os.path.exists(args.source_image):
        logger.error(f"Source image not found: {args.source_image}")
        return
    
    logger.info(f"Loading source image: {args.source_image}")
    source_image = Image.open(args.source_image).convert("RGB")
    
    # Run inference
    logger.info(f"Running inference with {args.num_inference_steps} steps...")
    output_image = evaluate(model, source_image, args.num_inference_steps)
    
    # Save outputs
    os.makedirs(args.output_path, exist_ok=True)
    
    input_name = os.path.splitext(os.path.basename(args.source_image))[0]
    
    existing_files = [f for f in os.listdir(args.output_path) if f.startswith(f"{input_name}_output_image_")]
    version = len(existing_files) + 1
    
    source_path = os.path.join(args.output_path, f"{input_name}_source_image.jpg")
    output_path = os.path.join(args.output_path, f"{input_name}_output_image_{version}.jpg")
    
    source_image.save(source_path)
    output_image.save(output_path)
    
    logger.info(f"✓ Relighting complete!")
    logger.info(f"  Source image: {source_path}")
    logger.info(f"  Output image: {output_path}")


if __name__ == "__main__":
    main()
