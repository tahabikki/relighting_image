# Icao Relighting Project

Standalone relighting inference using LBM (Light Bending Model).

## Quick Start

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
source .venv/bin/activate  # On macOS/Linux

pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Note:** Replace `cu121` with your CUDA version if needed (e.g., `cu122` for CUDA 12.2).

### Usage

```bash
python relighting_inference.py \
    --source_image "path/to/image.jpg" \
    --output_path "path/to/output" \
    --num_inference_steps 50
```

### Arguments

- `--source_image` (required): Path to input image (jpg, png supported)
- `--output_path` (required): Output directory
- `--num_inference_steps` (optional, default=50): Quality level (higher = better but slower)
- `--model_path` (optional): Path to model weights (auto-detected from ./models/relighting)

### Examples

**Fast inference (1 step):**
```bash
python relighting_inference.py \
    --source_image image.jpg \
    --output_path results \
    --num_inference_steps 1
```

**High quality (50 steps):**
```bash
python relighting_inference.py \
    --source_image image.jpg \
    --output_path results \
    --num_inference_steps 50
```

## Project Structure

```
Icao_relighting/
├── relighting_inference.py  # Main inference script
├── requirements.txt          # Dependencies
├── models/
│   └── relighting/          # Model weights (auto-loaded)
│       ├── config.yaml
│       └── model.safetensors
└── src/lbm/                 # Core LBM inference code
    ├── config.py
    ├── inference/
    ├── models/
    └── data/
```

## Output

- `{input_name}_source_image.jpg` - Input image
- `{input_name}_output_image_{n}.jpg` - Relighting result (versioned, won't overwrite)

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- See requirements.txt for dependencies
