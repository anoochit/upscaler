# Image Upscaler

A simple command-line image upscaler powered by **Real-ESRGAN** and **PyTorch**.

It supports single images and directories, multiple Real-ESRGAN models, CUDA acceleration, tile-based processing for GPUs with limited VRAM, and optional face restoration with GFPGAN.

## Features

* Upscale individual images or entire directories
* Recursively process images inside directories
* Real-ESRGAN models:

  * `RealESRGAN_x4plus`
  * `RealESRGAN_x4plus_anime_6B`
  * `RealESRGAN_x2plus`
* CUDA acceleration when an NVIDIA GPU is available
* Automatic half-precision processing on CUDA
* CPU fallback
* Tile-based processing for reducing GPU VRAM usage
* Optional GFPGAN face enhancement
* Configurable output scale
* Supports common image formats
* Automatically downloads model weights on first use

## Requirements

* Python 3.12
* PyTorch
* OpenCV
* BasicSR
* Real-ESRGAN

The project currently targets Python 3.12:

```text
Python >= 3.12, < 3.13
```

## Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for Python environment and dependency management.

Clone the repository:

```bash
git clone https://github.com/anoochit/upscaler.git
cd upscaler
```

Create the environment and install dependencies:

```bash
uv sync
```

Run the application with:

```bash
uv run python main.py
```

### Using Python directly

If you prefer to manage the virtual environment yourself:

```bash
python -m venv .venv
```

Activate it:

**Linux / macOS**

```bash
source .venv/bin/activate
```

**Windows**

```powershell
.venv\Scripts\activate
```

Then install the project:

```bash
pip install -e .
```

## Usage

The basic syntax is:

```bash
python main.py INPUT [OPTIONS]
```

Or with `uv`:

```bash
uv run python main.py INPUT [OPTIONS]
```

### Upscale a single image

```bash
uv run python main.py input.jpg
```

The result is written to:

```text
results/input_up.jpg
```

### Specify an output directory

```bash
uv run python main.py input.jpg -o output/
```

### Upscale a directory

```bash
uv run python main.py ./photos -o ./upscaled
```

Directories are searched recursively for supported image files.

### Specify the output scale

```bash
uv run python main.py input.jpg -s 2
```

For example, a 1000 × 1000 image with `-s 4` produces an output of approximately:

```text
4000 × 4000
```

The final output scale is independent of the model's native scale.

## Models

Three Real-ESRGAN models are currently available.

### RealESRGAN_x4plus

The default model for general photographic and natural images.

```bash
uv run python main.py input.jpg \
  --model RealESRGAN_x4plus
```

### RealESRGAN_x4plus_anime_6B

Designed for anime and illustration-style images.

```bash
uv run python main.py input.png \
  --model RealESRGAN_x4plus_anime_6B
```

### RealESRGAN_x2plus

A 2× model suitable when a smaller enlargement is required.

```bash
uv run python main.py input.jpg \
  --model RealESRGAN_x2plus \
  -s 2
```

## GPU Acceleration

The application automatically detects CUDA.

When an NVIDIA GPU is available, the application reports the GPU:

```text
Device: CUDA - NVIDIA GeForce RTX ...
```

Otherwise it falls back to CPU:

```text
Device: CPU (slow)
```

CUDA processing uses half precision by default to reduce GPU memory usage.

### Force FP32

To disable half precision:

```bash
uv run python main.py input.jpg --fp32
```

This can be useful for GPUs or workloads where FP16 processing causes compatibility or quality issues.

## Low VRAM GPUs

Large images can require significant GPU memory.

Use the `--tile` option to split an image into smaller tiles:

```bash
uv run python main.py input.jpg --tile 400
```

If processing runs out of GPU memory, the application suggests:

```text
OOM — retry with --tile 400 (or lower)
```

You can reduce the tile size further:

```bash
uv run python main.py input.jpg --tile 200
```

A smaller tile size generally reduces VRAM requirements at the cost of additional processing overhead.

## Face Enhancement

Face restoration can be enabled with:

```bash
uv run python main.py portrait.jpg --face-enhance
```

This enables **GFPGAN** in addition to Real-ESRGAN.

The background is still processed using the selected Real-ESRGAN model.

For example:

```bash
uv run python main.py ./portraits \
  -o ./enhanced \
  --face-enhance
```

GFPGAN model weights are downloaded automatically when required.

## Output Naming

By default, output files receive the `_up` suffix.

For example:

```text
input.jpg
```

becomes:

```text
input_up.jpg
```

Use `--suffix` to change the suffix:

```bash
uv run python main.py input.jpg --suffix _4x
```

Result:

```text
input_4x.jpg
```

## Supported Image Formats

The following formats are recognized:

```text
.jpg
.jpeg
.png
.bmp
.webp
.tif
.tiff
```

When an input image contains an alpha channel, the output is written as PNG to preserve transparency.

## Command-Line Options

| Option             | Description                       | Default             |
| ------------------ | --------------------------------- | ------------------- |
| `input`            | Image file or directory           | Required            |
| `-o`, `--output`   | Output directory                  | `results`           |
| `-n`, `--model`    | Real-ESRGAN model                 | `RealESRGAN_x4plus` |
| `-s`, `--outscale` | Final output scale                | `4`                 |
| `-t`, `--tile`     | Tile size for reducing VRAM usage | `0`                 |
| `--face-enhance`   | Enable GFPGAN face restoration    | Disabled            |
| `--suffix`         | Output filename suffix            | `_up`               |
| `--fp32`           | Disable FP16 processing           | Disabled            |

View the built-in help:

```bash
uv run python main.py --help
```

## Examples

### General photo

```bash
uv run python main.py photo.jpg -s 4
```

### Anime image

```bash
uv run python main.py anime.png \
  --model RealESRGAN_x4plus_anime_6B \
  -s 4
```

### Low-VRAM GPU

```bash
uv run python main.py photo.jpg \
  --tile 400
```

### Portrait with face restoration

```bash
uv run python main.py portrait.jpg \
  --face-enhance
```

### Batch processing

```bash
uv run python main.py ./photos \
  -o ./upscaled \
  -s 4
```

### Batch processing with face enhancement

```bash
uv run python main.py ./portraits \
  -o ./enhanced \
  --face-enhance \
  --tile 400
```

## Processing Pipeline

The application follows a simple processing pipeline:

```text
Input
  │
  ├── Single image
  │
  └── Directory
        │
        └── Recursive image discovery
                │
                ▼
          Load image with OpenCV
                │
                ▼
          Real-ESRGAN
                │
                ├── Optional GFPGAN
                │
                ▼
          Save upscaled image
```

When face enhancement is enabled:

```text
Input Image
     │
     ▼
  GFPGAN
     │
     ├── Face restoration
     │
     ▼
Real-ESRGAN
     │
     ▼
Output Image
```

## Model Weights

Model weights are downloaded automatically by the underlying Real-ESRGAN and GFPGAN components when they are first required.

The currently configured models are:

| Model                        | Scale | Intended Use            |
| ---------------------------- | ----: | ----------------------- |
| `RealESRGAN_x4plus`          |    4× | General images          |
| `RealESRGAN_x4plus_anime_6B` |    4× | Anime and illustrations |
| `RealESRGAN_x2plus`          |    2× | General 2× upscaling    |
| `GFPGANv1.3`                 |     — | Face restoration        |

Internet access is therefore required the first time a model is downloaded.

## Project Structure

```text
upscaler/
├── main.py
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
└── README.md
```

### `main.py`

Contains the complete command-line application, including:

* Model configuration
* Real-ESRGAN initialization
* Input discovery
* CUDA detection
* Image processing
* GFPGAN integration
* Output generation
* CLI argument parsing

### `pyproject.toml`

Defines the project metadata, Python version requirement, and dependencies.

## Performance Considerations

GPU acceleration is strongly recommended for practical processing speeds.

For large images:

* Use CUDA when available.
* Use `--tile` when GPU memory is insufficient.
* Start with a tile size such as `400`.
* Reduce the tile size if CUDA out-of-memory errors continue.
* Use FP16 unless FP32 is specifically required.

CPU processing is supported but can be significantly slower, particularly for large images or batch processing.

## Troubleshooting

### CUDA out-of-memory

Try a smaller tile size:

```bash
uv run python main.py input.jpg --tile 200
```

or:

```bash
uv run python main.py input.jpg --tile 100
```

### Processing on CPU

The application automatically falls back to CPU when CUDA is unavailable.

Check the startup message:

```text
Device: CPU (slow)
```

For substantially better performance, install a compatible CUDA-enabled PyTorch environment and NVIDIA GPU driver.

### No images found

If processing a directory, make sure it contains one or more supported image formats:

```text
.jpg
.jpeg
.png
.bmp
.webp
.tif
.tiff
```

The directory is searched recursively.

### Image cannot be read

Unreadable images are skipped and reported:

```text
[1/10] skip (unreadable): image.jpg
```

The remaining images continue processing.

## License

This repository does not currently specify a project license.

The project also depends on third-party software and model weights, including Real-ESRGAN, BasicSR, PyTorch, OpenCV, and optionally GFPGAN. Their respective licenses and terms should be reviewed separately.

## Acknowledgements

This project builds on the following open-source projects:

* [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
* [BasicSR](https://github.com/XPixelGroup/BasicSR)
* [GFPGAN](https://github.com/TencentARC/GFPGAN)
* [PyTorch](https://pytorch.org/)
* [OpenCV](https://opencv.org/)

Special thanks to the authors and maintainers of these projects for making high-quality image restoration and super-resolution models available to the community.

## Author

**Anuchit Chalothorn**

GitHub: https://github.com/anoochit

---

Built with Python, PyTorch, OpenCV, and Real-ESRGAN.
