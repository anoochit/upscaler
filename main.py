#!/usr/bin/env python3
"""
ESRGAN / Real-ESRGAN image upscaler.
Usage:
    python upscale.py input.jpg -o out/ -s 4
    python upscale.py ./photos -o ./upscaled -s 2 --face-enhance
"""

import argparse
import os
from pathlib import Path

import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

MODELS = {
    # name: (arch_kwargs, netscale, url)
    "RealESRGAN_x4plus": (
        dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
        4,
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    ),
    "RealESRGAN_x4plus_anime_6B": (
        dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4),
        4,
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    ),
    "RealESRGAN_x2plus": (
        dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
        2,
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    ),
}


def build_upsampler(model_name: str, tile: int, half: bool) -> RealESRGANer:
    arch_kwargs, netscale, url = MODELS[model_name]
    model = RRDBNet(**arch_kwargs)
    return RealESRGANer(
        scale=netscale,
        model_path=url,          # auto-downloads to weights/ on first run
        model=model,
        tile=tile,               # >0 splits image into tiles to save VRAM
        tile_pad=10,
        pre_pad=0,
        half=half,
        gpu_id=None,
    )


def collect_inputs(path: Path):
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in EXTS)


def main():
    ap = argparse.ArgumentParser(description="Upscale images with Real-ESRGAN")
    ap.add_argument("input", type=Path, help="image file or folder")
    ap.add_argument("-o", "--output", type=Path, default=Path("results"))
    ap.add_argument("-n", "--model", default="RealESRGAN_x4plus", choices=list(MODELS))
    ap.add_argument("-s", "--outscale", type=float, default=4, help="final scale factor")
    ap.add_argument("-t", "--tile", type=int, default=0, help="tile size, e.g. 400 for low VRAM")
    ap.add_argument("--face-enhance", action="store_true", help="GFPGAN face restoration")
    ap.add_argument("--suffix", default="_up")
    ap.add_argument("--fp32", action="store_true", help="disable half precision")
    args = ap.parse_args()

    use_cuda = torch.cuda.is_available()
    half = use_cuda and not args.fp32
    print(f"Device: {'CUDA - ' + torch.cuda.get_device_name(0) if use_cuda else 'CPU (slow)'}")

    upsampler = build_upsampler(args.model, args.tile, half)

    face_enhancer = None
    if args.face_enhance:
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
            upscale=args.outscale,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=upsampler,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    files = collect_inputs(args.input)
    if not files:
        raise SystemExit("No images found.")

    for i, f in enumerate(files, 1):
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[{i}/{len(files)}] skip (unreadable): {f.name}")
            continue

        try:
            if face_enhancer:
                _, _, output = face_enhancer.enhance(
                    img, has_aligned=False, only_center_face=False, paste_back=True
                )
            else:
                output, _ = upsampler.enhance(img, outscale=args.outscale)
        except torch.cuda.OutOfMemoryError:
            print("  OOM — retry with --tile 400 (or lower)")
            continue

        ext = ".png" if (img.ndim == 3 and img.shape[2] == 4) else f.suffix
        dst = args.output / f"{f.stem}{args.suffix}{ext}"
        cv2.imwrite(str(dst), output)
        print(f"[{i}/{len(files)}] {f.name} {img.shape[1]}x{img.shape[0]} -> {output.shape[1]}x{output.shape[0]}  ->  {dst.name}")

    print("Done.")


if __name__ == "__main__":
    main()