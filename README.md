# Video Encoder Starter

This repository now includes:

- Supports `yuv420p` 8-bit raw files (`.yuv`)
- Prints metadata and estimated frame count
- Extracts one frame and writes it as `.ppm` image
- Runs RD experiments with H.264/H.265 using FFmpeg
- Computes PSNR, optional LPIPS, plots RD curves, and computes BD-Rate

## 1) YUV Reader Quick Start

```bash
python yuv_reader.py info Bosphorus_1920x1080_120fps_420_8bit_YUV.yuv
python yuv_reader.py extract Bosphorus_1920x1080_120fps_420_8bit_YUV.yuv --frame 0 --out frame0.ppm
```

If the filename does not contain resolution/fps metadata, pass these args:

```bash
python yuv_reader.py info input.yuv --width 1920 --height 1080 --fps 30
```

## 2) RD Experiment Quick Start

Prerequisites:

- `ffmpeg` in PATH
- Python packages: `numpy`, `matplotlib`
- Optional LPIPS: `torch`, `lpips`

Install dependencies:

```bash
python -m pip install numpy matplotlib
python -m pip install torch lpips
```

Fast smoke test (short run, PSNR only):

```bash
python rd_experiment.py --input-glob "*.yuv" --output-dir outputs --bitrate-start-kbps 200 --bitrate-step-kbps 50 --num-points 4 --max-frames 120
```

Full run with LPIPS:

```bash
python rd_experiment.py --input-glob "*.yuv" --output-dir outputs --bitrate-start-kbps 200 --bitrate-step-kbps 50 --num-points 4 --compute-lpips --lpips-sample-every 30
```

Outputs:

- `outputs/encoded/*.mp4`
- `outputs/rd_results.csv`
- `outputs/bd_rate.csv`
- `outputs/plots/*_rd.png`
