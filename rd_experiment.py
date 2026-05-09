 #!/usr/bin/env python3
import argparse
import csv
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class VideoSpec:
    path: Path
    width: int
    height: int
    fps: float
    pixel_format: str = "yuv420p"

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def bytes_per_frame(self) -> int:
        if self.pixel_format != "yuv420p":
            raise ValueError(f"Unsupported pixel format: {self.pixel_format}")
        return self.width * self.height * 3 // 2

    @property
    def frame_count(self) -> int:
        return self.path.stat().st_size // self.bytes_per_frame


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Command not found: {cmd[0]}") from exc


def assert_ffmpeg_ready() -> None:
    proc = run_cmd(["ffmpeg", "-version"])
    if proc.returncode != 0:
        raise RuntimeError(
            "Cannot run ffmpeg. Please install ffmpeg and ensure it is available in PATH."
        )


def infer_spec_from_name(path: Path, default_fps: float) -> VideoSpec:
    pattern = (
        r"(?P<w>\d+)x(?P<h>\d+)_"
        r"(?P<fps>\d+(?:\.\d+)?)fps_"
        r"(?P<subsample>420|422|444)_"
        r"(?P<bitdepth>\d+)bit"
    )
    match = re.search(pattern, path.name)
    if not match:
        raise ValueError(
            f"Cannot infer metadata from filename: {path.name}. "
            "Expected pattern like 1920x1080_120fps_420_8bit."
        )
    if match.group("subsample") != "420" or match.group("bitdepth") != "8":
        raise ValueError(f"Only 420 8-bit YUV is supported now: {path.name}")
    fps = float(match.group("fps")) if match.group("fps") else default_fps
    return VideoSpec(
        path=path,
        width=int(match.group("w")),
        height=int(match.group("h")),
        fps=fps,
    )


def encode_video(
    spec: VideoSpec,
    codec: str,
    bitrate_kbps: int,
    output_path: Path,
    max_frames: int | None = None,
) -> tuple[float, int]:
    codec_map = {"h264": "libx264", "h265": "libx265"}
    if codec not in codec_map:
        raise ValueError(f"Unsupported codec: {codec}")

    frames = max_frames if max_frames is not None else spec.frame_count
    ff_codec = codec_map[codec]
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        spec.pixel_format,
        "-video_size",
        f"{spec.width}x{spec.height}",
        "-framerate",
        str(spec.fps),
        "-i",
        str(spec.path),
        "-frames:v",
        str(frames),
        "-an",
        "-c:v",
        ff_codec,
        "-b:v",
        f"{bitrate_kbps}k",
        "-maxrate",
        f"{bitrate_kbps}k",
        "-bufsize",
        f"{bitrate_kbps * 2}k",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
    ]
    if codec == "h265":
        cmd.extend(["-x265-params", "log-level=error"])
    cmd.append(str(output_path))

    proc = run_cmd(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed:\n{proc.stderr}")

    duration_s = frames / spec.fps
    actual_bitrate_kbps = output_path.stat().st_size * 8.0 / duration_s / 1000.0
    return actual_bitrate_kbps, frames


def compute_psnr(spec: VideoSpec, encoded_path: Path, frames: int) -> float:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "info",
        "-f",
        "rawvideo",
        "-pixel_format",
        spec.pixel_format,
        "-video_size",
        f"{spec.width}x{spec.height}",
        "-framerate",
        str(spec.fps),
        "-i",
        str(spec.path),
        "-i",
        str(encoded_path),
        "-frames:v",
        str(frames),
        "-lavfi",
        "[0:v][1:v]psnr",
        "-f",
        "null",
        "-",
    ]
    proc = run_cmd(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg psnr failed:\n{proc.stderr}")

    text = proc.stderr + "\n" + proc.stdout
    match = re.search(r"average:([0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        raise RuntimeError("Cannot parse PSNR from ffmpeg output.")
    return float(match.group(1))


def _start_rgb_pipe_from_source(spec: VideoSpec, frames: int) -> subprocess.Popen:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        spec.pixel_format,
        "-video_size",
        f"{spec.width}x{spec.height}",
        "-framerate",
        str(spec.fps),
        "-i",
        str(spec.path),
        "-frames:v",
        str(frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def _start_rgb_pipe_from_encoded(encoded_path: Path, frames: int) -> subprocess.Popen:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(encoded_path),
        "-frames:v",
        str(frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def compute_lpips(
    spec: VideoSpec,
    encoded_path: Path,
    frames: int,
    sample_every: int,
    net: str = "alex",
) -> float:
    try:
        import lpips  # type: ignore
        import torch
    except Exception as exc:
        raise RuntimeError(
            "LPIPS requires dependencies `torch` and `lpips`. "
            "Install them first, or run without --compute-lpips."
        ) from exc

    if sample_every <= 0:
        raise ValueError("sample_every must be > 0")

    loss_fn = lpips.LPIPS(net=net)
    loss_fn.eval()

    src_proc = _start_rgb_pipe_from_source(spec, frames)
    enc_proc = _start_rgb_pipe_from_encoded(encoded_path, frames)
    assert src_proc.stdout is not None
    assert enc_proc.stdout is not None

    frame_size = spec.width * spec.height * 3
    vals: list[float] = []
    idx = 0

    try:
        while True:
            src_bytes = src_proc.stdout.read(frame_size)
            enc_bytes = enc_proc.stdout.read(frame_size)
            if len(src_bytes) < frame_size or len(enc_bytes) < frame_size:
                break

            if idx % sample_every == 0:
                src_np = np.frombuffer(src_bytes, dtype=np.uint8).reshape(spec.height, spec.width, 3)
                enc_np = np.frombuffer(enc_bytes, dtype=np.uint8).reshape(spec.height, spec.width, 3)

                src_t = torch.from_numpy(src_np).permute(2, 0, 1).float() / 127.5 - 1.0
                enc_t = torch.from_numpy(enc_np).permute(2, 0, 1).float() / 127.5 - 1.0

                with torch.no_grad():
                    score = loss_fn(src_t.unsqueeze(0), enc_t.unsqueeze(0)).item()
                vals.append(float(score))
            idx += 1
    finally:
        for proc in (src_proc, enc_proc):
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    if not vals:
        raise RuntimeError("No frames were sampled for LPIPS.")
    return float(np.mean(vals))


def bd_rate(anchor_rates: np.ndarray, anchor_metric: np.ndarray, test_rates: np.ndarray, test_metric: np.ndarray) -> float | None:
    if len(anchor_rates) < 4 or len(test_rates) < 4:
        return None

    a = np.array(sorted(zip(anchor_metric, anchor_rates), key=lambda x: x[0]), dtype=float)
    b = np.array(sorted(zip(test_metric, test_rates), key=lambda x: x[0]), dtype=float)
    qa, ra = a[:, 0], a[:, 1]
    qb, rb = b[:, 0], b[:, 1]

    q_min = max(qa.min(), qb.min())
    q_max = min(qa.max(), qb.max())
    if q_max <= q_min:
        return None

    pa = np.polyfit(qa, np.log(ra), 3)
    pb = np.polyfit(qb, np.log(rb), 3)

    inta = np.polyval(np.polyint(pa), q_max) - np.polyval(np.polyint(pa), q_min)
    intb = np.polyval(np.polyint(pb), q_max) - np.polyval(np.polyint(pb), q_min)
    avg_diff = (intb - inta) / (q_max - q_min)
    return (math.exp(avg_diff) - 1.0) * 100.0


def save_csv(rows: list[dict[str, str | float | int]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video",
        "codec",
        "target_kbps",
        "actual_kbps",
        "frames",
        "psnr_db",
        "lpips",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def unique_ordered(seq: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def plot_rd(rows: list[dict[str, str]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = unique_ordered(r["video"] for r in rows)
    codecs = unique_ordered(r["codec"] for r in rows)

    for video in videos:
        sub = [r for r in rows if r["video"] == video]
        for metric_key, ylabel in [("psnr_db", "PSNR (dB)"), ("lpips", "LPIPS (lower is better)")]:
            has_metric = any(r.get(metric_key, "") not in ("", "nan", "None") for r in sub)
            if not has_metric:
                continue
            plt.figure(figsize=(7.2, 5.0))
            for codec in codecs:
                s = [r for r in sub if r["codec"] == codec and r.get(metric_key, "") not in ("", "nan", "None")]
                if not s:
                    continue
                xs = np.array([float(r["actual_kbps"]) for r in s], dtype=float)
                ys = np.array([float(r[metric_key]) for r in s], dtype=float)
                idx = np.argsort(xs)
                xs, ys = xs[idx], ys[idx]
                plt.plot(xs, ys, marker="o", label=codec)

            plt.title(f"{video} RD Curve ({metric_key})")
            plt.xlabel("Bitrate (kbps)")
            plt.ylabel(ylabel)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"{video}_{metric_key}_rd.png", dpi=200)
            plt.close()


def compute_bd_rate_table(rows: list[dict[str, str]]) -> list[dict[str, str | float]]:
    out: list[dict[str, str | float]] = []
    videos = unique_ordered(r["video"] for r in rows)
    for video in videos:
        h264 = [r for r in rows if r["video"] == video and r["codec"] == "h264"]
        h265 = [r for r in rows if r["video"] == video and r["codec"] == "h265"]
        if len(h264) < 4 or len(h265) < 4:
            continue

        def _arr(items: list[dict[str, str]], key: str) -> tuple[np.ndarray, np.ndarray]:
            rates = np.array([float(i["actual_kbps"]) for i in items], dtype=float)
            metric = np.array([float(i[key]) for i in items], dtype=float)
            return rates, metric

        h264_r, h264_psnr = _arr(h264, "psnr_db")
        h265_r, h265_psnr = _arr(h265, "psnr_db")
        psnr_bd = bd_rate(h264_r, h264_psnr, h265_r, h265_psnr)
        if psnr_bd is not None:
            out.append({"video": video, "metric": "PSNR", "h265_vs_h264_bd_rate_percent": psnr_bd})

        has_lpips = all(i.get("lpips", "") not in ("", "nan", "None") for i in h264 + h265)
        if has_lpips:
            h264_r2, h264_lp = _arr(h264, "lpips")
            h265_r2, h265_lp = _arr(h265, "lpips")
            lpips_bd = bd_rate(h264_r2, -h264_lp, h265_r2, -h265_lp)
            if lpips_bd is not None:
                out.append({"video": video, "metric": "LPIPS", "h265_vs_h264_bd_rate_percent": lpips_bd})
    return out


def save_bd_rate_csv(rows: list[dict[str, str | float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "metric", "h265_vs_h264_bd_rate_percent"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YUV codec RD experiments (H.264/H.265) with PSNR/LPIPS and BD-Rate."
    )
    parser.add_argument("--input-glob", default="*.yuv", help="Glob pattern of input YUV files")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--bitrate-start-kbps", type=int, default=200, help="Start bitrate")
    parser.add_argument("--bitrate-step-kbps", type=int, default=50, help="Step between points")
    parser.add_argument("--num-points", type=int, default=4, help="Number of bitrate points per codec")
    parser.add_argument("--codecs", nargs="+", default=["h264", "h265"], choices=["h264", "h265"])
    parser.add_argument("--default-fps", type=float, default=30.0, help="Fallback fps for filename parsing")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap for fast test")
    parser.add_argument("--compute-lpips", action="store_true", help="Compute LPIPS (requires torch+lpips)")
    parser.add_argument("--lpips-sample-every", type=int, default=30, help="Sample every N frames for LPIPS")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_ffmpeg_ready()

    out_dir = Path(args.output_dir)
    enc_dir = out_dir / "encoded"
    plot_dir = out_dir / "plots"
    enc_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    bitrates = [args.bitrate_start_kbps + i * args.bitrate_step_kbps for i in range(args.num_points)]
    src_paths = sorted(Path(".").glob(args.input_glob))
    if not src_paths:
        raise RuntimeError(f"No input matched: {args.input_glob}")

    rows: list[dict[str, str | float | int]] = []

    for src in src_paths:
        spec = infer_spec_from_name(src, args.default_fps)
        print(f"[video] {spec.name} ({spec.width}x{spec.height}@{spec.fps}fps)")
        for codec in args.codecs:
            for br in bitrates:
                out_file = enc_dir / f"{spec.name}_{codec}_{br}kbps.mp4"
                print(f"  [encode] codec={codec} target={br}kbps")
                actual_kbps, frames = encode_video(spec, codec, br, out_file, max_frames=args.max_frames)

                psnr = compute_psnr(spec, out_file, frames)
                lpips_val: float | None = None
                if args.compute_lpips:
                    lpips_val = compute_lpips(
                        spec=spec,
                        encoded_path=out_file,
                        frames=frames,
                        sample_every=args.lpips_sample_every,
                    )

                row: dict[str, str | float | int] = {
                    "video": spec.name,
                    "codec": codec,
                    "target_kbps": br,
                    "actual_kbps": round(actual_kbps, 4),
                    "frames": frames,
                    "psnr_db": round(psnr, 6),
                    "lpips": round(lpips_val, 6) if lpips_val is not None else "",
                }
                rows.append(row)
                print(
                    f"    actual={row['actual_kbps']}kbps "
                    f"psnr={row['psnr_db']} "
                    f"lpips={row['lpips'] if row['lpips'] != '' else 'skipped'}"
                )

    result_csv = out_dir / "rd_results.csv"
    save_csv(rows, result_csv)

    str_rows = [{k: str(v) for k, v in row.items()} for row in rows]
    plot_rd(str_rows, plot_dir)

    bd_rows = compute_bd_rate_table(str_rows)
    save_bd_rate_csv(bd_rows, out_dir / "bd_rate.csv")

    print(f"[done] results: {result_csv}")
    print(f"[done] plots: {plot_dir}")
    print(f"[done] bd-rate: {out_dir / 'bd_rate.csv'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
