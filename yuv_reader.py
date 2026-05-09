#!/usr/bin/env python3
import argparse
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VideoSpec:
    width: int
    height: int
    fps: float = 30.0
    pixel_format: str = "yuv420p"

    @property
    def bytes_per_frame(self) -> int:
        if self.pixel_format != "yuv420p":
            raise ValueError(f"Unsupported pixel format: {self.pixel_format}")
        return self.width * self.height * 3 // 2


class YUVReader:
    def __init__(self, file_path: Path, spec: VideoSpec) -> None:
        self.file_path = Path(file_path)
        self.spec = spec
        self.file_size = self.file_path.stat().st_size
        self.frame_count = self.file_size // self.spec.bytes_per_frame
        self.remainder = self.file_size % self.spec.bytes_per_frame

    def read_frame(self, frame_index: int) -> tuple[bytes, bytes, bytes]:
        if frame_index < 0 or frame_index >= self.frame_count:
            raise IndexError(
                f"Frame index {frame_index} out of range [0, {self.frame_count - 1}]"
            )

        w = self.spec.width
        h = self.spec.height
        y_size = w * h
        uv_size = (w // 2) * (h // 2)
        offset = frame_index * self.spec.bytes_per_frame

        with self.file_path.open("rb") as f:
            f.seek(offset)
            y = f.read(y_size)
            u = f.read(uv_size)
            v = f.read(uv_size)

        if len(y) != y_size or len(u) != uv_size or len(v) != uv_size:
            raise ValueError("Failed to read complete YUV frame from file")

        return y, u, v


def _clamp_u8(x: int) -> int:
    if x < 0:
        return 0
    if x > 255:
        return 255
    return x


def yuv420p_to_rgb24(width: int, height: int, y: bytes, u: bytes, v: bytes) -> bytes:
    rgb = bytearray(width * height * 3)
    uv_width = width // 2

    out_idx = 0
    for j in range(height):
        y_row = j * width
        uv_row = (j // 2) * uv_width
        for i in range(width):
            yv = y[y_row + i]
            uv_idx = uv_row + (i // 2)
            uu = u[uv_idx]
            vv = v[uv_idx]

            c = yv - 16
            d = uu - 128
            e = vv - 128

            if c < 0:
                c = 0

            r = (298 * c + 409 * e + 128) >> 8
            g = (298 * c - 100 * d - 208 * e + 128) >> 8
            b = (298 * c + 516 * d + 128) >> 8

            rgb[out_idx] = _clamp_u8(r)
            rgb[out_idx + 1] = _clamp_u8(g)
            rgb[out_idx + 2] = _clamp_u8(b)
            out_idx += 3

    return bytes(rgb)


def write_ppm(path: Path, width: int, height: int, rgb24: bytes) -> None:
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    with Path(path).open("wb") as f:
        f.write(header)
        f.write(rgb24)


def parse_spec_from_filename(file_name: str) -> VideoSpec | None:
    pattern = r"(?P<w>\d+)x(?P<h>\d+)_(?P<fps>\d+(?:\.\d+)?)fps_(?P<subsample>420|422|444)_(?P<bitdepth>\d+)bit"
    match = re.search(pattern, file_name)
    if not match:
        return None

    subsample = match.group("subsample")
    bitdepth = match.group("bitdepth")
    if subsample != "420" or bitdepth != "8":
        return None

    return VideoSpec(
        width=int(match.group("w")),
        height=int(match.group("h")),
        fps=float(match.group("fps")),
        pixel_format="yuv420p",
    )


def resolve_spec(args: argparse.Namespace, file_path: Path) -> VideoSpec:
    if args.width and args.height:
        return VideoSpec(
            width=args.width,
            height=args.height,
            fps=args.fps,
            pixel_format=args.pixel_format,
        )

    guessed = parse_spec_from_filename(file_path.name)
    if guessed:
        return guessed

    raise ValueError(
        "Cannot infer video spec from filename. Please pass --width, --height and optional --fps."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read raw YUV420p files and inspect/export frames.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("input", type=Path, help="Path to .yuv raw file")
        p.add_argument("--width", type=int, default=None, help="Frame width")
        p.add_argument("--height", type=int, default=None, help="Frame height")
        p.add_argument("--fps", type=float, default=30.0, help="Frame rate")
        p.add_argument(
            "--pixel-format",
            default="yuv420p",
            choices=["yuv420p"],
            help="Pixel format (currently only yuv420p is supported)",
        )

    info = sub.add_parser("info", help="Show parsed video info and estimated frame count")
    add_common_flags(info)

    extract = sub.add_parser("extract", help="Extract one frame to a PPM image")
    add_common_flags(extract)
    extract.add_argument("--frame", type=int, default=0, help="0-based frame index")
    extract.add_argument("--out", type=Path, default=Path("frame.ppm"), help="Output PPM file path")

    return parser


def cmd_info(args: argparse.Namespace) -> None:
    spec = resolve_spec(args, args.input)
    reader = YUVReader(args.input, spec)

    print(f"input: {args.input}")
    print(f"size_bytes: {reader.file_size}")
    print(f"resolution: {spec.width}x{spec.height}")
    print(f"fps: {spec.fps}")
    print(f"pixel_format: {spec.pixel_format}")
    print(f"bytes_per_frame: {spec.bytes_per_frame}")
    print(f"frame_count: {reader.frame_count}")
    if reader.remainder:
        print(f"warning: trailing_bytes={reader.remainder} (file may be truncated or padded)")


def cmd_extract(args: argparse.Namespace) -> None:
    spec = resolve_spec(args, args.input)
    reader = YUVReader(args.input, spec)
    y, u, v = reader.read_frame(args.frame)
    rgb = yuv420p_to_rgb24(spec.width, spec.height, y, u, v)
    write_ppm(args.out, spec.width, spec.height, rgb)
    print(f"saved: {args.out}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "info":
        cmd_info(args)
    elif args.command == "extract":
        cmd_extract(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
