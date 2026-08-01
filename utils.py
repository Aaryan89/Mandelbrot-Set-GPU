import os
import sys
import time
import math
import argparse
import collections
import threading
import numpy as np
from PIL import Image
from numba import cuda
from typing import List, Tuple, Callable, Optional, Dict, Any

from mandelbrot_kernel import MandelbrotRenderer

class ColorPaletteManager:
    """
    Manages vibrant procedural LUT color palettes for smooth Mandelbrot & Julia set iteration mapping.
    Supported V3 Palettes:
      0: Cyberpunk Neon
      1: Deep Space
      2: Fire / Magma
      3: Emerald Forest
      4: Monochrome Synth
    """
    def __init__(self, lut_size: int = 4096):
        self.lut_size = lut_size
        self.palettes = [
            "Cyberpunk Neon",
            "Deep Space",
            "Fire / Magma",
            "Emerald Forest",
            "Monochrome Synth"
        ]
        self.luts: List[np.ndarray] = []
        self._generate_luts()

    def _generate_luts(self) -> None:
        # 0: Cyberpunk Neon
        cyber_colors = np.array([
            [10, 5, 25],      # Void Indigo
            [138, 43, 226],   # Electric Violet
            [255, 0, 128],    # Neon Pink / Magenta
            [0, 240, 255],    # Cyber Cyan
            [255, 230, 0]     # Neon Yellow
        ], dtype=np.float32)
        lut_cyber = self._interpolate_palette_ramp(cyber_colors, self.lut_size)

        # 1: Deep Space
        space_colors = np.array([
            [4, 4, 16],       # Deep Obsidian
            [55, 10, 110],    # Nebula Purple
            [15, 80, 200],    # Deep Cosmic Blue
            [0, 210, 255],    # Electric Cyan
            [240, 248, 255]   # Starlight Ice
        ], dtype=np.float32)
        lut_space = self._interpolate_palette_ramp(space_colors, self.lut_size)

        # 2: Fire / Magma
        fire_colors = np.array([
            [10, 2, 8],        # Dark interior border
            [160, 12, 10],     # Deep Crimson
            [255, 100, 0],     # Bright Orange
            [255, 215, 0],     # Golden Yellow
            [255, 255, 240]    # White Hot
        ], dtype=np.float32)
        lut_fire = self._interpolate_palette_ramp(fire_colors, self.lut_size)

        # 3: Emerald Forest
        emerald_colors = np.array([
            [4, 20, 12],      # Shadow Moss
            [9, 100, 50],     # Deep Emerald
            [16, 185, 129],   # Electric Mint
            [52, 211, 153],   # Bright Jade
            [236, 253, 245]   # Pale Golden White
        ], dtype=np.float32)
        lut_emerald = self._interpolate_palette_ramp(emerald_colors, self.lut_size)

        # 4: Monochrome Synth
        mono_colors = np.array([
            [12, 12, 15],     # Dark Charcoal
            [50, 55, 65],     # Slate Gray
            [120, 130, 145],  # Cool Gray
            [200, 210, 225],  # Bright Silver
            [255, 255, 255]   # Pure White
        ], dtype=np.float32)
        lut_mono = self._interpolate_palette_ramp(mono_colors, self.lut_size)

        self.luts = [lut_cyber, lut_space, lut_fire, lut_emerald, lut_mono]

    def _interpolate_palette_ramp(self, key_colors: np.ndarray, num_samples: int) -> np.ndarray:
        n_keys = len(key_colors)
        key_pos = np.linspace(0.0, 1.0, n_keys)
        samples_pos = np.linspace(0.0, 1.0, num_samples)

        lut = np.zeros((num_samples, 3), dtype=np.float32)
        for c in range(3):
            lut[:, c] = np.interp(samples_pos, key_pos, key_colors[:, c])
        return lut

    def get_palette_name(self, index: int) -> str:
        return self.palettes[index % len(self.palettes)]

    def map_iterations_to_rgb(self, smooth_iters: np.ndarray, max_iter: int, palette_idx: int,
                              phase_offset: float, out_rgb: np.ndarray) -> None:
        """
        Maps 2D float array of smooth continuous iterations to RGB with phase offset and smooth LUT interpolation.
        """
        lut = self.luts[palette_idx % len(self.luts)]
        lut_len = len(lut)

        clean_iters = np.nan_to_num(smooth_iters, nan=0.0, posinf=float(max_iter), neginf=-1.0)
        exterior_mask = clean_iters > 0.0

        cycle_freq = 0.04
        pos = ((clean_iters * cycle_freq + phase_offset) * lut_len) % lut_len
        idx0 = pos.astype(np.int32) % lut_len
        idx1 = (idx0 + 1) % lut_len
        frac = (pos - np.floor(pos))[:, :, np.newaxis]

        rgb = lut[idx0] * (1.0 - frac) + lut[idx1] * frac

        interior_mask = ~exterior_mask
        if np.any(interior_mask):
            int_vals = np.abs(clean_iters[interior_mask])
            int_glow = np.clip(int_vals * 15.0, 0.0, 45.0)[:, np.newaxis]
            interior_colors = np.array([3, 4, 8], dtype=np.float32) + int_glow * np.array([0.2, 0.3, 0.6], dtype=np.float32)
            rgb[interior_mask] = interior_colors

        out_rgb[...] = np.clip(rgb, 0, 255).astype(np.uint8)


class FPSTracker:
    """Tracks rolling average FPS and 100-frame latency window."""
    def __init__(self, window_size: int = 100, smoothing: float = 0.9):
        self.window_size = window_size
        self.smoothing = smoothing
        self.last_time = time.perf_counter()
        self.frame_time_ms = 0.0
        self.fps = 0.0
        self.initialized = False
        self.latency_window = collections.deque(maxlen=window_size)

    def update(self) -> Tuple[float, float, float, float, float]:
        current_time = time.perf_counter()
        dt = current_time - self.last_time
        self.last_time = current_time

        if dt <= 0:
            dt = 1e-4

        inst_fps = 1.0 / dt
        inst_ms = dt * 1000.0
        self.latency_window.append(inst_ms)

        if not self.initialized:
            self.fps = inst_fps
            self.frame_time_ms = inst_ms
            self.initialized = True
        else:
            self.fps = self.smoothing * self.fps + (1.0 - self.smoothing) * inst_fps
            self.frame_time_ms = self.smoothing * self.frame_time_ms + (1.0 - self.smoothing) * inst_ms

        avg_ms = sum(self.latency_window) / len(self.latency_window) if self.latency_window else inst_ms
        min_ms = min(self.latency_window) if self.latency_window else inst_ms
        max_ms = max(self.latency_window) if self.latency_window else inst_ms

        return self.fps, self.frame_time_ms, avg_ms, min_ms, max_ms


def compute_performance_metrics(width: int, height: int, max_iter: int, frame_ms: float) -> Tuple[float, float]:
    """Calculates Megapixels/sec and estimated GFLOPS throughput."""
    dt_sec = max(1e-4, frame_ms / 1000.0)
    mpx_sec = (width * height) / (dt_sec * 1e6)
    gflops = (width * height * max_iter * 8.0) / (dt_sec * 1e9)
    return mpx_sec, gflops


def get_gpu_memory_info(backend_name: str = "") -> str:
    """Safely retrieves string describing active Numba CUDA VRAM status."""
    try:
        if cuda.is_available():
            dev = cuda.get_current_device()
            dev_name = dev.name.decode('utf-8') if isinstance(dev.name, bytes) else str(dev.name)
            ctx = cuda.current_context()
            free_b, total_b = ctx.get_memory_info()
            used_b = total_b - free_b
            return f"{dev_name} | VRAM: {used_b / (1024**3):.2f}/{total_b / (1024**3):.2f} GB ({free_b / (1024**3):.2f} GB Free)"
        else:
            return "Active Host Memory (CPU Acceleration)"
    except Exception:
        return "NVIDIA CUDA GPU Active"


def export_video_async(frames: List[np.ndarray], output_path: str, fps: int = 30,
                       status_callback: Optional[Callable[[bool, str], None]] = None) -> None:
    """Asynchronously exports recorded RGB numpy frames to MP4 video."""
    def worker():
        try:
            import imageio
            writer = imageio.get_writer(output_path, fps=fps, quality=8, macro_block_size=None)
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            if status_callback:
                status_callback(True, output_path)
        except Exception as e1:
            try:
                import cv2
                if len(frames) > 0:
                    h, w, _ = frames[0].shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                    for frame in frames:
                        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        out.write(bgr)
                    out.release()
                    if status_callback:
                        status_callback(True, output_path)
            except Exception as e2:
                if status_callback:
                    status_callback(False, f"Export error: {e1} / {e2}")

    t = threading.Thread(target=worker, daemon=True)
    t.start()


def run_cli_benchmark(output_md: str = "BENCHMARK_RESULTS.md") -> None:
    """
    Executes automated headless compute benchmark across FP32/FP64 modes and resolutions.
    Outputs a formatted markdown table to BENCHMARK_RESULTS.md.
    """
    print("=" * 70)
    print("  AUTOMATED GPU MANDELBROT BENCHMARK SUITE")
    print("=" * 70)

    resolutions = [
        ("720p HD", 1280, 720),
        ("1080p Full HD", 1920, 1080),
        ("1440p QHD", 2560, 1440),
        ("4K UHD", 3840, 2160)
    ]
    precisions = ["float32", "float64"]

    palette_mgr = ColorPaletteManager()
    results = []

    for name, w, h in resolutions:
        renderer = MandelbrotRenderer(width=w, height=h)
        for prec in precisions:
            print(f"\n[Benchmarking] Resolution: {name} ({w}x{h}) | Precision: {prec.upper()}...")
            min_x, max_x = -2.0, 0.5
            min_y, max_y = -1.25, 1.25
            max_iter = 500

            # Warmup launch
            out_buf = np.zeros((w, h, 3), dtype=np.uint8)
            renderer.render(min_x, max_x, min_y, max_y, max_iter, precision_mode=prec,
                            palette_mgr=palette_mgr, palette_idx=0, out_rgb=out_buf)

            # Benchmark 15 frames iteration loop
            num_frames = 15
            start_t = time.perf_counter()
            frame_times = []

            for _ in range(num_frames):
                f_start = time.perf_counter()
                renderer.render(min_x, max_x, min_y, max_y, max_iter, precision_mode=prec,
                                palette_mgr=palette_mgr, palette_idx=0, out_rgb=out_buf)
                f_end = time.perf_counter()
                frame_times.append((f_end - f_start) * 1000.0)

            total_t = time.perf_counter() - start_t
            avg_fps = num_frames / total_t
            min_frame_ms = min(frame_times)
            avg_frame_ms = sum(frame_times) / len(frame_times)
            mpx_sec, gflops = compute_performance_metrics(w, h, max_iter, avg_frame_ms)

            print(f"  -> Avg FPS: {avg_fps:.1f} | Min Latency: {min_frame_ms:.2f} ms | Output: {mpx_sec:.1f} MP/s | GFLOPS: {gflops:.2f}")

            results.append({
                "resolution": f"{name} ({w}x{h})",
                "precision": prec.upper(),
                "avg_fps": f"{avg_fps:.1f}",
                "min_ms": f"{min_frame_ms:.2f}",
                "mpx_sec": f"{mpx_sec:.1f}",
                "gflops": f"{gflops:.2f}"
            })


    # Write BENCHMARK_RESULTS.md table
    md_content = []
    md_content.append("# GPU Mandelbrot Renderer V3 - Benchmark Results\n")
    md_content.append(f"**Hardware Device**: {get_gpu_memory_info()}\n")
    md_content.append(f"**Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    md_content.append("| Resolution | Precision | Avg FPS | Min Frame Time (ms) | Compute Output (MP/s) | GFLOPS |")
    md_content.append("|---|---|---|---|---|---|")

    for r in results:
        md_content.append(f"| {r['resolution']} | {r['precision']} | {r['avg_fps']} | {r['min_ms']} | {r['mpx_sec']} | {r['gflops']} |")

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))

    print("\n" + "=" * 70)
    print(f" SUCCESS: Benchmark complete! Results exported to: {output_md}")
    print("=" * 70)


def generate_readme_assets(output_dir: str = "docs") -> None:
    """
    Batch-renders 3 ultra-high-res 4K PNG screenshots across different color palettes
    and saves them in the docs/ folder for repository README documentation.
    """
    print("=" * 70)
    print("  GENERATING 4K README ASSETS IN docs/")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)
    w, h = 3840, 2160
    renderer = MandelbrotRenderer(width=w, height=h)
    palette_mgr = ColorPaletteManager()

    targets = [
        ("mandelbrot_cyberpunk_4k.png", 0, -0.7436438870371587, 0.1318259042053119, 0.005, "Cyberpunk Neon"),
        ("mandelbrot_fire_4k.png", 2, -0.75, 0.0, 2.5, "Fire / Magma"),
        ("mandelbrot_emerald_4k.png", 3, -0.7436438870371587, 0.1318259042053119, 0.0001, "Emerald Forest")
    ]

    for fname, pal_idx, cx, cy, vw, pname in targets:
        print(f"[Rendering 4K Screenshot] {fname} ({pname} Palette)...")
        vh = vw * (h / w)
        min_x, max_x = cx - vw / 2.0, cx + vw / 2.0
        min_y, max_y = cy - vh / 2.0, cy + vh / 2.0

        out_rgb = np.zeros((w, h, 3), dtype=np.uint8)
        renderer.render(min_x, max_x, min_y, max_y, max_iter=2000, precision_mode="float64",
                        palette_mgr=palette_mgr, palette_idx=pal_idx, phase_offset=0.0, out_rgb=out_rgb)

        # Transpose HxWxC for image saving
        img_arr = np.transpose(out_rgb, (1, 0, 2))
        img = Image.fromarray(img_arr)
        save_path = os.path.join(output_dir, fname)
        img.save(save_path)
        print(f"  -> Saved: {save_path}")

    print("\n" + "=" * 70)
    print(f" SUCCESS: 4K README assets created in '{output_dir}/'")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Mandelbrot Renderer Utility Suite")
    parser.add_argument("--benchmark", action="store_true", help="Run automated CLI benchmark suite")
    parser.add_argument("--generate-readme-assets", action="store_true", help="Generate 4K screenshots in docs/")
    args = parser.parse_args()

    if args.benchmark:
        run_cli_benchmark()
    elif args.generate_readme_assets:
        generate_readme_assets()
    else:
        parser.print_help()
