import math
import os
import sys
import ctypes
import numpy as np
from typing import Optional, Tuple, Any

# Setup CUDA Toolkit DLL paths for Numba CUDA on Windows if present
cuda_toolkit_path = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
if os.path.exists(cuda_toolkit_path):
    bin_dir = os.path.join(cuda_toolkit_path, 'bin', 'x64')
    bin_root = os.path.join(cuda_toolkit_path, 'bin')
    nvvm_dir = os.path.join(cuda_toolkit_path, 'nvvm', 'bin', 'x64')
    libdevice_bc = os.path.join(cuda_toolkit_path, 'nvvm', 'libdevice', 'libdevice.10.bc')

    for p in [bin_dir, bin_root, nvvm_dir]:
        if os.path.exists(p):
            try:
                os.add_dll_directory(p)
            except Exception:
                pass

    if os.path.exists(libdevice_bc):
        os.environ['NUMBA_CUDA_LIBDEVICE'] = libdevice_bc

    try:
        import numba.cuda.cudadrv.libs as numba_libs
        import numba.cuda.cudadrv.nvvm as nvvm
        import numba.cuda.cudadrv.driver as driver

        nvvm_dll = os.path.join(nvvm_dir, 'nvvm64_40_0.dll')
        if os.path.exists(nvvm_dll):
            nvvm.open_cudalib = lambda lib, track=True: ctypes.CDLL(nvvm_dll)

        if os.path.exists(libdevice_bc):
            with open(libdevice_bc, 'rb') as f:
                bc_content = f.read()
            nvvm.get_libdevice = lambda: libdevice_bc
            nvvm.open_libdevice = lambda: bc_content

        # Map compute capability 12.0 -> 8.0 for NVVM compilation compatibility
        orig_find_arch = nvvm.find_closest_arch
        def patched_find_arch(cc):
            if cc[0] >= 10:
                return (8, 0)
            return orig_find_arch(cc)
        nvvm.find_closest_arch = patched_find_arch

        # Patch NVVM IR compiler to output supported PTX version (9.1)
        orig_compile_ir = nvvm.compile_ir
        def patched_compile_ir(irs, **options):
            ptx = orig_compile_ir(irs, **options)
            if isinstance(ptx, str):
                ptx = ptx.replace('.version 9.3', '.version 9.1').replace('.version 9.2', '.version 9.1')
            elif isinstance(ptx, bytes):
                ptx = ptx.replace(b'.version 9.3', b'.version 9.1').replace(b'.version 9.2', b'.version 9.1')
            return ptx
        nvvm.compile_ir = patched_compile_ir

        # Patch CUDA Linker add_ptx as double safeguard
        orig_add_ptx = driver.Linker.add_ptx
        def patched_add_ptx(self, ptx, name='<string>'):
            if isinstance(ptx, str):
                ptx = ptx.replace('.version 9.3', '.version 9.1').replace('.version 9.2', '.version 9.1')
            elif isinstance(ptx, bytes):
                ptx = ptx.replace(b'.version 9.3', b'.version 9.1').replace(b'.version 9.2', b'.version 9.1')
            return orig_add_ptx(self, ptx, name=name)
        driver.Linker.add_ptx = patched_add_ptx

    except Exception as e:
        print(f"[CUDA Initialization] Environment patch warning: {e}")

from numba import cuda


def compute_auto_max_iter(base_iter: int, min_x: float, max_x: float) -> int:
    """
    Computes zoom-adaptive max_iter based on zoom level.
    Clamped strictly between 200 and 50,000.
    """
    current_zoom_level = 3.5 / max(1e-18, float(max_x - min_x))
    log_zoom = math.log10(max(1.0, current_zoom_level))
    scaled = int(base_iter + 150 * log_zoom)
    return max(200, min(50000, scaled))


# -------------------------------------------------------------------------
# Float32 Mandelbrot CUDA Kernel
# -------------------------------------------------------------------------
@cuda.jit
def mandelbrot_cuda_f32_kernel(w, h, min_x, max_x, min_y, max_y, max_iter, out_iters):
    x, y = cuda.grid(2)
    if x >= w or y >= h:
        return

    dx = (max_x - min_x) / float(w)
    dy = (max_y - min_y) / float(h)
    cr = min_x + float(x) * dx
    ci = min_y + float(y) * dy

    zr = 0.0
    zi = 0.0
    n = 0
    while zr * zr + zi * zi <= 4.0 and n < max_iter:
        temp = zr * zr - zi * zi + cr
        zi = 2.0 * zr * zi + ci
        zr = temp
        n += 1

    if n < max_iter:
        zi = 2.0 * zr * zi + ci
        zr = zr * zr - zi * zi + cr
        n += 1
        zi = 2.0 * zr * zi + ci
        zr = zr * zr - zi * zi + cr
        n += 1

        mod_sq = zr * zr + zi * zi
        if mod_sq > 1.0:
            log_zn = math.log(mod_sq) / 2.0
            nu = math.log(log_zn / 0.6931471805599453) / 0.6931471805599453
            out_iters[x, y] = float(n) + 1.0 - nu
        else:
            out_iters[x, y] = float(n)
    else:
        mod_sq = zr * zr + zi * zi
        out_iters[x, y] = -1.0 - math.sqrt(mod_sq)


# -------------------------------------------------------------------------
# Float64 Mandelbrot CUDA Kernel
# -------------------------------------------------------------------------
@cuda.jit
def mandelbrot_cuda_f64_kernel(w, h, min_x, max_x, min_y, max_y, max_iter, out_iters):
    x, y = cuda.grid(2)
    if x >= w or y >= h:
        return

    dx = (max_x - min_x) / float(w)
    dy = (max_y - min_y) / float(h)
    cr = min_x + float(x) * dx
    ci = min_y + float(y) * dy

    zr = 0.0
    zi = 0.0
    n = 0
    while zr * zr + zi * zi <= 4.0 and n < max_iter:
        temp = zr * zr - zi * zi + cr
        zi = 2.0 * zr * zi + ci
        zr = temp
        n += 1

    if n < max_iter:
        zi = 2.0 * zr * zi + ci
        zr = zr * zr - zi * zi + cr
        n += 1
        zi = 2.0 * zr * zi + ci
        zr = zr * zr - zi * zi + cr
        n += 1

        mod_sq = zr * zr + zi * zi
        if mod_sq > 1.0:
            log_zn = math.log(mod_sq) / 2.0
            nu = math.log(log_zn / 0.6931471805599453) / 0.6931471805599453
            out_iters[x, y] = float(n) + 1.0 - nu
        else:
            out_iters[x, y] = float(n)
    else:
        mod_sq = zr * zr + zi * zi
        out_iters[x, y] = -1.0 - math.sqrt(mod_sq)


# -------------------------------------------------------------------------
# Float32 Julia Set CUDA Kernel
# -------------------------------------------------------------------------
@cuda.jit
def julia_cuda_f32_kernel(w, h, min_x, max_x, min_y, max_y, c_re, c_im, max_iter, out_iters):
    x, y = cuda.grid(2)
    if x >= w or y >= h:
        return

    dx = (max_x - min_x) / float(w)
    dy = (max_y - min_y) / float(h)
    zr = min_x + float(x) * dx
    zi = min_y + float(y) * dy

    cr = float(c_re)
    ci = float(c_im)

    n = 0
    while zr * zr + zi * zi <= 4.0 and n < max_iter:
        temp = zr * zr - zi * zi + cr
        zi = 2.0 * zr * zi + ci
        zr = temp
        n += 1

    if n < max_iter:
        zi = 2.0 * zr * zi + ci
        zr = zr * zr - zi * zi + cr
        n += 1
        zi = 2.0 * zr * zi + ci
        zr = zr * zr - zi * zi + cr
        n += 1

        mod_sq = zr * zr + zi * zi
        if mod_sq > 1.0:
            log_zn = math.log(mod_sq) / 2.0
            nu = math.log(log_zn / 0.6931471805599453) / 0.6931471805599453
            out_iters[x, y] = float(n) + 1.0 - nu
        else:
            out_iters[x, y] = float(n)
    else:
        mod_sq = zr * zr + zi * zi
        out_iters[x, y] = -1.0 - math.sqrt(mod_sq)


# -------------------------------------------------------------------------
# Float64 Julia Set CUDA Kernel
# -------------------------------------------------------------------------
@cuda.jit
def julia_cuda_f64_kernel(w, h, min_x, max_x, min_y, max_y, c_re, c_im, max_iter, out_iters):
    x, y = cuda.grid(2)
    if x >= w or y >= h:
        return

    dx = (max_x - min_x) / float(w)
    dy = (max_y - min_y) / float(h)
    zr = min_x + float(x) * dx
    zi = min_y + float(y) * dy

    cr = float(c_re)
    ci = float(c_im)

    n = 0
    while zr * zr + zi * zi <= 4.0 and n < max_iter:
        temp = zr * zr - zi * zi + cr
        zi = 2.0 * zr * zi + ci
        zr = temp
        n += 1

    if n < max_iter:
        zi = 2.0 * zr * zi + ci
        zr = zr * zr - zi * zi + cr
        n += 1
        zi = 2.0 * zr * zi + ci
        zr = zr * zr - zi * zi + cr
        n += 1

        mod_sq = zr * zr + zi * zi
        if mod_sq > 1.0:
            log_zn = math.log(mod_sq) / 2.0
            nu = math.log(log_zn / 0.6931471805599453) / 0.6931471805599453
            out_iters[x, y] = float(n) + 1.0 - nu
        else:
            out_iters[x, y] = float(n)
    else:
        mod_sq = zr * zr + zi * zi
        out_iters[x, y] = -1.0 - math.sqrt(mod_sq)


def mandelbrot_cpu_vectorized(w: int, h: int, min_x: float, max_x: float, min_y: float, max_y: float,
                             max_iter: int, precision_mode: str = "float32") -> np.ndarray:
    dtype = np.float64 if precision_mode == "float64" else np.float32
    x = np.linspace(min_x, max_x, w, dtype=dtype)
    y = np.linspace(min_y, max_y, h, dtype=dtype)
    X, Y = np.meshgrid(x, y, indexing='ij')
    c = X + 1j * Y
    z = np.zeros_like(c)
    smooth_iters = np.zeros((w, h), dtype=np.float32)
    escaped = np.zeros((w, h), dtype=bool)

    for n in range(max_iter):
        mask = ~escaped
        if not np.any(mask):
            break
        z[mask] = z[mask]**2 + c[mask]
        new_escaped = mask & (np.real(z[mask])**2 + np.imag(z[mask])**2 > 4.0)
        if np.any(new_escaped):
            mod_sq = np.real(z[new_escaped])**2 + np.imag(z[new_escaped])**2
            log_zn = np.log(np.maximum(1e-10, mod_sq)) / 2.0
            nu = np.log(np.maximum(1e-10, log_zn / 0.6931471805599453)) / 0.6931471805599453
            smooth_iters[new_escaped] = (float(n) + 1.0) + 1.0 - nu
            escaped[new_escaped] = True

    interior_mask = ~escaped
    if np.any(interior_mask):
        mod_sq = np.real(z[interior_mask])**2 + np.imag(z[interior_mask])**2
        smooth_iters[interior_mask] = -1.0 - np.sqrt(mod_sq)

    return smooth_iters


def julia_cpu_vectorized(w: int, h: int, min_x: float, max_x: float, min_y: float, max_y: float,
                        c_re: float, c_im: float, max_iter: int, precision_mode: str = "float32") -> np.ndarray:
    dtype = np.float64 if precision_mode == "float64" else np.float32
    x = np.linspace(min_x, max_x, w, dtype=dtype)
    y = np.linspace(min_y, max_y, h, dtype=dtype)
    X, Y = np.meshgrid(x, y, indexing='ij')
    z = X + 1j * Y
    c = complex(c_re, c_im)
    smooth_iters = np.zeros((w, h), dtype=np.float32)
    escaped = np.zeros((w, h), dtype=bool)

    for n in range(max_iter):
        mask = ~escaped
        if not np.any(mask):
            break
        z[mask] = z[mask]**2 + c
        new_escaped = mask & (np.real(z[mask])**2 + np.imag(z[mask])**2 > 4.0)
        if np.any(new_escaped):
            mod_sq = np.real(z[new_escaped])**2 + np.imag(z[new_escaped])**2
            log_zn = np.log(np.maximum(1e-10, mod_sq)) / 2.0
            nu = np.log(np.maximum(1e-10, log_zn / 0.6931471805599453)) / 0.6931471805599453
            smooth_iters[new_escaped] = (float(n) + 1.0) + 1.0 - nu
            escaped[new_escaped] = True

    interior_mask = ~escaped
    if np.any(interior_mask):
        mod_sq = np.real(z[interior_mask])**2 + np.imag(z[interior_mask])**2
        smooth_iters[interior_mask] = -1.0 - np.sqrt(mod_sq)

    return smooth_iters


class MandelbrotRenderer:
    """
    Numba CUDA GPU Mandelbrot & Julia Set Renderer supporting Float32 and Float64 precision modes
    with automatic CPU fallback execution.
    """
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.use_cuda = True

        try:
            if cuda.is_available() or cuda.detect():
                dev = cuda.get_current_device()
                self.gpu_device_name = dev.name.decode('utf-8') if isinstance(dev.name, bytes) else str(dev.name)
            else:
                self.use_cuda = False
                self.gpu_device_name = "CPU Vectorized Engine (Host Fallback)"
        except Exception:
            self.use_cuda = False
            self.gpu_device_name = "CPU Vectorized Engine (Host Fallback)"

        self.backend = "Numba (CUDA)" if self.use_cuda else "CPU Vectorized"
        print(f"[GPU Initialized] {self.gpu_device_name}")
        print(f"[MandelbrotRenderer] Active compute engine: {self.backend}")

    def render(self, min_x: float, max_x: float, min_y: float, max_y: float, max_iter: int,
               precision_mode: str = "float32", palette_mgr: Any = None, palette_idx: int = 0,
               phase_offset: float = 0.0, scale_factor: float = 1.0, out_rgb: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Renders Mandelbrot set into out_rgb NumPy array using CUDA or vectorized CPU fallback.
        """
        w = max(16, int(self.width * scale_factor))
        h = max(16, int(self.height * scale_factor))

        effective_max_iter = max(200, min(50000, int(max_iter)))

        if self.use_cuda:
            try:
                d_out = cuda.device_array((w, h), dtype=np.float32)
                threads_per_block = (16, 16)
                blocks_per_grid_x = math.ceil(w / 16)
                blocks_per_grid_y = math.ceil(h / 16)
                blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

                if precision_mode == "float64":
                    mandelbrot_cuda_f64_kernel[blocks_per_grid, threads_per_block](
                        w, h, np.float64(min_x), np.float64(max_x), np.float64(min_y), np.float64(max_y),
                        effective_max_iter, d_out
                    )
                else:
                    mandelbrot_cuda_f32_kernel[blocks_per_grid, threads_per_block](
                        w, h, np.float32(min_x), np.float32(max_x), np.float32(min_y), np.float32(max_y),
                        effective_max_iter, d_out
                    )

                cuda.synchronize()
                smooth_iters = d_out.copy_to_host()
            except Exception as cuda_err:
                smooth_iters = mandelbrot_cpu_vectorized(
                    w, h, min_x, max_x, min_y, max_y, effective_max_iter, precision_mode
                )
        else:
            smooth_iters = mandelbrot_cpu_vectorized(
                w, h, min_x, max_x, min_y, max_y, effective_max_iter, precision_mode
            )

        if out_rgb is None or out_rgb.shape != (self.width, self.height, 3):
            out_rgb = np.zeros((self.width, self.height, 3), dtype=np.uint8)

        if palette_mgr is not None:
            if scale_factor < 1.0:
                low_rgb = np.zeros((w, h, 3), dtype=np.uint8)
                palette_mgr.map_iterations_to_rgb(smooth_iters, effective_max_iter, palette_idx, phase_offset, low_rgb)
                rx = max(1, self.width // w)
                ry = max(1, self.height // h)
                up_rgb = np.repeat(np.repeat(low_rgb, rx, axis=0), ry, axis=1)
                actual_w = min(self.width, up_rgb.shape[0])
                actual_h = min(self.height, up_rgb.shape[1])
                out_rgb[:actual_w, :actual_h, :] = up_rgb[:actual_w, :actual_h, :]
            else:
                palette_mgr.map_iterations_to_rgb(smooth_iters, effective_max_iter, palette_idx, phase_offset, out_rgb)

        return out_rgb

    def render_julia(self, min_x: float, max_x: float, min_y: float, max_y: float, c_re: float, c_im: float,
                     max_iter: int, precision_mode: str = "float32", palette_mgr: Any = None, palette_idx: int = 0,
                     phase_offset: float = 0.0, scale_factor: float = 1.0, out_rgb: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Renders Julia Set z_{n+1} = z_n^2 + c into out_rgb NumPy array using CUDA or vectorized CPU fallback.
        """
        w = max(16, int(self.width * scale_factor))
        h = max(16, int(self.height * scale_factor))

        effective_max_iter = max(200, min(50000, int(max_iter)))

        if self.use_cuda:
            try:
                d_out = cuda.device_array((w, h), dtype=np.float32)
                threads_per_block = (16, 16)
                blocks_per_grid_x = math.ceil(w / 16)
                blocks_per_grid_y = math.ceil(h / 16)
                blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

                if precision_mode == "float64":
                    julia_cuda_f64_kernel[blocks_per_grid, threads_per_block](
                        w, h, np.float64(min_x), np.float64(max_x), np.float64(min_y), np.float64(max_y),
                        np.float64(c_re), np.float64(c_im), effective_max_iter, d_out
                    )
                else:
                    julia_cuda_f32_kernel[blocks_per_grid, threads_per_block](
                        w, h, np.float32(min_x), np.float32(max_x), np.float32(min_y), np.float32(max_y),
                        np.float32(c_re), np.float32(c_im), effective_max_iter, d_out
                    )

                cuda.synchronize()
                smooth_iters = d_out.copy_to_host()
            except Exception:
                smooth_iters = julia_cpu_vectorized(
                    w, h, min_x, max_x, min_y, max_y, c_re, c_im, effective_max_iter, precision_mode
                )
        else:
            smooth_iters = julia_cpu_vectorized(
                w, h, min_x, max_x, min_y, max_y, c_re, c_im, effective_max_iter, precision_mode
            )

        if out_rgb is None or out_rgb.shape != (self.width, self.height, 3):
            out_rgb = np.zeros((self.width, self.height, 3), dtype=np.uint8)

        if palette_mgr is not None:
            if scale_factor < 1.0:
                low_rgb = np.zeros((w, h, 3), dtype=np.uint8)
                palette_mgr.map_iterations_to_rgb(smooth_iters, effective_max_iter, palette_idx, phase_offset, low_rgb)
                rx = max(1, self.width // w)
                ry = max(1, self.height // h)
                up_rgb = np.repeat(np.repeat(low_rgb, rx, axis=0), ry, axis=1)
                actual_w = min(self.width, up_rgb.shape[0])
                actual_h = min(self.height, up_rgb.shape[1])
                out_rgb[:actual_w, :actual_h, :] = up_rgb[:actual_w, :actual_h, :]
            else:
                palette_mgr.map_iterations_to_rgb(smooth_iters, effective_max_iter, palette_idx, phase_offset, out_rgb)

        return out_rgb
