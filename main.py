import sys
import os
import math
import time
import datetime
import argparse
import numpy as np
import pygame

from mandelbrot_kernel import MandelbrotRenderer, compute_auto_max_iter
from utils import (
    ColorPaletteManager,
    FPSTracker,
    compute_performance_metrics,
    get_gpu_memory_info,
    export_video_async,
    run_cli_benchmark,
    generate_readme_assets
)
from gui import GUIPanel


class MandelbrotAppV3:
    """
    Production-Grade GPU Mandelbrot & Julia Set Inspector V3
    Featuring Numba CUDA acceleration, double precision fallback,
    integrated GUI control panel, split-screen Julia Set inspector,
    automated CLI benchmarking, and 4K README asset generation.
    """
    MINIBROT_X = np.float64(-0.743643887037158704752191506114774)
    MINIBROT_Y = np.float64(0.131825904205311970493132056385139)

    def __init__(self, width: int = 1280, height: int = 720):
        pygame.init()
        pygame.font.init()

        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption("GPU Mandelbrot Renderer V3 - Julia Inspector & Benchmark Suite")

        # Initial complex plane camera bounds using strict float64 precision
        self.DEFAULT_CENTER_X = np.float64(-0.75)
        self.DEFAULT_CENTER_Y = np.float64(0.0)
        self.DEFAULT_VIEW_WIDTH = np.float64(2.5)

        self.center_x = self.DEFAULT_CENTER_X
        self.center_y = self.DEFAULT_CENTER_Y
        self.view_width = self.DEFAULT_VIEW_WIDTH
        self.view_height = self.view_width * np.float64(self.height / self.width)

        # Julia Set Viewport & Inspector State
        self.julia_mode = False
        self.julia_c_re = np.float64(-0.7)
        self.julia_c_im = np.float64(0.27015)
        self.julia_center_x = np.float64(0.0)
        self.julia_center_y = np.float64(0.0)
        self.julia_view_width = np.float64(3.2)
        self.julia_view_height = np.float64(3.2)

        # Rendering & Controls State
        self.base_iter = 300
        self.max_iter = 300
        self.precision_mode = "float32"
        self.palette_idx = 0
        self.scale_factor = 1.0
        self.color_phase = 0.0
        self.show_hud = True
        self.show_hints = True

        # V3 Modes & Features State
        self.stress_mode = False
        self.auto_zoom = False
        self.is_recording = False
        self.rec_start_time = 0.0
        self.rec_duration = 10.0
        self.recorded_frames = []
        self.status_toast = ""
        self.toast_start_time = 0.0

        self.saved_base_iter = 300
        self.saved_precision_mode = "float32"

        # Mouse Drag State
        self.is_dragging = False
        self.drag_start_pos = (0, 0)

        # Core Components
        self.renderer = MandelbrotRenderer(width=self.width, height=self.height)
        self.palette_mgr = ColorPaletteManager()
        self.fps_tracker = FPSTracker(window_size=100)
        self.clock = pygame.time.Clock()

        # Fonts
        self.title_font = pygame.font.SysFont("Segoe UI", 16, bold=True)
        self.section_font = pygame.font.SysFont("Segoe UI", 13, bold=True)
        self.body_font = pygame.font.SysFont("Consolas", 13, bold=True)
        self.hint_font = pygame.font.SysFont("Segoe UI", 12, bold=False)

        # Offscreen surface array buffers
        self.render_buffer = np.zeros((self.width, self.height, 3), dtype=np.uint8)
        self.left_buffer = np.zeros((self.width // 2, self.height, 3), dtype=np.uint8)
        self.right_buffer = np.zeros((self.width // 2, self.height, 3), dtype=np.uint8)

        # Initialize Integrated GUI Control Panel
        self._init_gui_panel()

    def _init_gui_panel(self) -> None:
        """Constructs collapsible GUI control panel widgets."""
        self.gui_panel = GUIPanel(x=15, y=325, width=460, title="Real-Time Control Panel")

        self.gui_slider_iter = self.gui_panel.add_slider(
            label="Max Iterations", min_val=200, max_val=50000, initial_val=self.max_iter,
            callback=self._gui_on_iter_change, is_int=True
        )

        self.gui_slider_phase = self.gui_panel.add_slider(
            label="Color Phase Shift", min_val=0.0, max_val=1.0, initial_val=self.color_phase,
            callback=self._gui_on_phase_change, is_int=False
        )

        self.gui_slider_scale = self.gui_panel.add_slider(
            label="Render Target Scale", min_val=0.25, max_val=1.0, initial_val=self.scale_factor,
            callback=self._gui_on_scale_change, is_int=False
        )

        self.gui_dropdown_palette = self.gui_panel.add_dropdown(
            label="Color Palette", options=self.palette_mgr.palettes, selected_idx=self.palette_idx,
            callback=self._gui_on_palette_select
        )

        self.gui_dropdown_precision = self.gui_panel.add_dropdown(
            label="Precision Mode", options=["FP32 (Single)", "FP64 (Double)"],
            selected_idx=0 if self.precision_mode == "float32" else 1,
            callback=self._gui_on_precision_select
        )

        self.gui_panel.add_button("Toggle Julia Inspector [J]", self.toggle_julia_mode, accent_color=(168, 85, 247))
        self.gui_panel.add_button("Toggle Stress Benchmark [B]", self.toggle_stress_mode, accent_color=(245, 158, 11))
        self.gui_panel.add_button("Record MP4 Video [R]", self.toggle_recording, accent_color=(239, 68, 68))
        self.gui_panel.add_button("Reset Camera View [Home]", self.reset_camera, accent_color=(56, 189, 248))

    def _gui_on_iter_change(self, val: float) -> None:
        self.base_iter = int(val)
        self.max_iter = int(val)

    def _gui_on_phase_change(self, val: float) -> None:
        self.color_phase = float(val)

    def _gui_on_scale_change(self, val: float) -> None:
        self.scale_factor = float(val)

    def _gui_on_palette_select(self, idx: int) -> None:
        self.palette_idx = idx
        p_name = self.palette_mgr.get_palette_name(idx)
        self.set_toast(f"GUI Palette Selected: {p_name}")

    def _gui_on_precision_select(self, idx: int) -> None:
        self.precision_mode = "float32" if idx == 0 else "float64"
        self.set_toast(f"GUI Precision Mode: {self.precision_mode.upper()}")

    @property
    def min_x(self) -> np.float64:
        return self.center_x - self.view_width / np.float64(2.0)

    @property
    def max_x(self) -> np.float64:
        return self.center_x + self.view_width / np.float64(2.0)

    @property
    def min_y(self) -> np.float64:
        return self.center_y - self.view_height / np.float64(2.0)

    @property
    def max_y(self) -> np.float64:
        return self.center_y + self.view_height / np.float64(2.0)

    def reset_camera(self) -> None:
        self.center_x = self.DEFAULT_CENTER_X
        self.center_y = self.DEFAULT_CENTER_Y
        self.view_width = self.DEFAULT_VIEW_WIDTH
        self.view_height = self.view_width * np.float64(self.height / self.width)
        self.auto_zoom = False
        self.set_toast("Camera View Reset")

    def toggle_julia_mode(self) -> None:
        """Toggles real-time Julia Set split-screen inspector (Press [J])."""
        self.julia_mode = not self.julia_mode
        state = "ACTIVATED (Split-Screen View)" if self.julia_mode else "DEACTIVATED"
        self.set_toast(f"JULIA SET INSPECTOR {state}")

    def toggle_stress_mode(self) -> None:
        self.stress_mode = not self.stress_mode
        if self.stress_mode:
            self.saved_base_iter = self.base_iter
            self.saved_precision_mode = self.precision_mode
            self.base_iter = 10000
            self.max_iter = 10000
            self.precision_mode = "float64"
            self.scale_factor = 1.0
            self.set_toast("STRESS MODE ACTIVATED (10,000 Iterations | Float64 | Native Scale)")
        else:
            self.base_iter = self.saved_base_iter
            self.precision_mode = self.saved_precision_mode
            self.set_toast("STRESS MODE DEACTIVATED")

    def toggle_recording(self) -> None:
        if not self.is_recording:
            self.is_recording = True
            self.rec_start_time = time.time()
            self.recorded_frames = []
            self.set_toast("RECORDING STARTED (10s MP4 Animation Export)")
        else:
            self.stop_and_export_recording()

    def stop_and_export_recording(self) -> None:
        if not self.is_recording:
            return
        self.is_recording = False
        if not self.recorded_frames:
            self.set_toast("Recording cancelled: no frames captured")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mandelbrot_zoom_{timestamp}.mp4"
        self.set_toast(f"EXPORTING VIDEO ({len(self.recorded_frames)} frames)...")

        def callback(success, path_or_msg):
            if success:
                self.set_toast(f"VIDEO SAVED: {path_or_msg}")
            else:
                self.set_toast(f"Export Failed: {path_or_msg}")

        export_video_async(self.recorded_frames, filename, fps=30, status_callback=callback)

    def set_toast(self, message: str) -> None:
        self.status_toast = message
        self.toast_start_time = time.time()

    def compute_scaled_max_iter(self) -> int:
        if self.stress_mode:
            return 10000
        current_zoom_level = float(np.float64(3.5) / max(np.float64(1e-18), self.view_width))
        log_zoom = math.log10(max(1.0, current_zoom_level))
        scaled = int(self.base_iter + 150 * log_zoom)
        return max(200, min(50000, scaled))

    def zoom_at_point(self, mouse_x: int, mouse_y: int, zoom_in: bool = True) -> None:
        if self.julia_mode and mouse_x >= self.width // 2:
            # Zoom Julia set right viewport
            w_half = self.width // 2
            mx_ratio = np.float64(mouse_x - w_half) / np.float64(w_half) - np.float64(0.5)
            my_ratio = np.float64(mouse_y) / np.float64(self.height) - np.float64(0.5)
            cursor_x = self.julia_center_x + mx_ratio * self.julia_view_width
            cursor_y = self.julia_center_y + my_ratio * self.julia_view_height

            zoom_factor = np.float64(0.82) if zoom_in else np.float64(1.22)
            self.julia_view_width *= zoom_factor
            self.julia_view_height *= zoom_factor
            self.julia_center_x = cursor_x - mx_ratio * self.julia_view_width
            self.julia_center_y = cursor_y - my_ratio * self.julia_view_height
        else:
            # Zoom Mandelbrot view
            w_eff = (self.width // 2) if self.julia_mode else self.width
            mx_ratio = np.float64(mouse_x) / np.float64(w_eff) - np.float64(0.5)
            my_ratio = np.float64(mouse_y) / np.float64(self.height) - np.float64(0.5)

            cursor_x = self.center_x + mx_ratio * self.view_width
            cursor_y = self.center_y + my_ratio * self.view_height

            zoom_factor = np.float64(0.82) if zoom_in else np.float64(1.22)
            self.view_width *= zoom_factor
            self.view_height *= zoom_factor

            self.center_x = cursor_x - mx_ratio * self.view_width
            self.center_y = cursor_y - my_ratio * self.view_height

    def update_auto_zoom(self, dt: float) -> None:
        if not self.auto_zoom:
            return

        pan_speed = np.float64(0.06)
        zoom_speed = np.float64(0.983)

        self.center_x += (self.MINIBROT_X - self.center_x) * pan_speed
        self.center_y += (self.MINIBROT_Y - self.center_y) * pan_speed

        self.view_width *= zoom_speed
        self.view_height = self.view_width * np.float64(self.height / self.width)

        if self.view_width < np.float64(1e-15):
            self.view_width = np.float64(1.0e-3)
            self.view_height = self.view_width * np.float64(self.height / self.width)

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            # Pass events to GUI Panel first
            if self.gui_panel.handle_event(event):
                continue

            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.VIDEORESIZE:
                self.width, self.height = event.w, event.h
                self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
                self.renderer.width = self.width
                self.renderer.height = self.height
                self.view_height = self.view_width * np.float64(self.height / self.width)
                self.render_buffer = np.zeros((self.width, self.height, 3), dtype=np.uint8)
                self.left_buffer = np.zeros((self.width // 2, self.height, 3), dtype=np.uint8)
                self.right_buffer = np.zeros((self.width // 2, self.height, 3), dtype=np.uint8)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.is_dragging = True
                    self.drag_start_pos = event.pos

                    # Update Julia constant c when clicking on Mandelbrot view
                    if self.julia_mode:
                        w_half = self.width // 2
                        if event.pos[0] < w_half:
                            mx_ratio = np.float64(event.pos[0]) / np.float64(w_half) - np.float64(0.5)
                            my_ratio = np.float64(event.pos[1]) / np.float64(self.height) - np.float64(0.5)
                            self.julia_c_re = self.center_x + mx_ratio * self.view_width
                            self.julia_c_im = self.center_y + my_ratio * self.view_height
                            self.set_toast(f"Julia c = ({self.julia_c_re:.4f}, {self.julia_c_im:.4f}i)")

                elif event.button in (4, 5) or getattr(event, 'button', None) in (4, 5):
                    zoom_in = (event.button == 4)
                    self.zoom_at_point(event.pos[0], event.pos[1], zoom_in=zoom_in)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.is_dragging = False

            elif event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                self.zoom_at_point(mx, my, zoom_in=(event.y > 0))

            elif event.type == pygame.MOUSEMOTION:
                if self.is_dragging:
                    dx = np.float64(event.pos[0] - self.drag_start_pos[0])
                    dy = np.float64(event.pos[1] - self.drag_start_pos[1])
                    self.drag_start_pos = event.pos

                    w_eff = (self.width // 2) if self.julia_mode else self.width
                    self.center_x -= (dx / np.float64(w_eff)) * self.view_width
                    self.center_y -= (dy / np.float64(self.height)) * self.view_height

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_j:
                    self.toggle_julia_mode()
                elif event.key == pygame.K_c:
                    self.palette_idx = (self.palette_idx + 1) % len(self.palette_mgr.palettes)
                    p_name = self.palette_mgr.get_palette_name(self.palette_idx)
                    self.gui_dropdown_palette.selected_idx = self.palette_idx
                    self.set_toast(f"Palette Switched: {p_name}")
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
                    self.palette_idx = (event.key - pygame.K_1) % len(self.palette_mgr.palettes)
                    self.gui_dropdown_palette.selected_idx = self.palette_idx
                    p_name = self.palette_mgr.get_palette_name(self.palette_idx)
                    self.set_toast(f"Palette Selected: {p_name}")
                elif event.key == pygame.K_b:
                    self.toggle_stress_mode()
                elif event.key == pygame.K_p:
                    self.auto_zoom = not self.auto_zoom
                    status = "ENABLED" if self.auto_zoom else "DISABLED"
                    self.set_toast(f"Auto-Zoom Trajectory {status}")
                elif event.key == pygame.K_r:
                    self.toggle_recording()
                elif event.key in (pygame.K_HOME, pygame.K_BACKSPACE):
                    self.reset_camera()
                elif event.key == pygame.K_SPACE:
                    if not self.stress_mode:
                        self.precision_mode = "float64" if self.precision_mode == "float32" else "float32"
                        self.gui_dropdown_precision.selected_idx = 0 if self.precision_mode == "float32" else 1
                        self.set_toast(f"Precision Mode: {self.precision_mode.upper()}")
                elif event.key == pygame.K_h:
                    self.show_hints = not self.show_hints
                    self.set_toast(f"Control Hints: {'SHOWN' if self.show_hints else 'HIDDEN'}")
                elif event.key == pygame.K_UP:
                    if not self.stress_mode:
                        self.base_iter = min(50000, self.base_iter + 100)
                        self.gui_slider_iter.set_value(self.base_iter)
                elif event.key == pygame.K_DOWN:
                    if not self.stress_mode:
                        self.base_iter = max(200, self.base_iter - 100)
                        self.gui_slider_iter.set_value(self.base_iter)

        return True

    def render_hud(self, fps: float, frame_ms: float, avg_ms: float, min_ms: float, max_ms: float) -> None:
        """Renders sleeker dark UI panel HUD."""
        hud_w = 460
        hud_h = 300
        hud_surface = pygame.Surface((hud_w, hud_h), pygame.SRCALPHA)
        hud_surface.fill((12, 16, 26, 220))

        border_color = (168, 85, 247) if self.julia_mode else ((239, 68, 68) if self.stress_mode else (56, 189, 248))
        pygame.draw.rect(hud_surface, border_color, (0, 0, hud_w, hud_h), width=1, border_radius=10)

        mpx_sec, gflops = compute_performance_metrics(self.width, self.height, self.max_iter, frame_ms)
        zoom_level = float(np.float64(3.5) / max(np.float64(1e-18), self.view_width))
        palette_name = self.palette_mgr.get_palette_name(self.palette_idx)
        gpu_info = get_gpu_memory_info(self.renderer.backend)

        badges = []
        if self.is_recording:
            rec_elapsed = time.time() - self.rec_start_time
            badges.append((f"REC {rec_elapsed:04.1f}s / {self.rec_duration:02.0f}s", (239, 68, 68)))
        if self.julia_mode:
            badges.append(("JULIA INSPECTOR", (168, 85, 247)))
        if self.stress_mode:
            badges.append(("STRESS MODE (10K ITER)", (245, 158, 11)))
        if self.auto_zoom:
            badges.append(("AUTO-ZOOM ACTIVE", (56, 189, 248)))
        badges.append((self.precision_mode.upper(), (34, 197, 94)))

        title_txt = self.title_font.render("GPU MANDELBROT & JULIA INSPECTOR V3", True, (56, 189, 248))
        hud_surface.blit(title_txt, (14, 10))

        badge_x = 14
        badge_y = 35
        for b_text, b_color in badges:
            b_surf = self.hint_font.render(f" {b_text} ", True, (255, 255, 255))
            b_w = b_surf.get_width() + 8
            b_bg = pygame.Surface((b_w, 18), pygame.SRCALPHA)
            b_bg.fill((*b_color, 200))
            pygame.draw.rect(b_bg, b_color, (0, 0, b_w, 18), width=1, border_radius=4)
            b_bg.blit(b_surf, (4, 1))
            hud_surface.blit(b_bg, (badge_x, badge_y))
            badge_x += b_w + 8

        lines = [
            (f"Frame Rate & Latency : {fps:5.1f} FPS  |  {frame_ms:5.2f} ms", (240, 240, 245), self.body_font),
            (f"100-Frame Window Lat: Avg {avg_ms:.1f}ms | Min {min_ms:.1f}ms | Max {max_ms:.1f}ms", (148, 163, 184), self.hint_font),
            (f"Compute Throughput  : {mpx_sec:5.1f} Mpx/s  |  {gflops:6.2f} GFLOPS", (250, 204, 21), self.body_font),
            (f"Zoom Magnification  : {zoom_level:.2e}x", (168, 85, 247), self.body_font),
            (f"Max Iterations      : {self.max_iter:d} (Base: {self.base_iter:d})", (220, 220, 230), self.body_font),
            (f"Color Palette       : [{self.palette_idx + 1}] {palette_name}  [C]", (236, 72, 153), self.body_font),
            (f"Animated Flow Phase : {self.color_phase:.3f} (Continuous)", (94, 234, 212), self.hint_font),
            (f"CUDA VRAM Status    : {gpu_info}", (148, 163, 184), self.hint_font),
        ]

        y_off = 62
        for text, color, font in lines:
            txt_surf = font.render(text, True, color)
            hud_surface.blit(txt_surf, (14, y_off))
            y_off += 22

        self.screen.blit(hud_surface, (15, 15))

        # Toast notification banner
        if self.status_toast and (time.time() - self.toast_start_time < 3.5):
            toast_surf = self.section_font.render(f"  {self.status_toast}  ", True, (255, 255, 255))
            tw = toast_surf.get_width() + 16
            th = toast_surf.get_height() + 10
            tbox = pygame.Surface((tw, th), pygame.SRCALPHA)
            tbox.fill((15, 23, 42, 230))
            pygame.draw.rect(tbox, (56, 189, 248), (0, 0, tw, th), width=1, border_radius=6)
            tbox.blit(toast_surf, (8, 5))
            self.screen.blit(tbox, ((self.width - tw) // 2, 20))

        # Controls hint bar
        if self.show_hints:
            hint_w = 680
            hint_h = 28
            hint_surf = pygame.Surface((hint_w, hint_h), pygame.SRCALPHA)
            hint_surf.fill((12, 16, 26, 200))
            pygame.draw.rect(hint_surf, (71, 85, 105), (0, 0, hint_w, hint_h), width=1, border_radius=5)
            hint_str = "[J] Julia Set | [C] Palette | [B] Stress | [P] Auto-Zoom | [R] Record | [SPACE] Precision | [H] Hints"
            hint_txt = self.hint_font.render(hint_str, True, (226, 232, 240))
            hint_surf.blit(hint_txt, (12, 5))
            self.screen.blit(hint_surf, (self.width - hint_w - 15, self.height - 40))

    def run(self) -> None:
        """Main application execution loop."""
        running = True
        last_frame_time = time.perf_counter()

        while running:
            current_time = time.perf_counter()
            dt = current_time - last_frame_time
            last_frame_time = current_time

            running = self.handle_events()

            # Advance animated color phase flow
            self.color_phase = (self.color_phase + dt * 0.04) % 1.0

            # Update camera auto-zoom trajectory
            self.update_auto_zoom(dt)

            # Auto scale max_iter
            self.max_iter = self.compute_scaled_max_iter()

            # Dynamic resolution scale
            self.scale_factor = 1.0 if self.stress_mode else (0.5 if self.is_dragging else 1.0)

            if self.julia_mode:
                # Split-screen mode: Left = Mandelbrot, Right = Julia Set
                w_half = self.width // 2
                h_full = self.height

                # Render Left Mandelbrot View
                self.renderer.width = w_half
                self.renderer.height = h_full
                left_rgb = self.renderer.render(
                    min_x=self.min_x, max_x=self.max_x, min_y=self.min_y, max_y=self.max_y,
                    max_iter=self.max_iter, precision_mode=self.precision_mode,
                    palette_mgr=self.palette_mgr, palette_idx=self.palette_idx,
                    phase_offset=self.color_phase, scale_factor=self.scale_factor,
                    out_rgb=self.left_buffer
                )

                # Render Right Julia Set View
                j_min_x = self.julia_center_x - self.julia_view_width / 2.0
                j_max_x = self.julia_center_x + self.julia_view_width / 2.0
                j_min_y = self.julia_center_y - self.julia_view_height / 2.0
                j_max_y = self.julia_center_y + self.julia_view_height / 2.0

                right_rgb = self.renderer.render_julia(
                    min_x=j_min_x, max_x=j_max_x, min_y=j_min_y, max_y=j_max_y,
                    c_re=self.julia_c_re, c_im=self.julia_c_im,
                    max_iter=self.max_iter, precision_mode=self.precision_mode,
                    palette_mgr=self.palette_mgr, palette_idx=self.palette_idx,
                    phase_offset=self.color_phase, scale_factor=self.scale_factor,
                    out_rgb=self.right_buffer
                )

                # Blit left and right viewports onto main screen
                self.screen.blit(pygame.surfarray.make_surface(left_rgb), (0, 0))
                self.screen.blit(pygame.surfarray.make_surface(right_rgb), (w_half, 0))

                # Draw Split Divider Line
                pygame.draw.line(self.screen, (168, 85, 247), (w_half, 0), (w_half, h_full), width=2)

                # Draw Selected Julia Parameter c Crosshair on Mandelbrot View
                c_px = int(((self.julia_c_re - self.min_x) / self.view_width) * w_half)
                c_py = int(((self.julia_c_im - self.min_y) / self.view_height) * h_full)
                if 0 <= c_px < w_half and 0 <= c_py < h_full:
                    pygame.draw.circle(self.screen, (255, 255, 255), (c_px, c_py), 6, width=1)
                    pygame.draw.circle(self.screen, (236, 72, 153), (c_px, c_py), 3)

                # Restore full renderer dimensions
                self.renderer.width = self.width
                self.renderer.height = self.height

                # Record frame if active
                if self.is_recording:
                    frame_surf = pygame.display.get_surface()
                    arr = pygame.surfarray.array3d(frame_surf)
                    frame_copy = np.transpose(arr, (1, 0, 2)).copy()
                    self.recorded_frames.append(frame_copy)
                    if time.time() - self.rec_start_time >= self.rec_duration:
                        self.stop_and_export_recording()

            else:
                # Fullscreen Mandelbrot mode
                out_rgb = self.renderer.render(
                    min_x=self.min_x, max_x=self.max_x, min_y=self.min_y, max_y=self.max_y,
                    max_iter=self.max_iter, precision_mode=self.precision_mode,
                    palette_mgr=self.palette_mgr, palette_idx=self.palette_idx,
                    phase_offset=self.color_phase, scale_factor=self.scale_factor,
                    out_rgb=self.render_buffer
                )
                pygame.surfarray.blit_array(self.screen, out_rgb)

                if self.is_recording:
                    frame_copy = np.transpose(out_rgb, (1, 0, 2)).copy()
                    self.recorded_frames.append(frame_copy)
                    if time.time() - self.rec_start_time >= self.rec_duration:
                        self.stop_and_export_recording()

            # Update FPS statistics
            fps, frame_ms, avg_ms, min_ms, max_ms = self.fps_tracker.update()

            # Render HUD Overlay & GUI Panel
            if self.show_hud:
                self.render_hud(fps, frame_ms, avg_ms, min_ms, max_ms)
                self.gui_panel.draw(self.screen, self.hint_font)

            pygame.display.flip()
            self.clock.tick(144)

        pygame.quit()
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Mandelbrot Renderer V3 Suite")
    parser.add_argument("--benchmark", "-b", action="store_true", help="Run 30-second automated compute benchmark suite")
    parser.add_argument("--generate-readme-assets", action="store_true", help="Batch render 4K screenshots in docs/")
    args = parser.parse_args()

    if args.benchmark:
        run_cli_benchmark()
    elif args.generate_readme_assets:
        generate_readme_assets()
    else:
        app = MandelbrotAppV3(width=1280, height=720)
        app.run()
