import pygame
from typing import List, Tuple, Callable, Optional, Dict, Any

class UIWidget:
    """Base class for interactive PyGame GUI widgets."""
    def __init__(self, x: int, y: int, width: int, height: int):
        self.rect = pygame.Rect(x, y, width, height)
        self.is_hovered = False
        self.is_active = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        return False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        pass


class Button(UIWidget):
    """Interactive push button widget."""
    def __init__(self, x: int, y: int, width: int, height: int, text: str, callback: Callable[[], None],
                 accent_color: Tuple[int, int, int] = (56, 189, 248)):
        super().__init__(x, y, width, height)
        self.text = text
        self.callback = callback
        self.accent_color = accent_color

    def handle_event(self, event: pygame.event.Event) -> bool:
        mouse_pos = pygame.mouse.get_pos()
        self.is_hovered = self.rect.collidepoint(mouse_pos)

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.is_hovered:
            self.is_active = True
            return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.is_active:
            if self.is_hovered:
                self.callback()
            self.is_active = False
            return True
        return False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        bg_color = (30, 41, 59, 230) if not self.is_hovered else (51, 65, 85, 240)
        if self.is_active:
            bg_color = (15, 23, 42, 250)

        btn_surf = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        btn_surf.fill(bg_color)
        border_col = self.accent_color if self.is_hovered else (71, 85, 105)
        pygame.draw.rect(btn_surf, border_col, (0, 0, self.rect.width, self.rect.height), width=1, border_radius=6)

        txt_surf = font.render(self.text, True, (241, 245, 249))
        txt_rect = txt_surf.get_rect(center=(self.rect.width // 2, self.rect.height // 2))
        btn_surf.blit(txt_surf, txt_rect)
        surface.blit(btn_surf, self.rect)


class Slider(UIWidget):
    """Interactive continuous numeric slider widget."""
    def __init__(self, x: int, y: int, width: int, height: int, label: str, min_val: float, max_val: float,
                 initial_val: float, callback: Callable[[float], None], is_int: bool = False):
        super().__init__(x, y, width, height)
        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.callback = callback
        self.is_int = is_int
        self.is_dragging = False

        self.handle_width = 14
        self.track_rect = pygame.Rect(x + 10, y + 22, width - 20, 6)

    def set_value(self, val: float) -> None:
        self.value = max(self.min_val, min(self.max_val, val))

    def handle_event(self, event: pygame.event.Event) -> bool:
        mouse_pos = pygame.mouse.get_pos()
        self.is_hovered = self.rect.collidepoint(mouse_pos)

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.is_hovered:
            self.is_dragging = True
            self._update_val_from_mouse(mouse_pos[0])
            return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.is_dragging:
                self.is_dragging = False
                return True
        elif event.type == pygame.MOUSEMOTION and self.is_dragging:
            self._update_val_from_mouse(mouse_pos[0])
            return True
        return False

    def _update_val_from_mouse(self, mouse_x: int) -> None:
        norm = (mouse_x - self.track_rect.left) / float(self.track_rect.width)
        norm = max(0.0, min(1.0, norm))
        val = self.min_val + norm * (self.max_val - self.min_val)
        if self.is_int:
            val = round(val)
        self.value = val
        self.callback(self.value)

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        # Label & Value Text
        val_str = f"{int(self.value):d}" if self.is_int else f"{self.value:.2f}"
        lbl_surf = font.render(f"{self.label}: {val_str}", True, (226, 232, 240))
        surface.blit(lbl_surf, (self.rect.x + 10, self.rect.y + 2))

        # Track Line
        pygame.draw.rect(surface, (51, 65, 85), self.track_rect, border_radius=3)

        # Filled portion
        norm = (self.value - self.min_val) / float(self.max_val - self.min_val)
        fill_width = int(norm * self.track_rect.width)
        if fill_width > 0:
            fill_rect = pygame.Rect(self.track_rect.x, self.track_rect.y, fill_width, self.track_rect.height)
            pygame.draw.rect(surface, (56, 189, 248), fill_rect, border_radius=3)

        # Handle Knob
        handle_x = self.track_rect.left + int(norm * self.track_rect.width) - self.handle_width // 2
        handle_rect = pygame.Rect(handle_x, self.track_rect.centery - 7, self.handle_width, 14)
        handle_col = (248, 250, 252) if (self.is_hovered or self.is_dragging) else (203, 213, 225)
        pygame.draw.rect(surface, handle_col, handle_rect, border_radius=4)
        pygame.draw.rect(surface, (56, 189, 248), handle_rect, width=1, border_radius=4)


class Dropdown(UIWidget):
    """Interactive expandable select dropdown widget."""
    def __init__(self, x: int, y: int, width: int, height: int, label: str, options: List[str],
                 selected_idx: int, callback: Callable[[int], None]):
        super().__init__(x, y, width, height)
        self.label = label
        self.options = options
        self.selected_idx = selected_idx
        self.callback = callback
        self.is_open = False
        self.option_height = 24

    def handle_event(self, event: pygame.event.Event) -> bool:
        mouse_pos = pygame.mouse.get_pos()
        head_rect = pygame.Rect(self.rect.x, self.rect.y, self.rect.width, self.rect.height)

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if head_rect.collidepoint(mouse_pos):
                self.is_open = not self.is_open
                return True
            elif self.is_open:
                # Check option click
                for i in range(len(self.options)):
                    opt_rect = pygame.Rect(self.rect.x, self.rect.y + self.rect.height + i * self.option_height,
                                           self.rect.width, self.option_height)
                    if opt_rect.collidepoint(mouse_pos):
                        self.selected_idx = i
                        self.callback(i)
                        self.is_open = False
                        return True
                self.is_open = False
        return False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        head_rect = pygame.Rect(self.rect.x, self.rect.y, self.rect.width, self.rect.height)
        head_surf = pygame.Surface((head_rect.width, head_rect.height), pygame.SRCALPHA)
        head_surf.fill((30, 41, 59, 230))
        pygame.draw.rect(head_surf, (71, 85, 105), (0, 0, head_rect.width, head_rect.height), width=1, border_radius=6)

        txt_str = f"{self.label}: {self.options[self.selected_idx % len(self.options)]}"
        txt_surf = font.render(txt_str, True, (241, 245, 249))
        head_surf.blit(txt_surf, (10, 5))

        arrow_str = "▲" if self.is_open else "▼"
        arrow_surf = font.render(arrow_str, True, (56, 189, 248))
        head_surf.blit(arrow_surf, (head_rect.width - 20, 5))
        surface.blit(head_surf, head_rect)

        # Draw Open Dropdown Options Overlay
        if self.is_open:
            total_opt_h = len(self.options) * self.option_height
            opt_box = pygame.Surface((self.rect.width, total_opt_h), pygame.SRCALPHA)
            opt_box.fill((15, 23, 42, 245))
            pygame.draw.rect(opt_box, (56, 189, 248), (0, 0, self.rect.width, total_opt_h), width=1, border_radius=6)

            mouse_pos = pygame.mouse.get_pos()
            for i, opt in enumerate(self.options):
                opt_rect = pygame.Rect(self.rect.x, self.rect.y + self.rect.height + i * self.option_height,
                                       self.rect.width, self.option_height)
                is_opt_hover = opt_rect.collidepoint(mouse_pos)
                if is_opt_hover:
                    pygame.draw.rect(opt_box, (51, 65, 85), (2, i * self.option_height + 2, self.rect.width - 4, self.option_height - 4), border_radius=4)

                opt_col = (56, 189, 248) if i == self.selected_idx else (226, 232, 240)
                opt_txt = font.render(opt, True, opt_col)
                opt_box.blit(opt_txt, (10, i * self.option_height + 3))

            surface.blit(opt_box, (self.rect.x, self.rect.y + self.rect.height + 2))



class GUIPanel:
    """
    Floating, collapsible real-time control panel containing sliders, dropdowns, and buttons.
    """
    def __init__(self, x: int = 15, y: int = 330, width: int = 460, title: str = "Real-Time Control Panel"):
        self.x = x
        self.y = y
        self.width = width
        self.title = title
        self.is_collapsed = False
        self.header_height = 32
        self.is_dragging_header = False
        self.drag_offset = (0, 0)

        self.widgets: List[UIWidget] = []
        self.dropdowns: List[Dropdown] = []

    def add_button(self, text: str, callback: Callable[[], None], accent_color=(56, 189, 248)) -> Button:
        btn = Button(self.x + 15, self.y + self._get_next_y(), self.width - 30, 28, text, callback, accent_color)
        self.widgets.append(btn)
        return btn

    def add_slider(self, label: str, min_val: float, max_val: float, initial_val: float,
                   callback: Callable[[float], None], is_int: bool = False) -> Slider:
        slider = Slider(self.x + 15, self.y + self._get_next_y(), self.width - 30, 36, label, min_val, max_val, initial_val, callback, is_int)
        self.widgets.append(slider)
        return slider

    def add_dropdown(self, label: str, options: List[str], selected_idx: int,
                     callback: Callable[[int], None]) -> Dropdown:
        dd = Dropdown(self.x + 15, self.y + self._get_next_y(), self.width - 30, 26, label, options, selected_idx, callback)
        self.widgets.append(dd)
        self.dropdowns.append(dd)
        return dd

    def _get_next_y(self) -> int:
        offset = self.header_height + 10
        for w in self.widgets:
            offset += w.rect.height + 8
        return offset

    def get_total_height(self) -> int:
        if self.is_collapsed:
            return self.header_height
        return self._get_next_y() + 6

    def handle_event(self, event: pygame.event.Event) -> bool:
        mouse_pos = pygame.mouse.get_pos()
        header_rect = pygame.Rect(self.x, self.y, self.width, self.header_height)

        # Header Collapse & Dragging
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if header_rect.collidepoint(mouse_pos):
                toggle_btn_rect = pygame.Rect(self.x + self.width - 30, self.y + 4, 24, 24)
                if toggle_btn_rect.collidepoint(mouse_pos):
                    self.is_collapsed = not self.is_collapsed
                else:
                    self.is_dragging_header = True
                    self.drag_offset = (mouse_pos[0] - self.x, mouse_pos[1] - self.y)
                return True

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.is_dragging_header:
                self.is_dragging_header = False
                return True

        elif event.type == pygame.MOUSEMOTION and self.is_dragging_header:
            self.x = mouse_pos[0] - self.drag_offset[0]
            self.y = mouse_pos[1] - self.drag_offset[1]
            self._reposition_widgets()
            return True

        # Pass event to open dropdowns first to capture selection
        if not self.is_collapsed:
            for dd in self.dropdowns:
                if dd.is_open:
                    if dd.handle_event(event):
                        return True

            for w in self.widgets:
                if w.handle_event(event):
                    return True

        return False

    def _reposition_widgets(self) -> None:
        curr_y = self.header_height + 10
        for w in self.widgets:
            w.rect.x = self.x + 15
            w.rect.y = self.y + curr_y
            if isinstance(w, Slider):
                w.track_rect.x = w.rect.x + 10
                w.track_rect.y = w.rect.y + 22
            curr_y += w.rect.height + 8

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        total_h = self.get_total_height()
        panel_surf = pygame.Surface((self.width, total_h), pygame.SRCALPHA)
        panel_surf.fill((12, 16, 26, 225))
        pygame.draw.rect(panel_surf, (56, 189, 248), (0, 0, self.width, total_h), width=1, border_radius=8)

        # Header Bar
        pygame.draw.rect(panel_surf, (30, 41, 59, 240), (1, 1, self.width - 2, self.header_height), border_top_left_radius=7, border_top_right_radius=7)
        title_txt = font.render(self.title, True, (56, 189, 248))
        panel_surf.blit(title_txt, (12, 6))

        # Collapse Button Symbol
        symbol = "[ + ]" if self.is_collapsed else "[ − ]"
        sym_txt = font.render(symbol, True, (226, 232, 240))
        panel_surf.blit(sym_txt, (self.width - 45, 6))

        surface.blit(panel_surf, (self.x, self.y))

        # Draw inner widgets if expanded
        if not self.is_collapsed:
            for w in self.widgets:
                w.draw(surface, font)

            # Draw open dropdown menus on top
            for dd in self.dropdowns:
                if dd.is_open:
                    dd.draw(surface, font)
