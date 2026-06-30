"""Pure-torch 5x7 bitmap text drawing for caption overlays (uppercase font, letters included).

Mirrors the in-place ``draw_text`` contract of ``cuvis_ai.utils.torch_draw`` but ships a font that
covers A-Z plus digits and a little punctuation. The catalog font is digit-only (sufficient for its
object-id overlays, blank for word captions), so the caption nodes draw through this local font
instead. Text is upper-cased before lookup; unknown characters render as a space.
"""

from __future__ import annotations

import torch
from torch import Tensor

# 5-wide, 7-tall glyphs. Digits / `-` / `.` / space match cuvis_ai.utils.torch_draw; A-Z are added
# here. Each entry is 7 strings of 5 "0"/"1" bits.
_FONT_5x7: dict[str, list[str]] = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "00110", "00110"],
    "%": ["11000", "11001", "00010", "00100", "01000", "10011", "00011"],
    "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["01110", "10001", "00001", "00110", "00001", "10001", "01110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["01110", "10000", "11110", "10001", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "11100"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11100", "10010", "10001", "10001", "10001", "10010", "11100"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00111", "00010", "00010", "00010", "00010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
}


def _as_color(color: Tensor | tuple[int, int, int], device: torch.device) -> Tensor:
    """Normalize an RGB color to a uint8 ``(3,)`` tensor on the target device."""
    c = torch.as_tensor(color, dtype=torch.uint8, device=device)
    if c.shape != (3,):
        raise ValueError(f"Expected color shape (3,), got {tuple(c.shape)}")
    return c


def _glyph(text: str, device: torch.device) -> Tensor:
    """Render ``text`` into a ``[7, W]`` uint8 bitmap (1-column space between glyphs)."""
    if not text:
        return torch.zeros((7, 0), dtype=torch.uint8, device=device)
    parts: list[Tensor] = []
    for idx, char in enumerate(text):
        rows = _FONT_5x7.get(char, _FONT_5x7[" "])
        glyph = torch.tensor(
            [[1 if bit == "1" else 0 for bit in row] for row in rows],
            dtype=torch.uint8,
            device=device,
        )
        parts.append(glyph)
        if idx < len(text) - 1:
            parts.append(torch.zeros((7, 1), dtype=torch.uint8, device=device))
    return torch.cat(parts, dim=1)


@torch.no_grad()
def draw_text(
    img: Tensor,
    x: int,
    y: int,
    text: str,
    color: Tensor | tuple[int, int, int],
    scale: int = 2,
    bg: bool = True,
) -> None:
    """Draw upper-cased bitmap text in-place on a uint8 HWC image."""
    if img.ndim != 3 or img.shape[-1] != 3 or img.dtype != torch.uint8:
        raise ValueError(
            f"Expected image shape (H, W, 3) uint8, got {tuple(img.shape)} {img.dtype}"
        )

    glyph = _glyph(text.upper(), img.device)
    s = max(1, int(scale))
    if s > 1:
        glyph = glyph.repeat_interleave(s, dim=0).repeat_interleave(s, dim=1)

    gh, gw = int(glyph.shape[0]), int(glyph.shape[1])
    if gh == 0 or gw == 0:
        return

    h, w = int(img.shape[0]), int(img.shape[1])
    x_i, y_i = int(x), int(y)
    color_t = _as_color(color, img.device)

    if bg:
        pad = max(1, s)
        rx0, ry0 = max(0, x_i - pad), max(0, y_i - pad)
        rx1, ry1 = min(w, x_i + gw + pad), min(h, y_i + gh + pad)
        if rx1 > rx0 and ry1 > ry0:
            region = img[ry0:ry1, rx0:rx1, :].to(torch.float32)
            img[ry0:ry1, rx0:rx1, :] = torch.round(region * 0.25).to(torch.uint8)

    x0, y0 = max(0, x_i), max(0, y_i)
    x1, y1 = min(w, x_i + gw), min(h, y_i + gh)
    if x1 <= x0 or y1 <= y0:
        return

    gx0, gy0 = x0 - x_i, y0 - y_i
    mask_crop = glyph[gy0 : gy0 + (y1 - y0), gx0 : gx0 + (x1 - x0)].to(torch.bool)
    if not torch.any(mask_crop):
        return
    region = img[y0:y1, x0:x1, :]
    region[mask_crop] = color_t


__all__ = ["draw_text"]
