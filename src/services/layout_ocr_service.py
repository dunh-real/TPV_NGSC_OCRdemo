"""
layout_ocr_service.py

Sử dụng Deepdoc VietOCR để detect layout
"""

import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# deepdoc_vietocr path
_DEEPDOC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "deepdoc_vietocr"))
if _DEEPDOC_DIR not in sys.path:
    sys.path.insert(0, _DEEPDOC_DIR)
    sys.path.insert(0, os.path.join(_DEEPDOC_DIR, "vietocr"))

_ocr_instance = None


def _get_ocr():
    global _ocr_instance
    if _ocr_instance is None:
        from module.ocr import OCR
        _ocr_instance = OCR()
        logger.info("[LayoutOCR] VietOCR loaded")
    return _ocr_instance


@dataclass
class TextBlock:
    x0: float
    y0: float
    x1: float
    y1: float
    page: int = 1

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class LayoutOCRConfig:
    min_height:   float = 5.0
    min_width:    float = 5.0
    max_img_side: int   = 2000


class LayoutOCRService:
    def __init__(self, config: Optional[LayoutOCRConfig] = None):
        self.cfg = config or LayoutOCRConfig()
        _get_ocr()

    def process_pages(self, pages: list[Image.Image]) -> list[list[TextBlock]]:
        results = []
        for i, img in enumerate(pages):
            blocks = self._process_page(img, page_num=i + 1)
            results.append(blocks)
            logger.info(f"[LayoutOCR] Page {i+1}/{len(pages)} — {len(blocks)} blocks")
        return results

    def process_page(self, img: Image.Image, page_num: int = 1) -> list[TextBlock]:
        return self._process_page(img, page_num)

    def _process_page(self, img: Image.Image, page_num: int) -> list[TextBlock]:
        img_rgb, scale = self._prepare_image(img)
        ocr = _get_ocr()

        raw = ocr(np.array(img_rgb))
        if not raw:
            return []

        blocks = []
        for box, (text, conf) in raw:
            # box: [[x0,y0],[x1,y0],[x1,y1],[x0,y1]] 
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x0, y0 = min(xs) / scale, min(ys) / scale
            x1, y1 = max(xs) / scale, max(ys) / scale

            if (x1 - x0) < self.cfg.min_width or (y1 - y0) < self.cfg.min_height:
                continue

            blocks.append(TextBlock(x0=x0, y0=y0, x1=x1, y1=y1, page=page_num))

        blocks.sort(key=lambda b: (round(b.y0 / 5) * 5, b.x0))
        return blocks

    def _prepare_image(self, img: Image.Image) -> tuple[Image.Image, float]:
        img_rgb = img.convert("RGB")
        w, h    = img_rgb.size
        max_side = max(w, h)
        if max_side <= self.cfg.max_img_side:
            return img_rgb, 1.0
        scale   = self.cfg.max_img_side / max_side
        img_rgb = img_rgb.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return img_rgb, scale
