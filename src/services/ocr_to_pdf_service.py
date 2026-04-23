import json
import logging
import os

from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from services.font_style_service import FontStyleService

_FONT_DIR         = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "fonts"))
FONT_PATH         = os.path.join(_FONT_DIR, "TimesNewRoman.ttf")
FONT_PATH_BOLD    = os.path.join(_FONT_DIR, "TimesNewRoman-Bold.ttf")
FONT_PATH_ITALIC  = os.path.join(_FONT_DIR, "TimesNewRoman-Italic.ttf")
FONT_PATH_BOLD_ITALIC = os.path.join(_FONT_DIR, "TimesNewRoman-BoldItalic.ttf")

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "result_pdf"))

A4_W = 595.0
A4_H = 842.0

_FONT_REGISTERED = False
_style_svc       = FontStyleService()


def _register_font():
    global _FONT_REGISTERED
    if _FONT_REGISTERED:
        return
    pdfmetrics.registerFont(TTFont("TNR",            FONT_PATH))
    pdfmetrics.registerFont(TTFont("TNR-Bold",       FONT_PATH_BOLD))
    pdfmetrics.registerFont(TTFont("TNR-Italic",     FONT_PATH_ITALIC))
    pdfmetrics.registerFont(TTFont("TNR-BoldItalic", FONT_PATH_BOLD_ITALIC))
    _FONT_REGISTERED = True


def _pick_font(bold: bool, italic: bool) -> str:
    if bold and italic: return "TNR-BoldItalic"
    if bold:            return "TNR-Bold"
    if italic:          return "TNR-Italic"
    return "TNR"


class OcrToPdfService:
    def __init__(self, margin: float = 20.0, line_spacing: float = 1.1):
        for path in [FONT_PATH, FONT_PATH_BOLD, FONT_PATH_ITALIC, FONT_PATH_BOLD_ITALIC]:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Font not found: {path}")
        _register_font()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.margin       = margin
        self.line_spacing = line_spacing
        logging.info("[OcrToPdfService] Ready.")

    def _scale_x(self, val: float, src_dim: float, dst_dim: float) -> float:
        return (val / src_dim) * (dst_dim - 2 * self.margin) + self.margin

    def _scale_y(self, val: float, src_dim: float, dst_dim: float) -> float:
        center = (dst_dim - 2 * self.margin) / 2 + self.margin
        scaled = (val / src_dim) * (dst_dim - 2 * self.margin) + self.margin
        return center + (scaled - center) * self.line_spacing

    def _render_page(self, c: canvas.Canvas, page: dict):
        src_w = page["width"]
        src_h = page["height"]

        for block in page["blocks"]:
            text = block["text"].strip()
            if not text:
                continue

            x0 = self._scale_x(block["x0"], src_w, A4_W)
            y0 = self._scale_y(block["y0"], src_h, A4_H)
            x1 = self._scale_x(block["x1"], src_w, A4_W)
            y1 = self._scale_y(block["y1"], src_h, A4_H)
            pdf_y = A4_H - y1

            style     = block.get("style", {})
            is_bold   = style.get("bold",      False)
            is_italic = style.get("italic",    False)
            is_under  = style.get("underline", False)
            align     = style.get("align",     "left")
            font_size = float(style.get("size", 13.0))

            fn           = _pick_font(is_bold, is_italic)
            block_center = (x0 + x1) / 2
            line_h       = font_size * 1.3
            lines        = text.split("\n")

            c.setFont(fn, font_size)
            for i, line in enumerate(lines):
                ly = pdf_y + (len(lines) - 1 - i) * line_h
                if align == "center":
                    c.drawCentredString(block_center, ly, line)
                    lx0 = block_center - c.stringWidth(line, fn, font_size) / 2
                    lx1 = block_center + c.stringWidth(line, fn, font_size) / 2
                elif align == "right":
                    c.drawRightString(x1, ly, line)
                    lx0 = x1 - c.stringWidth(line, fn, font_size)
                    lx1 = x1
                else:
                    c.drawString(x0, ly, line)
                    lx0 = x0
                    lx1 = x0 + c.stringWidth(line, fn, font_size)

                if is_under:
                    c.setLineWidth(0.5)
                    c.line(lx0, ly - font_size * 0.15, lx1, ly - font_size * 0.15)

    def convert(self, ocr_json_path: str) -> str:
        if not os.path.isfile(ocr_json_path):
            raise FileNotFoundError(f"OCR JSON not found: {ocr_json_path}")

        with open(ocr_json_path, encoding="utf-8") as f:
            data = json.load(f)

        _style_svc.enrich(data)

        pdf_name = os.path.splitext(os.path.basename(ocr_json_path))[0]

        pdf_path = os.path.join(OUTPUT_DIR, f"{pdf_name}.pdf")

        c = canvas.Canvas(pdf_path, pagesize=(A4_W, A4_H))
        for page in data["pages"]:
            self._render_page(c, page)
            c.showPage()

        c.save()
        logging.info(f"[OcrToPdfService] Saved → {pdf_path}")
        
        return pdf_path