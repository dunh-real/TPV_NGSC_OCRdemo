import json
import logging
import os
import numpy as np
import pdfplumber
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

_FONT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "fonts"))
FONT_PATH           = os.path.join(_FONT_DIR, "TimesNewRoman.ttf")
FONT_PATH_BOLD      = os.path.join(_FONT_DIR, "TimesNewRoman-Bold.ttf")
FONT_PATH_ITALIC    = os.path.join(_FONT_DIR, "TimesNewRoman-Italic.ttf")
FONT_PATH_BOLD_ITALIC = os.path.join(_FONT_DIR, "TimesNewRoman-BoldItalic.ttf")

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "result_pdf"))

A4_W = 595.0
A4_H = 842.0
FONT_SIZE = 13.0
BOLD_RATIO = 1.40       # avg_sw / baseline >= ngưỡng → word là bold
BOLD_MAJORITY = 0.50    # >= 50% word alpha trong block là bold → block là bold

_FONT_REGISTERED = False


def _register_font():
    global _FONT_REGISTERED
    if not _FONT_REGISTERED:
        pdfmetrics.registerFont(TTFont("TNR",            FONT_PATH))
        pdfmetrics.registerFont(TTFont("TNR-Bold",       FONT_PATH_BOLD))
        pdfmetrics.registerFont(TTFont("TNR-Italic",     FONT_PATH_ITALIC))
        pdfmetrics.registerFont(TTFont("TNR-BoldItalic", FONT_PATH_BOLD_ITALIC))
        _FONT_REGISTERED = True


def _stroke_widths(binary_row: np.ndarray) -> list:
    widths, in_s, w = [], False, 0
    for px in binary_row:
        if px:
            in_s = True; w += 1
        elif in_s:
            widths.append(w); in_s = False; w = 0
    if in_s and w > 0:
        widths.append(w)
    return widths


def _region_avg_sw(img_gray: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    """Tính avg stroke width của một vùng ảnh."""
    if x1 <= x0 or y1 <= y0 or (x1 - x0) < 15:
        return 0.0
    crop = img_gray[y0:y1, x0:x1]
    binary = (crop < 128).astype(np.uint8)
    sws = []
    for row in binary:
        sws.extend(_stroke_widths(row))
    sws = [s for s in sws if 1 <= s <= 10]
    return float(np.mean(sws)) if sws else 0.0


def _word_bbox_px(bx0: int, bx1: int, words: list, wi: int, word: str) -> tuple:
    """Ước tính bbox pixel của word theo tỉ lệ ký tự trong block."""
    total_chars = max(sum(len(w) + 1 for w in words), 1)
    rs = sum(len(w) + 1 for w in words[:wi]) / total_chars
    re = (sum(len(w) + 1 for w in words[:wi]) + len(word)) / total_chars
    return bx0 + int((bx1 - bx0) * rs), bx0 + int((bx1 - bx0) * re)


def _block_bold_map(img_gray: np.ndarray, bx0: int, by0: int, bx1: int, by1: int,
                    words: list, baseline: float) -> list:
    """
    Trả về list bool bold cho từng word trong block.
    Áp dụng majority vote: nếu < BOLD_MAJORITY word alpha là bold → toàn block không bold.
    Nếu >= BOLD_MAJORITY → dùng kết quả từng word.
    """
    img_h, img_w = img_gray.shape
    results = []
    alpha_indices = []

    for wi, word in enumerate(words):
        has_alpha = any(ch.isalpha() for ch in word)
        if not has_alpha:
            results.append(False)
            continue
        wx0, wx1 = _word_bbox_px(bx0, bx1, words, wi, word)
        wx0 = max(0, min(wx0, img_w))
        wx1 = max(0, min(wx1, img_w))
        avg = _region_avg_sw(img_gray, wx0, by0, wx1, by1)
        is_bold = avg > 0 and (avg / baseline) >= BOLD_RATIO
        results.append(is_bold)
        alpha_indices.append((wi, is_bold))

    if not alpha_indices:
        return [False] * len(words)

    bold_count = sum(1 for _, b in alpha_indices if b)
    bold_ratio = bold_count / len(alpha_indices)

    # Majority vote: nếu < 50% word alpha là bold → không có word nào bold
    if bold_ratio < BOLD_MAJORITY:
        return [False] * len(words)

    return results


def _pick_font(is_bold: bool, is_italic: bool) -> str:
    if is_bold and is_italic:
        return "TNR-BoldItalic"
    if is_bold:
        return "TNR-Bold"
    if is_italic:
        return "TNR-Italic"
    return "TNR"


class OcrToPdfService:
    def __init__(self, margin: float = 20.0):
        for path in [FONT_PATH, FONT_PATH_BOLD, FONT_PATH_ITALIC, FONT_PATH_BOLD_ITALIC]:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Font not found: {path}")
        _register_font()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.margin = margin
        logging.info("[OcrToPdfService] Ready. Font: Times New Roman, size 13")

    def _scale(self, val: float, src_dim: float, dst_dim: float) -> float:
        return (val / src_dim) * (dst_dim - 2 * self.margin) + self.margin

    def _render_page(self, c: canvas.Canvas, page: dict, img_gray: np.ndarray, baseline: float):
        src_w, src_h = page["width"], page["height"]
        img_h, img_w = img_gray.shape

        for block in page["blocks"]:
            text = block["text"].strip()
            if not text:
                continue

            x0 = self._scale(block["x0"], src_w, A4_W)
            y0 = self._scale(block["y0"], src_h, A4_H)
            x1 = self._scale(block["x1"], src_w, A4_W)
            y1 = self._scale(block["y1"], src_h, A4_H)
            pdf_y = A4_H - y1

            page_center = A4_W / 2
            block_center = (x0 + x1) / 2
            if abs(block_center - page_center) < A4_W * 0.08:
                align = "center"
            elif x0 > A4_W * 0.55:
                align = "right"
            else:
                align = "left"

            lines = text.split("\n")
            n_lines = max(len(lines), 1)
            line_h = FONT_SIZE * 1.3

            block_style = block.get("style", {})
            block_italic = block_style.get("italic", False)
            block_underline = block_style.get("underline", False)

            bx0 = max(0, int(block["x0"])); bx1 = min(img_w, int(block["x1"]))
            by0 = max(0, int(block["y0"])); by1 = min(img_h, int(block["y1"]))

            if block.get("type") == "table":
                c.setFont("TNR", FONT_SIZE)
                for i, line in enumerate(lines):
                    ly = pdf_y + (n_lines - 1 - i) * line_h
                    c.drawString(x0, ly, line)

            elif align in ("center", "right"):
                # Center/right: render cả dòng, dùng block-level bold
                block_avg = _region_avg_sw(img_gray, bx0, by0, bx1, by1)
                block_bold = block_avg > 0 and (block_avg / baseline) >= BOLD_RATIO
                fn = _pick_font(block_bold, block_italic)
                c.setFont(fn, FONT_SIZE)
                for i, line in enumerate(lines):
                    ly = pdf_y + (n_lines - 1 - i) * line_h
                    if align == "center":
                        c.drawCentredString(block_center, ly, line)
                    else:
                        c.drawRightString(x1, ly, line)

            else:
                # Left align: word-level bold với majority vote
                for i, line in enumerate(lines):
                    ly = pdf_y + (n_lines - 1 - i) * line_h
                    words = line.split(" ")

                    bold_map = _block_bold_map(img_gray, bx0, by0, bx1, by1, words, baseline)

                    cur_x = x0
                    for wi, word in enumerate(words):
                        if not word:
                            cur_x += c.stringWidth(" ", "TNR", FONT_SIZE)
                            continue
                        fn = _pick_font(bold_map[wi], block_italic)
                        c.setFont(fn, FONT_SIZE)
                        c.drawString(cur_x, ly, word)
                        cur_x += c.stringWidth(word + " ", fn, FONT_SIZE)

            if block_underline:
                c.setLineWidth(0.5)
                c.line(x0, pdf_y - FONT_SIZE * 0.15, x1, pdf_y - FONT_SIZE * 0.15)

    def convert(self, ocr_json_path: str) -> str:
        if not os.path.isfile(ocr_json_path):
            raise FileNotFoundError(f"OCR JSON not found: {ocr_json_path}")

        with open(ocr_json_path, encoding="utf-8") as f:
            data = json.load(f)

        pdf_name = os.path.splitext(os.path.basename(ocr_json_path))[0]
        raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw"))
        pdf_raw = os.path.join(raw_dir, pdf_name + ".pdf")
        pdf_path = os.path.join(OUTPUT_DIR, f"{pdf_name}.pdf")

        c = canvas.Canvas(pdf_path, pagesize=(A4_W, A4_H))
        for page in data["pages"]:
            img_gray = None
            baseline = 2.3
            if os.path.isfile(pdf_raw):
                try:
                    with pdfplumber.open(pdf_raw) as pdf:
                        img = pdf.pages[page["page"] - 1].to_image(resolution=216).annotated
                    img_gray = np.array(img.convert("L"))
                    from services.style_analyzer import compute_page_baseline
                    baseline = compute_page_baseline(img_gray, page["blocks"])
                except Exception as e:
                    logging.warning(f"[OcrToPdfService] Cannot load page image: {e}")

            if img_gray is None:
                img_gray = np.ones((page["height"], page["width"]), dtype=np.uint8) * 255

            self._render_page(c, page, img_gray, baseline)
            c.showPage()

        c.save()
        logging.info(f"[OcrToPdfService] Saved → {pdf_path}")
        return pdf_path
