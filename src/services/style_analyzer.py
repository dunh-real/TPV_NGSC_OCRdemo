"""
style_analyzer.py
Phân tích bold / underline của từng OCR block bằng pixel analysis.

Thuật toán:
- Bold     : avg stroke width của block / baseline toàn trang >= BOLD_RATIO,
             tính theo cột dọc để tránh false positive với dòng hỗn hợp.
- Italic   : không detect được đáng tin cậy qua pixel với ảnh scan → luôn False.
- Underline: tồn tại đường ngang liên tục (dark ratio > UNDERLINE_DARK) trong dải
             pixel ngay dưới bbox.
"""

import numpy as np
import cv2
from PIL import Image

BOLD_RATIO     = 1.40  # avg_sw / baseline >= ngưỡng này → cột đó là bold
BOLD_COL_THR   = 0.20  # >= 20% cột bold → block là bold
UNDERLINE_DARK = 0.30  # tỉ lệ pixel tối trong dải dưới bbox → underline
UNDERLINE_SCAN = 10    # số pixel quét dưới bbox để tìm underline


def _to_gray_array(img) -> np.ndarray:
    if isinstance(img, Image.Image):
        return np.array(img.convert("L"))
    if isinstance(img, np.ndarray):
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img
    raise TypeError(f"Unsupported image type: {type(img)}")


def _stroke_widths(binary_row: np.ndarray) -> list[int]:
    """Trả về list chiều rộng các run of dark pixels trong 1 hàng."""
    widths, in_stroke, w = [], False, 0
    for px in binary_row:
        if px:
            in_stroke = True
            w += 1
        elif in_stroke:
            widths.append(w)
            in_stroke = False
            w = 0
    if in_stroke and w > 0:
        widths.append(w)
    return widths


def _block_avg_sw(img_gray: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    """Tính avg stroke width của một vùng ảnh."""
    binary = (img_gray[y0:y1, x0:x1] < 128).astype(np.uint8)
    sws = []
    for row in binary:
        sws.extend(_stroke_widths(row))
    sws = [s for s in sws if 1 <= s <= 10]
    return float(np.mean(sws)) if sws else 0.0


def _clip_bbox(block: dict, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    x0 = max(0, int(block["x0"]))
    y0 = max(0, int(block["y0"]))
    x1 = min(img_w, int(block["x1"]))
    y1 = min(img_h, int(block["y1"]))
    return x0, y0, x1, y1


def compute_page_baseline(img_gray: np.ndarray, blocks: list[dict]) -> float:
    """
    Tính stroke width baseline của text thường trên trang.
    Dùng word-level avg (chỉ lấy word dài >= 3 ký tự chữ cái),
    lấy median của các word ở p25-p65 để loại bỏ bold và noise.
    """
    h, w = img_gray.shape
    word_avgs = []

    for b in blocks:
        x0, y0, x1, y1 = _clip_bbox(b, w, h)
        if x1 <= x0 or y1 <= y0 or (y1 - y0) < 8:
            continue
        words = b["text"].split()
        total_chars = sum(len(ww) + 1 for ww in words)
        if total_chars == 0:
            continue

        for wi, word in enumerate(words):
            # Chỉ lấy word có ít nhất 3 ký tự chữ cái (bỏ -, +, số, ký tự đơn)
            alpha_chars = [c for c in word if c.isalpha()]
            if len(alpha_chars) < 3:
                continue

            rs = sum(len(ww) + 1 for ww in words[:wi]) / total_chars
            re = (sum(len(ww) + 1 for ww in words[:wi]) + len(word)) / total_chars
            wx0 = max(0, x0 + int((x1 - x0) * rs))
            wx1 = min(w, x0 + int((x1 - x0) * re))
            if wx1 - wx0 < 8:
                continue

            binary = (img_gray[y0:y1, wx0:wx1] < 128).astype(np.uint8)
            sws = []
            for row in binary:
                sws.extend(_stroke_widths(row))
            sws = [s for s in sws if 1 <= s <= 10]
            if sws:
                word_avgs.append(float(np.mean(sws)))

    if not word_avgs:
        return 2.3

    lower = float(np.percentile(word_avgs, 25))
    upper = float(np.percentile(word_avgs, 65))
    normal = [x for x in word_avgs if lower <= x <= upper]
    return float(np.mean(normal)) if normal else float(np.median(word_avgs))


def detect_bold(crop_gray: np.ndarray, baseline: float) -> bool:
    """
    True nếu phần lớn nội dung block là chữ đậm.
    Chia crop thành các cột ~20px, tính avg stroke width từng cột,
    kết luận bold nếu >= BOLD_COL_THR cột vượt ngưỡng BOLD_RATIO * baseline.
    """
    if baseline <= 0:
        return False
    h, w = crop_gray.shape
    if w < 4:
        return False

    col_w = max(20, w // max(3, w // 20))
    bold_cols, total_cols = 0, 0

    for cx in range(0, w, col_w):
        col = crop_gray[:, cx:min(cx + col_w, w)]
        binary = (col < 128).astype(np.uint8)
        sws = []
        for row in binary:
            sws.extend(_stroke_widths(row))
        sws = [s for s in sws if 1 <= s <= 10]
        if not sws:
            continue
        total_cols += 1
        if (float(np.mean(sws)) / baseline) >= BOLD_RATIO:
            bold_cols += 1

    if total_cols == 0:
        return False
    return (bold_cols / total_cols) >= BOLD_COL_THR


def detect_italic(_crop_gray: np.ndarray) -> bool:
    """
    Italic detection qua pixel không đáng tin cậy với ảnh scan.
    Luôn trả về False.
    """
    return False


def detect_underline(img_gray: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> bool:
    """
    True nếu có đường gạch chân ngay dưới bbox.
    Quét UNDERLINE_SCAN hàng pixel dưới y1, tìm hàng có dark ratio > UNDERLINE_DARK.
    """
    h, w = img_gray.shape
    for dy in range(1, UNDERLINE_SCAN + 1):
        row_y = y1 + dy
        if row_y >= h:
            break
        row = img_gray[row_y, x0:x1]
        if len(row) == 0:
            continue
        if float((row < 128).mean()) > UNDERLINE_DARK:
            return True
    return False


def analyze_styles(img, blocks: list[dict]) -> list[dict]:
    """
    Phân tích style cho từng block, thêm field 'style'.

    Args:
        img: PIL Image hoặc numpy array của trang (cùng tọa độ với blocks).
        blocks: List block từ OCRService.

    Returns:
        List block đã thêm field 'style': {'bold': bool, 'italic': bool, 'underline': bool}
    """
    img_gray = _to_gray_array(img)
    img_h, img_w = img_gray.shape
    baseline = compute_page_baseline(img_gray, blocks)

    result = []
    for block in blocks:
        x0, y0, x1, y1 = _clip_bbox(block, img_w, img_h)
        b = dict(block)

        if x1 <= x0 or y1 <= y0:
            b["style"] = {"bold": False, "italic": False, "underline": False}
            result.append(b)
            continue

        crop = img_gray[y0:y1, x0:x1]
        b["style"] = {
            "bold":      detect_bold(crop, baseline),
            "italic":    detect_italic(crop),
            "underline": detect_underline(img_gray, x0, y0, x1, y1),
        }
        result.append(b)

    return result
