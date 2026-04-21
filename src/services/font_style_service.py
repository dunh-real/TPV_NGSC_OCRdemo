"""
font_style_service.py

Rules:
  - Quốc hiệu (CỘNG HÒA...): bold, center
  - Tiêu ngữ (ĐỘC LẬP... HẠNH PHÚC): bold, underline, center
  - Tên Tòa án: bold, center
  - Tên bản án / quyết định: bold, center
  - Header bold (NHẬN ĐỊNH, TUYÊN XỬ, NỘI DUNG...): bold
  - Bản án số / Ngày / V/v: regular (không bold)
  - Chữ in hoa (all-caps): bold
  - Nơi nhận: bold, italic, left
  - Nội dung chính: justify
  - Fallback: regular left
"""

import re
from dataclasses import dataclass


@dataclass
class BlockStyle:
    font:      str   = "Times New Roman"
    size:      float = 12.0
    bold:      bool  = False
    italic:    bool  = False
    underline: bool  = False
    align:     str   = "left"


_RE_QUOC_HIEU  = re.compile(r"(CỘNG\s+HÒA|CONG\s+HOA)", re.I)
_RE_TIEU_NGU   = re.compile(r"(ĐỘC\s+LẬP|DOC\s+LAP|HẠNH\s+PHÚC|HANH\s+PHUC)", re.I)
_RE_TEN_TOA    = re.compile(r"(TÒA\s+ÁN|TOA\s+AN|VIỆN\s+KIỂM\s+SÁT|VIEN\s+KIEM\s+SAT)", re.I)
_RE_TEN_BAN_AN = re.compile(r"^(BẢN\s+ÁN|BAN\s+AN|QUYẾT\s+ĐỊNH|QUYET\s+DINH|THÔNG\s+BÁO|THONG\s+BAO)", re.I)
_RE_HEADER     = re.compile(r"^(NHẬN\s+ĐỊNH|NHAN\s+DINH|TUYÊN\s+XỬ|TUYEN\s+XU|NỘI\s+DUNG\s+VỤ\s+ÁN|NOI\s+DUNG)", re.I)
_RE_SO_HIEU    = re.compile(r"^(Bản\s+án\s+số|Ban\s+an\s+so|Số\s+hiệu|So\s+hieu|Ngày|Ngay|V/v|VỀ\s+việc)", re.I)
_RE_CHU_KY     = re.compile(r"(TM\.|T\.M\.|Thẩm\s+phán|Tham\s+phan|Chủ\s+tọa|Chu\s+toa|Thư\s+ký|Thu\s+ky|Kiểm\s+sát\s+viên)", re.I)
_RE_NOI_NHAN   = re.compile(r"^(Nơi\s+nhận|Noi\s+nhan)\s*:?", re.I)
_RE_TABLE_CODE = re.compile(r"^\s*[\d,\.]+\s*$|^\s*[A-Z]{2,}\d+")


def _detect_alignment(x0: float, x1: float, page_w: float) -> str:
    center   = page_w / 2
    block_cx = (x0 + x1) / 2
    block_w  = x1 - x0

    if abs(block_cx - center) < page_w * 0.12 and block_w < page_w * 0.65:
        return "center"
    if x0 / page_w > 0.50 and block_w < page_w * 0.45:
        return "right"
    if block_w > page_w * 0.75:
        return "justify"
    return "left"


def _font_size_from_height(block_h: float) -> float:
    if block_h >= 85: return 16.0
    if block_h >= 72: return 14.0
    if block_h >= 55: return 13.0
    return 12.0


def infer_style(block: dict, page_w: float, page_h: float,
                page_blocks: list = None) -> BlockStyle:
    text  = block.get("text", "").strip()
    x0    = block["x0"]
    y0    = block["y0"]
    x1    = block["x1"]
    y1    = block["y1"]
    blk_h = y1 - y0
    blk_w = x1 - x0

    style       = BlockStyle()
    style.align = _detect_alignment(x0, x1, page_w)

    # Rule 1: Quốc hiệu (CỘNG HÒA XÃ HỘI...) 
    if _RE_QUOC_HIEU.search(text):
        style.bold  = True
        style.size  = 13.0
        style.align = "center"
        return style

    # Rule 2: Tiêu ngữ (ĐỘC LẬP - TỰ DO - HẠNH PHÚC)
    if _RE_TIEU_NGU.search(text):
        style.bold      = True
        style.underline = True
        style.size      = 13.0
        style.align     = "center"
        return style

    # Rule 3: Tên Tòa án / Cơ quan 
    if _RE_TEN_TOA.match(text):
        style.bold  = True
        style.size  = 14.0
        style.align = "center"
        return style

    # Rule 4: Bản án số / Ngày / V/v → regular (trước tên bản án)
    if _RE_SO_HIEU.match(text):
        style.bold  = False
        style.size  = 13.0
        return style

    # Rule 5: Tên bản án / quyết định 
    if _RE_TEN_BAN_AN.match(text):
        style.bold  = True
        style.size  = 14.0
        style.align = "center"
        return style

    # Rule 6: Header in hoa (NHẬN ĐỊNH, TUYÊN XỬ...) 
    if _RE_HEADER.match(text):
        style.bold  = True
        style.size  = 13.0
        style.align = "center"
        return style

    # Rule 7: Nơi nhận → bold, italic, left 
    if _RE_NOI_NHAN.match(text):
        style.bold   = True
        style.italic = True
        style.size   = 12.0
        style.align  = "left"
        return style

    # Rule 8: Chữ ký / Chức danh
    if _RE_CHU_KY.search(text):
        style.bold  = False
        style.size  = 12.0
        style.align = "right" if style.align in ("right", "center") else "left"
        return style

    # Rule 9: Bảng / mã số
    if _RE_TABLE_CODE.match(text):
        style.font  = "Courier New"
        style.size  = 11.0
        style.align = "left"
        return style

    # Rule 10: Chữ in hoa (all-caps) → bold 
    if text.isupper() and len(text) > 3:
        style.bold  = True
        style.size  = 13.0
        return style

    # Rule 11: Nội dung chính (block rộng) → justify 
    if blk_w > page_w * 0.6:
        style.size  = 13.0
        style.align = "justify"
        return style

    # Fallback 
    style.size = _font_size_from_height(blk_h)
    return style


class FontStyleService:
    def enrich(self, ocr_data: dict) -> dict:
        for page in ocr_data.get("pages", []):
            page_w = page["width"]
            page_h = page["height"]
            blocks = page.get("blocks", [])
            for block in blocks:
                s = infer_style(block, page_w, page_h, blocks)
                block["style"] = {
                    "font":      s.font,
                    "size":      s.size,
                    "bold":      s.bold,
                    "italic":    s.italic,
                    "underline": s.underline,
                    "align":     s.align,
                }
        return ocr_data
