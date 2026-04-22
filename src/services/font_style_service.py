import re
from dataclasses import dataclass


@dataclass
class BlockStyle:
    font:      str   = "Times New Roman"
    size:      float = 13.0
    bold:      bool  = False
    italic:    bool  = False
    underline: bool  = False
    align:     str   = "left"


_RE_TIEU_NGU = re.compile(r"(ĐỘC\s+LẬP|DOC\s+LAP).*(TỰ\s+DO|TU\s+DO).*(HẠNH\s+PHÚC|HANH\s+PHUC)", re.I)
_RE_NOI_NHAN = re.compile(r"^(Nơi\s+nhận|Noi\s+nhan)\s*:?", re.I)


def _is_allcaps(text: str) -> bool:
    """True nếu toàn bộ ký tự chữ cái trong text đều là chữ hoa."""
    letters = [c for c in text if c.isalpha()]
    return len(letters) > 0 and all(c.isupper() for c in letters)


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
    y1    = block["y1"]
    y0    = block["y0"]
    x1    = block["x1"]
    blk_h = y1 - y0
    blk_w = x1 - x0

    style       = BlockStyle()
    style.align = _detect_alignment(x0, x1, page_w)
    style.size  = _font_size_from_height(blk_h)

    # Nơi nhận: bold + italic
    if _RE_NOI_NHAN.match(text):
        style.bold   = True
        style.italic = True
        style.align  = "left"
        return style

    # Độc lập - Tự do - Hạnh Phúc: bold + underline
    if _RE_TIEU_NGU.search(text):
        style.bold      = True
        style.underline = True
        style.align     = "center"
        return style

    # All-caps: bold
    if _is_allcaps(text):
        style.bold = True
        return style

    # Còn lại: regular
    return style


class FontStyleService:
    def enrich(self, ocr_data: dict) -> dict:
        for page in ocr_data.get("pages", []):
            page_w = page["width"]
            page_h = page["height"]
            for block in page.get("blocks", []):
                s = infer_style(block, page_w, page_h)
                block["style"] = {
                    "font":      s.font,
                    "size":      s.size,
                    "bold":      s.bold,
                    "italic":    s.italic,
                    "underline": s.underline,
                    "align":     s.align,
                }
        return ocr_data
