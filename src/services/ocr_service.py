import logging
import json
import os
import sys
import re
import time
import threading
import numpy as np
import pdfplumber
from PIL import Image, ImageDraw

DEEPDOC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "deepdoc_vietocr"))
sys.path.insert(0, DEEPDOC_DIR)
sys.path.insert(0, os.path.join(DEEPDOC_DIR, "vietocr"))

from module.ocr import OCR
from module import LayoutRecognizer, TableStructureRecognizer

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "result_ocr"))

_PDFPLUMBER_LOCK = threading.Lock()


class OCRService:
    def __init__(self):
        logging.info("Initializing OCRService...")
        self.ocr = OCR()
        self.layout_recognizer = LayoutRecognizer("layout")
        self.tsr = TableStructureRecognizer()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logging.info("OCRService ready.")

    def _pdf_to_images(self, pdf_path: str, zoomin: int = 3) -> list:
        with _PDFPLUMBER_LOCK:
            pdf = pdfplumber.open(pdf_path)
            images = [p.to_image(resolution=72 * zoomin).annotated for p in pdf.pages]
            pdf.close()
        return images

    def _extract_table_blocks(self, img: Image.Image, table_region: dict, page_offset_x: int, page_offset_y: int) -> list[dict]:
        """Trả về list block từ vùng bảng, tọa độ đã được offset về page gốc."""
        x0, y0, x1, y1 = map(int, table_region["bbox"])
        table_img = img.crop((x0, y0, x1, y1))
        tb_cpns = self.tsr([table_img])[0]
        boxes = self.ocr(np.array(table_img))
        if not boxes:
            return []

        sorted_boxes = LayoutRecognizer.sort_Y_firstly(
            [{"x0": b[0][0], "x1": b[1][0],
              "top": b[0][1], "text": t[0],
              "bottom": b[-1][1],
              "layout_type": "table",
              "page_number": 0} for b, t in boxes if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]],
            np.mean([b[-1][1] - b[0][1] for b, _ in boxes]) / 3
        )

        def gather(kwd, fzy=10, ption=0.6):
            eles = LayoutRecognizer.sort_Y_firstly(
                [r for r in tb_cpns if re.match(kwd, r["label"])], fzy)
            eles = LayoutRecognizer.layouts_cleanup(sorted_boxes, eles, 5, ption)
            return LayoutRecognizer.sort_Y_firstly(eles, 0)

        headers = gather(r".*header$")
        rows = gather(r".* (row|header)")
        spans = gather(r".*spanning")
        clmns = sorted([r for r in tb_cpns if re.match(r"table column$", r["label"])], key=lambda x: x["x0"])
        clmns = LayoutRecognizer.layouts_cleanup(sorted_boxes, clmns, 5, 0.5)

        for b in sorted_boxes:
            ii = LayoutRecognizer.find_overlapped_with_threashold(b, rows, thr=0.3)
            if ii is not None:
                b["R"] = ii; b["R_top"] = rows[ii]["top"]; b["R_bott"] = rows[ii]["bottom"]
            ii = LayoutRecognizer.find_overlapped_with_threashold(b, headers, thr=0.3)
            if ii is not None:
                b["H_top"] = headers[ii]["top"]; b["H_bott"] = headers[ii]["bottom"]
                b["H_left"] = headers[ii]["x0"]; b["H_right"] = headers[ii]["x1"]; b["H"] = ii
            ii = LayoutRecognizer.find_horizontally_tightest_fit(b, clmns)
            if ii is not None:
                b["C"] = ii; b["C_left"] = clmns[ii]["x0"]; b["C_right"] = clmns[ii]["x1"]
            ii = LayoutRecognizer.find_overlapped_with_threashold(b, spans, thr=0.3)
            if ii is not None:
                b["H_top"] = spans[ii]["top"]; b["H_bott"] = spans[ii]["bottom"]
                b["H_left"] = spans[ii]["x0"]; b["H_right"] = spans[ii]["x1"]; b["SP"] = ii

        markdown = TableStructureRecognizer.construct_table(sorted_boxes, markdown=True)

        # Trả về 1 block đại diện cho toàn bộ bảng với tọa độ page gốc
        return [{
            "text": markdown,
            "x0": x0 + page_offset_x,
            "y0": y0 + page_offset_y,
            "x1": x1 + page_offset_x,
            "y1": y1 + page_offset_y,
            "type": "table",
        }]

    def _process_page(self, img: Image.Image, page_idx: int, threshold: float = 0.5) -> dict:
        """
        Xử lý một trang, trả về dict chứa kích thước trang và list các block có bbox.
        """
        img_w, img_h = img.size
        img_arr = np.array(img)
        layouts = self.layout_recognizer.forward([img_arr], thr=threshold)[0]

        mask = Image.new("1", img.size, 0)
        draw = ImageDraw.Draw(mask)
        blocks = []

        for region in layouts:
            bbox = region.get("bbox", [region.get("x0", 0), region.get("top", 0),
                                       region.get("x1", 0), region.get("bottom", 0)])
            rx0, ry0, rx1, ry1 = map(int, bbox)
            draw.rectangle([rx0, ry0, rx1, ry1], fill=1)

            if region.get("type", "").lower() == "table" and region.get("score", 1.0) >= threshold:
                region["bbox"] = bbox
                table_blocks = self._extract_table_blocks(img, region, 0, 0)
                blocks.extend(table_blocks)

        # OCR vùng không phải table (text, title, ...)
        inv_mask = mask.point(lambda p: 1 - p)
        if inv_mask.getbbox():
            cx0, cy0, cx1, cy1 = inv_mask.getbbox()
            crop_arr = np.array(img.crop((cx0, cy0, cx1, cy1)))
            ocr_results = self.ocr(crop_arr)
            if ocr_results:
                for box, (text, score) in ocr_results:
                    if not text:
                        continue
                    # box là tọa độ trong crop → offset về page gốc
                    bx0 = box[0][0] + cx0
                    by0 = box[0][1] + cy0
                    bx1 = box[1][0] + cx0
                    by1 = box[-1][1] + cy0
                    blocks.append({
                        "text": text,
                        "x0": float(bx0),
                        "y0": float(by0),
                        "x1": float(bx1),
                        "y1": float(by1),
                        "type": "text",
                    })

        # Sort theo y0 rồi x0
        blocks.sort(key=lambda b: (b["y0"], b["x0"]))

        return {
            "page": page_idx + 1,
            "width": img_w,
            "height": img_h,
            "blocks": blocks,
        }

    def process(self, pdf_path: str, threshold: float = 0.5) -> str:
        """
        OCR một file PDF, lưu kết quả JSON (có bbox) vào OUTPUT_DIR.

        Args:
            pdf_path:  Đường dẫn tuyệt đối tới file PDF.
            threshold: Ngưỡng confidence cho layout detection (default 0.5).

        Returns:
            Đường dẫn tới file JSON kết quả.
        """
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{pdf_name}.json")

        logging.info(f"[OCRService] Processing: {pdf_path}")
        total_start = time.time()

        images = self._pdf_to_images(pdf_path)
        logging.info(f"[OCRService] {len(images)} pages loaded.")

        pages = []
        for idx, img in enumerate(images):
            t0 = time.time()
            page_data = self._process_page(img, idx, threshold)
            pages.append(page_data)
            logging.info(f"[OCRService] Page {idx + 1}/{len(images)} done in {time.time() - t0:.2f}s — {len(page_data['blocks'])} blocks")

        result = {"pages": pages}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logging.info(f"[OCRService] Done in {time.time() - total_start:.2f}s → {out_path}")
        return out_path

    @staticmethod
    def to_plain_text(ocr_json_path: str) -> str:
        """
        Chuyển file JSON OCR thành plain text để đưa vào LLM.
        Giữ nguyên thứ tự đọc (top-to-bottom), phân trang bằng dấu ---
        """
        with open(ocr_json_path, encoding="utf-8") as f:
            data = json.load(f)

        pages_text = []
        for page in data["pages"]:
            lines = [b["text"] for b in page["blocks"] if b["text"].strip()]
            pages_text.append(f"<!-- Page {page['page']} -->\n" + "\n".join(lines))

        return "\n\n---\n\n".join(pages_text)
