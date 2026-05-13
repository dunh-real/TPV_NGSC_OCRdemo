import asyncio
import base64
import json
import logging
import os
import re
import time
from io import BytesIO
from typing import Optional

import pdfplumber
import requests
from PIL import Image, ImageDraw, ImageFont
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from services.layout_ocr_service import LayoutOCRService, LayoutOCRConfig, TextBlock

logger = logging.getLogger(__name__)

SERVER_URL = os.getenv("QWEN36_OCR_URL", "http://100.75.29.73:8008")
MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "result_ocr"))
_HEADERS   = {}
ZOOMIN     = 3
_CROP_PAD  = 4


def _make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=2, status_forcelist=[502, 503, 504],
                  allowed_methods=["GET", "POST"])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://",  HTTPAdapter(max_retries=retry))
    return s


def _crop_block(img: Image.Image, blk: TextBlock, pad: int = _CROP_PAD) -> Image.Image:
    iw, ih = img.size
    x0 = max(0, int(blk.x0) - pad)
    y0 = max(0, int(blk.y0) - pad)
    x1 = min(iw, int(blk.x1) + pad)
    y1 = min(ih, int(blk.y1) + pad)
    return img.convert("RGB").crop((x0, y0, x1, y1))


def _build_grid(crops: list[Image.Image], global_start: int,
                cell_h: int = 60, max_w: int = 1400) -> Image.Image:
    label_w = 40
    pad     = 4
    rows    = []
    for i, crop in enumerate(crops):
        cw, ch = crop.size
        if ch == 0:
            ch = 1
        scale   = cell_h / ch
        new_w   = min(int(cw * scale), max_w - label_w - pad * 2)
        resized = crop.resize((new_w, cell_h), Image.LANCZOS)
        rows.append((i, resized))

    total_h = (cell_h + pad) * len(rows) + pad
    grid    = Image.new("RGB", (max_w, total_h), color=(240, 240, 240))
    draw    = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for row_i, (local_i, resized) in enumerate(rows):
        global_i = global_start + local_i + 1
        y_off    = pad + row_i * (cell_h + pad)
        draw.rectangle([0, y_off, label_w - 2, y_off + cell_h], fill=(220, 50, 50))
        draw.text((4, y_off + cell_h // 2 - 8), str(global_i), fill="white", font=font)
        grid.paste(resized, (label_w, y_off))
        draw.line([(0, y_off + cell_h + pad // 2), (max_w, y_off + cell_h + pad // 2)],
                  fill=(180, 180, 180), width=1)

    return grid


def _image_to_base64(img: Image.Image, quality: int = 85) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _build_prompt(global_start: int, batch_size: int) -> str:
    first = global_start + 1
    last  = global_start + batch_size
    return (
        f"Ảnh gồm {batch_size} dòng văn bản tiếng Việt được đánh số từ {first} đến {last} "
        f"(nhãn đỏ bên trái mỗi dòng).\n\n"
        f"Nhiệm vụ: OCR chính xác text tiếng Việt của từng dòng. Giữ nguyên dấu tiếng Việt.\n"
        f"KHÔNG suy nghĩ, KHÔNG giải thích. Chỉ trả về JSON:\n"
        f'{{"{first}": "text dòng {first}", ..., "{last}": "text dòng {last}"}}'
    )


def _call_vllm(session: requests.Session, base_url: str,
               img_b64: str, prompt: str, timeout: int) -> dict:
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": [
            {"type": "text",      "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        ]}],
        "temperature": 0.0,
        "max_tokens":  4096,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = session.post(f"{base_url}/v1/chat/completions",
                        headers=_HEADERS, json=payload, timeout=timeout)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    logger.debug(f"[Qwen36] raw: {raw[:200]}")

    if "<think>" in raw:
        end = raw.find("</think>")
        raw = raw[end + len("</think>"):].strip() if end != -1 else raw

    m = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
    if m:
        raw = m.group(1).strip()
    elif not raw.lstrip().startswith("{"):
        start = raw.find("{")
        if start >= 0:
            raw = raw[start:]

    return json.loads(raw)


def _assemble_page(blocks: list[TextBlock], text_map: dict, page_num: int, img_w: int, img_h: int) -> dict:
    result_blocks = []

    for i, blk in enumerate(blocks, start=1):
        text = text_map.get(str(i), "").strip()
        if not text:
            continue
        result_blocks.append({
            "text": text,
            "x0":   round(blk.x0, 1),
            "y0":   round(blk.y0, 1),
            "x1":   round(blk.x1, 1),
            "y1":   round(blk.y1, 1),
            "page": page_num,
        })

    return {"page": page_num, "width": img_w, "height": img_h, "blocks": result_blocks}


class Qwen36OcrService:
    def __init__(self,
                 url: str = SERVER_URL,
                 timeout: int = 300,
                 batch_size: int = 20,
                 max_concurrent: int = 4,
                 layout_cfg: Optional[LayoutOCRConfig] = None):
        self.url            = url.rstrip("/")
        self.timeout        = timeout
        self.batch_size     = batch_size
        self.max_concurrent = max_concurrent
        self.session        = _make_session()
        self.lay_svc        = LayoutOCRService(layout_cfg or LayoutOCRConfig())
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        try:
            r = self.session.get(f"{self.url}/v1/models", headers=_HEADERS, timeout=10)
            models = [m["id"] for m in r.json().get("data", [])]
            logger.info(f"[Qwen36] Connected. Models: {models}")
        except Exception as e:
            logger.warning(f"[Qwen36] Cannot reach server: {e}")

    def _prepare_all_pages(self, pdf_path: str):
        """Convert PDF → images + layout detection (blocking)."""
        with pdfplumber.open(pdf_path) as pdf:
            images = [p.to_image(resolution=72 * ZOOMIN).annotated for p in pdf.pages]
        all_blocks = self.lay_svc.process_pages(images)
        return images, all_blocks

    def _ocr_page(self, img: Image.Image, blocks: list[TextBlock]) -> dict:
        text_map = {}
        total    = len(blocks)

        for start in range(0, total, self.batch_size):
            batch   = blocks[start:start + self.batch_size]
            crops   = [_crop_block(img, blk) for blk in batch]
            grid    = _build_grid(crops, global_start=start)
            img_b64 = _image_to_base64(grid)
            prompt  = _build_prompt(global_start=start, batch_size=len(batch))

            try:
                raw_map = _call_vllm(self.session, self.url, img_b64, prompt, self.timeout)
                text_map.update(raw_map)
                logger.debug(f"[Qwen36] batch {start+1}-{start+len(batch)}/{total}: {len(raw_map)} keys")
            except Exception as e:
                logger.error(f"[Qwen36] batch {start+1}-{start+len(batch)} error: {e}")

        missing = [str(i) for i in range(1, total + 1) if str(i) not in text_map]
        if missing:
            logger.warning(f"[Qwen36] {len(missing)} blocks missing: {missing[:10]}")

        return text_map

    async def _ocr_page_concurrent(self, img: Image.Image, blocks: list[TextBlock]) -> dict:
        """OCR all batches of a page concurrently via asyncio.gather."""
        text_map = {}
        total    = len(blocks)
        sem      = asyncio.Semaphore(self.max_concurrent)

        async def _run_batch(start: int, batch: list[TextBlock]):
            crops   = [_crop_block(img, blk) for blk in batch]
            grid    = _build_grid(crops, global_start=start)
            img_b64 = _image_to_base64(grid)
            prompt  = _build_prompt(global_start=start, batch_size=len(batch))

            async with sem:
                return await asyncio.to_thread(
                    _call_vllm, self.session, self.url,
                    img_b64, prompt, self.timeout,
                )

        tasks = []
        for start in range(0, total, self.batch_size):
            batch = blocks[start:start + self.batch_size]
            tasks.append((start, len(batch), asyncio.ensure_future(_run_batch(start, batch))))

        for start, batch_len, fut in tasks:
            try:
                raw_map = await fut
                text_map.update(raw_map)
                logger.debug(f"[Qwen36] batch {start+1}-{start+batch_len}/{total}: {len(raw_map)} keys")
            except Exception as e:
                logger.error(f"[Qwen36] batch {start+1}-{start+batch_len}/{total} error: {e}")

        missing = [str(i) for i in range(1, total + 1) if str(i) not in text_map]
        if missing:
            logger.warning(f"[Qwen36] {len(missing)} blocks missing: {missing[:10]}")

        return text_map

    def _ocr_single_page(self, img: Image.Image, blocks: list[TextBlock], page_num: int) -> dict:
        """OCR one page (sync, sequential batches)."""
        img_w, img_h = img.size
        if not blocks:
            logger.warning(f"[Qwen36] Page {page_num}: no blocks")
            return {"page": page_num, "width": img_w, "height": img_h, "blocks": []}
        try:
            text_map = self._ocr_page(img, blocks)
            return _assemble_page(blocks, text_map, page_num, img_w, img_h)
        except Exception as e:
            logger.error(f"[Qwen36] Page {page_num} error: {e}")
            return {"page": page_num, "width": img_w, "height": img_h, "blocks": []}

    async def _ocr_single_page_async(self, img: Image.Image, blocks: list[TextBlock], page_num: int) -> dict:
        """OCR one page (async, concurrent batches)."""
        img_w, img_h = img.size
        if not blocks:
            logger.warning(f"[Qwen36] Page {page_num}: no blocks")
            return {"page": page_num, "width": img_w, "height": img_h, "blocks": []}
        try:
            text_map = await self._ocr_page_concurrent(img, blocks)
            return _assemble_page(blocks, text_map, page_num, img_w, img_h)
        except Exception as e:
            logger.error(f"[Qwen36] Page {page_num} error: {e}")
            return {"page": page_num, "width": img_w, "height": img_h, "blocks": []}

    def process(self, pdf_path: str) -> str:
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{pdf_name}.json")

        logger.info(f"[Qwen36] Processing: {pdf_path}")
        t0 = time.time()

        images, all_blocks = self._prepare_all_pages(pdf_path)

        pages = []
        for idx, (img, blocks) in enumerate(zip(images, all_blocks)):
            page_num = idx + 1
            page = self._ocr_single_page(img, blocks, page_num)
            pages.append(page)
            logger.info(f"[Qwen36] Page {page_num}/{len(images)} — {len(page['blocks'])} blocks")

        n_total = sum(len(p["blocks"]) for p in pages)
        logger.info(f"[Qwen36] Done in {time.time()-t0:.2f}s — {len(pages)} pages, {n_total} blocks")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"pages": pages}, f, ensure_ascii=False, indent=2)

        return out_path

    async def process_streaming(self, pdf_path: str, queue: asyncio.Queue) -> str:
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{pdf_name}.json")

        logger.info(f"[Qwen36] Streaming process: {pdf_path}")
        t0 = time.time()

        # Prep: PDF → images + layout detection (blocking, run in thread)
        images, all_blocks = await asyncio.to_thread(
            self._prepare_all_pages, pdf_path
        )

        pages = []
        for idx, (img, blocks) in enumerate(zip(images, all_blocks)):
            page_num = idx + 1

            # OCR per page (concurrent batches within page)
            page = await self._ocr_single_page_async(img, blocks, page_num)

            pages.append(page)
            await queue.put(page)
            logger.info(
                f"[Qwen36] Page {page_num}/{len(images)} — "
                f"{len(page['blocks'])} blocks → queued"
            )

        # Save JSON for PDF/TIFF rendering
        n_total = sum(len(p["blocks"]) for p in pages)
        logger.info(f"[Qwen36] Done in {time.time()-t0:.2f}s — {len(pages)} pages, {n_total} blocks")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"pages": pages}, f, ensure_ascii=False, indent=2)

        return out_path
