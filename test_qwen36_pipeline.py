import asyncio
import os
import sys
import json
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from services.qwen36_ocr_service import Qwen36OcrService
from services.ocr_to_pdf_service import OcrToPdfService

PDF_PATH   = os.path.join(os.path.dirname(__file__), "data", "raw", "BA_05.2021.DS-ST.pdf")
QWEN36_URL = os.getenv("QWEN36_OCR_URL", "https://vks-ocr-hvks.loca.lt")


def test_pipeline():
    """Test sync pipeline."""
    assert os.path.isfile(PDF_PATH), f"PDF not found: {PDF_PATH}"

    ocr_svc = Qwen36OcrService(url=QWEN36_URL)
    pdf_svc = OcrToPdfService()

    # --- STEP 1: Layout + OCR ---
    print(f"\n{'='*60}")
    print("STEP 1: Layout (VietOCR) → OCR (Qwen3.6)")
    print(f"{'='*60}")
    t0 = time.time()
    ocr_json = ocr_svc.process(PDF_PATH)
    ocr_elapsed = time.time() - t0

    with open(ocr_json) as f:
        data = json.load(f)
    total_blocks = sum(len(p.get("blocks", [])) for p in data["pages"])
    print(f"✓ OCR done in {ocr_elapsed:.2f}s → {ocr_json}")
    print(f"  {len(data['pages'])} pages, {total_blocks} blocks")

    print("\n  Preview page 1 (first 8 blocks):")
    for b in data["pages"][0].get("blocks", [])[:8]:
        print(f"    ({b['x0']:.0f},{b['y0']:.0f}) | {b['text'][:55]}")

    # --- STEP 2: Render PDF ---
    print(f"\n{'='*60}")
    print("STEP 2: Render PDF")
    print(f"{'='*60}")
    t0 = time.time()
    pdf_path = pdf_svc.convert(ocr_json)
    pdf_elapsed = time.time() - t0
    print(f"✓ PDF done in {pdf_elapsed:.2f}s → {pdf_path}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"✓ Total: {ocr_elapsed + pdf_elapsed:.2f}s")
    print(f"{'='*60}")


async def test_streaming_pipeline():
    """Test async streaming pipeline (OCR + LLM song song)."""
    assert os.path.isfile(PDF_PATH), f"PDF not found: {PDF_PATH}"

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    from core.pipeline import OCRPipeline

    pipeline = OCRPipeline()

    print(f"\n{'='*60}")
    print("STREAMING PIPELINE: OCR ↔ LLM concurrent (sliding window)")
    print(f"{'='*60}")

    t0 = time.time()
    result = await pipeline.run_streaming(PDF_PATH)
    elapsed = time.time() - t0

    print(f"\n✓ Streaming pipeline done in {elapsed:.2f}s")
    print(f"  extract_json: {result['extract_json']}")
    print(f"  result_pdf:   {result['result_pdf']}")
    print(f"  result_tiff:  {result['result_tiff']}")

    # Preview extract result
    with open(result["extract_json"], encoding="utf-8") as f:
        extract = json.load(f)
    print(f"\n  thong_tin_chung keys: {list(extract.get('thong_tin_chung', {}).keys())}")
    print(f"  doi_tuong count: {len(extract.get('danh_sach_doi_tuong', []))}")

    for i, dt in enumerate(extract.get("danh_sach_doi_tuong", [])[:3]):
        bi_cao = dt.get("bi_cao_vn") or {}
        name = bi_cao.get("ho_va_ten", "(không có)")
        print(f"    [{i+1}] {name}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sync", "streaming", "both"], default="streaming")
    args = parser.parse_args()

    if args.mode in ("sync", "both"):
        test_pipeline()
    if args.mode in ("streaming", "both"):
        asyncio.run(test_streaming_pipeline())
