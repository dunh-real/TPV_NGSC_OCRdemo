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
    pdf_path = pdf_svc.convert(ocr_json, pdf_raw_path=PDF_PATH)
    pdf_elapsed = time.time() - t0
    print(f"✓ PDF done in {pdf_elapsed:.2f}s → {pdf_path}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"✓ Total: {ocr_elapsed + pdf_elapsed:.2f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_pipeline()
