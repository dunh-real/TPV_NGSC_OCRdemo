import os
import sys
import json
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from services.ocr_service import OCRService
from services.llm_service import LLMService

PDF_PATH = os.path.join(os.path.dirname(__file__), "data", "raw", "BA_05.2021.DS-ST.pdf")
EXTRACT_DIR = os.path.join(os.path.dirname(__file__), "data", "result_extract")


def test_pipeline():
    assert os.path.isfile(PDF_PATH), f"Test PDF not found: {PDF_PATH}"

    ocr_service = OCRService()
    llm_service = LLMService()

    # --- OCR ---
    print(f"\n{'='*60}")
    print("STEP 1: OCR")
    print(f"{'='*60}")

    t0 = time.time()
    md_path = ocr_service.process(PDF_PATH)
    ocr_elapsed = time.time() - t0

    assert os.path.isfile(md_path), f"OCR output not found: {md_path}"
    print(f"✓ OCR done in {ocr_elapsed:.2f}s → {md_path}")

    # --- LLM Extract ---
    print(f"\n{'='*60}")
    print("STEP 2: LLM Extract")
    print(f"{'='*60}")

    t0 = time.time()
    result = llm_service.extract_from_file(md_path)
    llm_elapsed = time.time() - t0

    assert isinstance(result, dict), "LLM result is not a dict"

    pdf_name = os.path.splitext(os.path.basename(PDF_PATH))[0]
    json_path = os.path.join(EXTRACT_DIR, f"{pdf_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    assert os.path.isfile(json_path), f"JSON output not created: {json_path}"
    print(f"✓ LLM extract done in {llm_elapsed:.2f}s → {json_path}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"✓ Total: {ocr_elapsed + llm_elapsed:.2f}s")
    print(f"{'='*60}")
    print("\n--- Extracted fields ---")
    for k, v in result.items():
        if v is not None:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    test_pipeline()
