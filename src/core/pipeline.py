import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.qwen36_ocr_service import Qwen36OcrService
from services.llm_service import LLMService
from services.ocr_to_pdf_service import OcrToPdfService
from services.pdf_to_tiff_service import PdfToTiffService

EXTRACT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "result_extract"))


class OCRPipeline:
    def __init__(self):
        # VietOCR (layout/bbox) + Qwen3.6 (OCR text) tích hợp trong Qwen36OcrService
        self.ocr_service  = Qwen36OcrService()
        self.llm_service  = LLMService()
        self.pdf_service  = OcrToPdfService()
        self.tiff_service = PdfToTiffService()
        os.makedirs(EXTRACT_DIR, exist_ok=True)

    def run(self, pdf_path: str) -> dict:
        """
        Sequential pipeline
        Args:
            pdf_path: Đường dẫn tới file PDF đầu vào.

        Returns:
            {
                "extract_json": str,  # JSON thông tin extract
                "result_pdf":   str,  # PDF giữ bố cục gốc
                "result_tiff":  str,  # TIFF giữ bố cục gốc
            }
        """
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

        # VietOCR detect bbox → Qwen3.6 OCR text → JSON
        ocr_json = self.ocr_service.process(pdf_path)

        # LLM extract thông tin
        extract_result = self.llm_service.extract_from_file(ocr_json)
        extract_json   = os.path.join(EXTRACT_DIR, f"{pdf_name}.json")
        with open(extract_json, "w", encoding="utf-8") as f:
            json.dump(extract_result, f, ensure_ascii=False, indent=2)

        # Render PDF với FontStyleService (rule-based)
        result_pdf  = self.pdf_service.convert(ocr_json)
        
        # Convert PDF → TIFF multi-page
        result_tiff = self.tiff_service.convert(result_pdf)

        return {
            "extract_json": extract_json,
            "result_pdf":   result_pdf,
            "result_tiff":  result_tiff,
        }

    async def run_streaming(self, pdf_path: str) -> dict:
        """
        Streaming pipeline: OCR và LLM chạy song song.
        OCR xong page nào, LLM extract ngay page đó (sliding window overlap).

        Args:
            pdf_path: Đường dẫn tới file PDF đầu vào.

        Returns:
            {
                "extract_json": str,
                "result_pdf":   str,
                "result_tiff":  str,
            }
        """
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        queue    = asyncio.Queue()

        # Producer: OCR page-by-page, đẩy vào queue
        producer = asyncio.create_task(
            self.ocr_service.process_streaming(pdf_path, queue)
        )

        # Consumer: LLM extract với sliding window
        consumer = asyncio.create_task(
            self.llm_service.extract_streaming(queue, overlap=1)
        )

        # Đợi producer xong rồi gửi sentinel cho consumer
        ocr_json = await producer
        await queue.put(None)

        extract_result = await consumer

        # Save extract result
        extract_json = os.path.join(EXTRACT_DIR, f"{pdf_name}.json")
        with open(extract_json, "w", encoding="utf-8") as f:
            json.dump(extract_result, f, ensure_ascii=False, indent=2)

        # Render PDF + TIFF
        result_pdf  = self.pdf_service.convert(ocr_json)
        result_tiff = self.tiff_service.convert(result_pdf)

        return {
            "extract_json": extract_json,
            "result_pdf":   result_pdf,
            "result_tiff":  result_tiff,
        }  

async def main():
    import time
    pipeline_client = OCRPipeline()

    ftime = time.time()
    input_file = "../../data/raw/BA_05.2021.DS-ST.pdf"

    # end-to-end pipeline
    result = await pipeline_client.run_streaming(input_file)

    etime = time.time() - ftime
    print(f"Total time: {etime}s")

    print(f"Extract JSON: {result['extract_json']}")
    print(f"Result PDF: {result['result_pdf']}")
    print(f"Result TIFF: {result['result_tiff']}")

    
if __name__ == "__main__":
    asyncio.run(main())