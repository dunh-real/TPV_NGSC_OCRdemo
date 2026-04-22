# TPV NGSC OCR Demo

## Pipeline hiện tại

```
Input PDF
   ↓
[Layout Detection] src/services/layout_ocr_service.py
   Deepdoc VietOCR detect từng dòng văn bản → bbox (x0, y0, x1, y1)
   ↓
[OCR Text] src/services/qwen36_ocr_service.py
   Crop từng bbox → gộp thành grid ảnh → gửi Qwen3.6-35B-A3B (vLLM)
   Qwen3.6 OCR text chính xác tiếng Việt cho từng dòng
   Output: data/result_ocr/<name>_qwen36.json
   ↓
[Font & Style] src/services/font_style_service.py
   Rule-based theo Thông tư 01/2011/TT-BNV
   Gán bold/italic/underline/align/size cho từng block
   ↓
[Render PDF] src/services/ocr_to_pdf_service.py
   ReportLab render PDF A4, font Times New Roman
   Output: data/result_pdf/<name>_qwen36.pdf
   ↓
[PDF to TIFF] src/services/pdf_to_tiff_service.py
   Convert PDF → TIFF multi-page, 300 DPI, nén LZW
   Output: data/result_tiff/<name>_qwen36.tiff
   ↓
[LLM Extract] src/services/llm_service.py
   Ollama (qwen2.5:7b) extract các trường thông tin cố định
   Output: data/result_extract/<name>.json
```

## Entry points

| File | Mục đích |
|------|----------|
| `src/core/pipeline.py` | Pipeline chính, gọi toàn bộ flow |
| `test_qwen36_pipeline.py` | Test OCR + render PDF |
| `test_pipeline.py` | Test VietOCR + LLM extract + render PDF |
| `qwen3.6_ocr_service.ipynb` | Khởi động vLLM server trên Colab |

## Cấu hình

- **Qwen3.6 server**: set env `QWEN36_OCR_URL` (default: `https://vks-ocr-hvks.loca.lt`)
- **Ollama**: chạy local tại `http://localhost:11434`, model `qwen2.5:7b`
- **DPI ảnh**: 216 DPI (72 × 3) — khớp giữa layout detection và render PDF
