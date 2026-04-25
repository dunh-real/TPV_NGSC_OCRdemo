# TPV NGSC OCR Demo

## Kiến trúc Pipeline

Hỗ trợ **2 chế độ** chạy pipeline:

### Chế độ 1 — Sequential (`run`)

Chạy tuần tự: OCR toàn bộ → LLM extract toàn bộ → Render PDF/TIFF.

```
Input PDF
   ↓
[Layout Detection]  Deepdoc VietOCR detect bbox
   ↓
[OCR Text]          Qwen3.6-35B (vLLM) — tuần tự từng batch
   ↓
[LLM Extract]       Ollama qwen2.5:14b — full document
   ↓
[Render PDF]        ReportLab — PDF A4 giữ bố cục
   ↓
[PDF → TIFF]        300 DPI, nén LZW
```

### Chế độ 2 — Async Streaming (`run_streaming`) ⚡

OCR và LLM chạy **song song** theo mô hình Producer–Consumer với `asyncio.Queue`.
OCR xử lý page, gửi kết quả vào Queue, LLM lấy kết quả từ Queue để extract (sliding window, overlap = 1).
Ví du: Window 1 xử lý page 1, Window 2 xử lý page 1 và 2, ... Window n xử lý page n và n+1

```
Input PDF
   ↓
┌──────────────────────────────────────────────────────┐
│             asyncio.Queue (page-by-page)             │
│                                                      │
│  [Producer: OCR]              [Consumer: LLM]        │
│  Layout detect → Qwen3.6     Sliding window extract  │
│  Concurrent batches           Smart merge results    │
│  (max batch/1 request=4)      (overlap=1, ctx=8192)  │  
│         ↓                           ↓                │
│     page data ──→ Queue ──→ extract & merge          │
└──────────────────────────────────────────────────────┘
   ↓
[Save extract JSON]
   ↓
[Render PDF]  →  [PDF → TIFF]
```

**Ưu điểm so với Sequential:**
- LLM bắt đầu extract ngay khi page đầu tiên OCR xong, không chờ toàn bộ.
- OCR concurrent batches (Semaphore=4) tận dụng vLLM continuous batching.
- Sliding window (2 trang, overlap 1) giảm context length, tăng chất lượng extract.

## Cấu trúc thư mục

```
TPV_NGSC_OCRdemo/
├── src/
│   ├── core/
│   │   └── pipeline.py              # Pipeline chính (run + run_streaming)
│   ├── services/
│   │   ├── layout_ocr_service.py    # Deepdoc VietOCR — detect bbox
│   │   ├── qwen36_ocr_service.py    # Qwen3.6 OCR — text recognition
│   │   ├── llm_service.py           # LLM extract + Pydantic schema + Smart merge
│   │   ├── font_style_service.py    # Rule-based font/style (TT 01/2011/TT-BNV)
│   │   ├── ocr_to_pdf_service.py    # ReportLab render PDF A4
│   │   └── pdf_to_tiff_service.py   # PDF → TIFF multi-page
│   ├── api/                         # FastAPI endpoints
│   ├── models/                      # Shared models
│   └── assets/                      # Font files, resources
├── data/
│   ├── raw/                         # PDF đầu vào
│   ├── result_ocr/                  # JSON kết quả OCR (text + bbox)
│   ├── result_extract/              # JSON kết quả extract (structured data)
│   ├── result_pdf/                  # PDF giữ bố cục gốc
│   └── result_tiff/                 # TIFF multi-page
├── qwen3.6_ocr_service.ipynb        # Notebook khởi động vLLM server
├── test_pipeline.py                 # Test VietOCR + LLM extract
├── test_qwen36_pipeline.py          # Test OCR + render PDF
├── requirements.txt
└── README.md
```

## Entry points

| File | Mục đích |
|------|----------|
| `src/core/pipeline.py` | Pipeline chính — `run()` (sync) và `run_streaming()` (async) |
| `qwen3.6_ocr_service.ipynb` | Khởi động vLLM server (Colab) |

## Cấu hình

| Biến môi trường | Mặc định | Mô tả |
|---|---|---|
| `QWEN36_OCR_URL` | `https://vks-ocr-hvks.loca.lt` | Endpoint vLLM server cho Qwen3.6 OCR |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server cho LLM extract |
| `OLLAMA_MODEL` | `qwen2.5:14b` | Model Ollama dùng để extract |

### Tham số pipeline

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| OCR `batch_size` | 20 | Số block/batch gửi Qwen3.6 |
| OCR `max_concurrent` | 4 | Semaphore cho concurrent batches |
| LLM `num_ctx` (sync) | 16384 | Context length cho full-document |
| LLM `num_ctx` (streaming) | 8192 | Context length cho sliding window |
| Sliding window `overlap` | 1 | Số trang chồng lấp giữa các window |
| Render DPI | 216 (72×3) | Khớp layout detection và render PDF |
| TIFF DPI | 300 | Output TIFF resolution |

## Chi tiết kỹ thuật

### OCR Service (`qwen36_ocr_service.py`)

1. **Layout Detection**: Deepdoc VietOCR phát hiện các text block (bbox) trên mỗi trang.
2. **Grid Building**: Crop từng bbox → gộp thành ảnh grid → gửi Qwen3.6 (vLLM).
3. **Concurrent Batches**: Các batch của cùng 1 trang được gửi đồng thời qua `asyncio.gather` với `Semaphore(4)`.
4. **Streaming**: `process_streaming()` đẩy kết quả từng trang vào `asyncio.Queue` ngay khi OCR xong.

### LLM Service (`llm_service.py`)

1. **Pydantic Schema**: `DuLieuBanAn` → `ThongTinChungBanAn` + `List[HoSoDoiTuong]` (bao gồm `ThongTinKetAn`, `ThongTinBiCao`, `PhapNhanThuongMai`).
2. **Sliding Window Extract**: Mỗi window gồm 2 trang liên tiếp (overlap=1), extract qua Ollama structured output.
3. **Smart Merge** (`_merge_results`):
   - `thong_tin_chung`: Deep merge — ghi đè null, giữ nguyên giá trị đã có.
   - `danh_sach_doi_tuong`: Deduplicate bằng composite key `(person_name, ngay_sinh, so_giay_to)`.
   - Chỉ tạo entry mới khi xuất hiện person name mới (`bi_cao_vn` / `bi_cao_nuoc_ngoai`).
   - Entry không có person name → merge vào entry cuối (bổ sung thông tin cross-page).
4. **Normalize & Cleanup**:
   - Chuẩn hóa dấu nháy Unicode (`'` `'` `ʼ` → `'`) để tránh trùng lặp do OCR.
   - Sanitize placeholder strings (`"null"`, `"Không rõ"`, `"N/A"`) → `None`.
