import asyncio
import logging
import os
import json
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional, List

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")

class DiaChi(BaseModel):
    """Cấu trúc dùng chung cho các trường địa chỉ (Cấp tỉnh, huyện, xã, chi tiết)"""
    tinh_thanh_pho: Optional[str] = Field(None, description="Tỉnh hoặc Thành phố trực thuộc trung ương")
    quan_huyen: Optional[str] = Field(None, description="Quận, Huyện, Thị xã hoặc Thành phố thuộc tỉnh")
    xa_phuong: Optional[str] = Field(None, description="Xã, Phường hoặc Thị trấn")
    dia_chi_chi_tiet: Optional[str] = Field(None, description="Số nhà, đường phố, thôn xóm (Địa chỉ cụ thể)")

class GiayToDinhDanh(BaseModel):
    """Cấu trúc dùng chung cho giấy tờ tùy thân"""
    loai_giay_to: Optional[str] = Field(None, description="Loại giấy tờ định danh (VD: CCCD, CMND, Hộ chiếu)")
    so_giay_to: Optional[str] = Field(None, description="Số của giấy tờ định danh")
    ngay_cap: Optional[str] = Field(None, description="Ngày cấp giấy tờ (Định dạng DD/MM/YYYY nếu có)")
    ngay_het_han: Optional[str] = Field(None, description="Ngày hết hạn của giấy tờ (Định dạng DD/MM/YYYY nếu có)")


class ThongTinChungBanAn(BaseModel):
    """Thông tin chung, áp dụng cho toàn bộ văn bản (Chỉ bóc tách 1 lần)"""
    so_ban_an_quyet_dinh: Optional[str] = Field(None, description="Số Bản án/Quyết định")
    ngay_ban_hanh: Optional[str] = Field(None, description="Ngày ban hành Bản án/Quyết định")
    ngay_hieu_luc: Optional[str] = Field(None, description="Ngày hiệu lực Bản án/Quyết định")
    ten_don_vi_ban_hanh: Optional[str] = Field(None, description="Tên đơn vị ban hành (Tòa án)")
    so_ban_an_lien_quan: Optional[str] = Field(None, description="Số Bản án/Quyết định liên quan")
    ngay_ban_hanh_lien_quan: Optional[str] = Field(None, description="Ngày ban hành Bản án/Quyết định liên quan")
    ngay_hieu_luc_lien_quan: Optional[str] = Field(None, description="Ngày hiệu lực Bản án/Quyết định liên quan")
    don_vi_ban_hanh_lien_quan: Optional[str] = Field(None, description="Đơn vị ban hành Bản án/Quyết định liên quan")
    trang_thai_ban_an: Optional[str] = Field(None, description="Trạng thái Bản án/Quyết định")
    ghi_chu: Optional[str] = Field(None, description="Ghi chú chung")

class ThongTinKetAn(BaseModel):
    """Án tích, Tội danh và Hình phạt — riêng biệt cho từng đối tượng (bị cáo/bị đơn)."""
    thong_tin_an_tich: Optional[str] = Field(None, description="Thông tin về án tích và tình trạng án tích")
    ten_toi_danh: List[str] = Field(
        default_factory=list,
        description=(
            "Danh sách TÊN TỘI DANH (hành vi phạm tội/vi phạm pháp luật). "
            "VD: 'Trộm cắp tài sản', 'Lừa đảo chiếm đoạt tài sản', 'Tranh chấp đất đai'. "
            "KHÔNG được điền tên người vào đây."
        ),
    )
    dieu_khoan_luat_ap_dung: List[str] = Field(
        default_factory=list,
        description="Danh sách các điều, khoản, điểm luật được áp dụng"
    )
    ten_hinh_phat_chinh: Optional[str] = Field(None, description="Tên hình phạt chính (VD: 'Tù có thời hạn', 'Phạt tiền')")
    thoi_han_gia_tri_hinh_phat_chinh: Optional[str] = Field(None, description="Thời hạn/giá trị hình phạt chính")
    ten_hinh_phat_bo_sung: List[str] = Field(
        default_factory=list,
        description="Danh sách tên các hình phạt bổ sung"
    )
    thoi_han_gia_tri_hinh_phat_bo_sung: Optional[str] = Field(None, description="Thời hạn/giá trị hình phạt bổ sung")
    an_phi: Optional[str] = Field(None, description="Án phí phải nộp")
    mien_phi: Optional[str] = Field(None, description="Miễn phí (án phí/lệ phí)")

class ThongTinBiCao(BaseModel):
    """Thông tin cá nhân của đương sự/bị cáo (Nguyên đơn, Bị đơn, Bị cáo, ...)."""
    ho_va_ten: Optional[str] = Field(None, description="Họ và tên đầy đủ của đương sự/bị cáo (là TÊN NGƯỜI, không phải tên tổ chức)")
    ten_goi_khac: Optional[str] = Field(None, description="Tên gọi khác/Bí danh")
    giay_to_dinh_danh: Optional[GiayToDinhDanh] = Field(None, description="Thông tin giấy tờ định danh")
    gioi_tinh: Optional[str] = Field(None, description="Giới tính")
    ngay_sinh: Optional[str] = Field(None, description="Ngày tháng năm sinh")
    noi_sinh: Optional[DiaChi] = Field(None, description="Thông tin nơi sinh (Tỉnh, Huyện, Xã, Địa chỉ)")
    quoc_tich: Optional[str] = Field(None, description="Quốc tịch")
    dan_toc: Optional[str] = Field(None, description="Dân tộc")
    ton_giao: Optional[str] = Field(None, description="Tôn giáo")
    noi_cu_tru: Optional[DiaChi] = Field(None, description="Nơi cư trú hiện tại")
    noi_thuong_tru: Optional[DiaChi] = Field(None, description="Nơi đăng ký thường trú")
    que_quan: Optional[DiaChi] = Field(None, description="Quê quán")
    ho_ten_cha: Optional[str] = Field(None, description="Họ và tên cha")
    ho_ten_me: Optional[str] = Field(None, description="Họ và tên mẹ")
    ho_ten_vo_chong: Optional[str] = Field(None, description="Họ và tên vợ/chồng")

class ThongTinBiCaoNuocNgoai(BaseModel):
    """Thông tin đương sự/bị cáo là người nước ngoài."""
    so_dinh_danh: Optional[str] = Field(None, description="Số định danh cá nhân người nước ngoài")
    loai_giay_to_xnc: Optional[str] = Field(None, description="Loại giấy tờ Xuất nhập cảnh")
    so_giay_to: Optional[str] = Field(None, description="Số giấy tờ Xuất nhập cảnh")
    ho_ten: Optional[str] = Field(None, description="Họ tên người nước ngoài")
    quoc_tich: Optional[str] = Field(None, description="Quốc tịch")
    ngay_sinh: Optional[str] = Field(None, description="Ngày sinh")
    gioi_tinh: Optional[str] = Field(None, description="Giới tính")

class PhapNhanThuongMai(BaseModel):
    """Chỉ điền khi đối tượng là PHÁP NHÂN/DOANH NGHIỆP bị xét xử (không phải cá nhân)."""
    loai_co_quan_to_chuc: Optional[str] = Field(None, description="Loại cơ quan/tổ chức")
    ten_co_quan_to_chuc: Optional[str] = Field(None, description=(
        "Tên pháp nhân/doanh nghiệp bị xét xử. "
        "KHÔNG điền vai trò tố tụng (Nguyên đơn, Bị đơn, Người có quyền lợi nghĩa vụ liên quan) vào đây."
    ))
    so_gcn_dang_ky_doanh_nghiep: Optional[str] = Field(None, description="Số Giấy chứng nhận đăng ký doanh nghiệp")
    ma_so_thue: Optional[str] = Field(None, description="Mã số thuế")
    ho_ten_nguoi_dai_dien: Optional[str] = Field(None, description="Họ và tên người đại diện pháp luật")
    so_dinh_danh_nguoi_dai_dien: Optional[str] = Field(None, description="Số định danh người đại diện")
    dia_chi_tru_so: Optional[DiaChi] = Field(None, description="Địa chỉ trụ sở chính")

class HoSoDoiTuong(BaseModel):
    """Mỗi phần tử đại diện cho 1 CÁ NHÂN (đương sự/bị cáo) trong vụ án.
    Mỗi cá nhân xuất hiện trong bản án nên có đúng 1 HoSoDoiTuong.
    PHẢI điền bi_cao_vn (nếu là người VN) hoặc bi_cao_nuoc_ngoai (nếu là người nước ngoài).
    Chỉ điền phap_nhan_pham_toi nếu đối tượng là doanh nghiệp/tổ chức."""
    thong_tin_ket_an: ThongTinKetAn = Field(..., description="Tội danh, hình phạt và án phí của riêng đối tượng này")
    bi_cao_vn: Optional[ThongTinBiCao] = Field(None, description="PHẢI điền nếu đối tượng là cá nhân công dân VN")
    bi_cao_nuoc_ngoai: Optional[ThongTinBiCaoNuocNgoai] = Field(None, description="Điền nếu là người nước ngoài")
    phap_nhan_pham_toi: Optional[PhapNhanThuongMai] = Field(None, description="Chỉ điền nếu đối tượng là pháp nhân/doanh nghiệp, KHÔNG điền cho cá nhân")

class DuLieuBanAn(BaseModel):
    """Toàn bộ thông tin bóc tách từ 1 văn bản Bản án/Quyết định."""
    thong_tin_chung: ThongTinChungBanAn = Field(..., description="Thông tin chung của bản án")
    danh_sach_doi_tuong: List[HoSoDoiTuong] = Field(
        default_factory=list,
        description=(
            "Danh sách các đương sự/bị cáo trong bản án. "
            "Mỗi CÁ NHÂN = 1 phần tử. Phải điền ho_va_ten vào bi_cao_vn hoặc bi_cao_nuoc_ngoai."
        ),
    )

class LLMService:
    def __init__(self, host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL, stream_num_ctx: int = 8192):
        # Sync pipeline
        llm = ChatOllama(
            model=model,
            base_url=host,
            temperature=0,
            num_ctx=16384,
            num_predict=4096,
        )
        self.structured_llm = llm.with_structured_output(DuLieuBanAn)

        # Async pipeline - Streaming LLM
        stream_llm = ChatOllama(
            model=model,
            base_url=host,
            temperature=0,
            num_ctx=stream_num_ctx,
            num_predict=4096,
        )
        self.stream_structured_llm = stream_llm.with_structured_output(DuLieuBanAn)

        self._system_prompt = (
            "Bạn là Chuyên gia Trích xuất Dữ liệu Pháp lý.\n"
            "Nhiệm vụ: Đọc nội dung OCR từ bản án/quyết định của tòa án Việt Nam "
            "và trích xuất thông tin chính xác theo schema.\n\n"
            "QUY TẮC BẮT BUỘC:\n"
            "1. Mỗi CÁ NHÂN (đương sự/bị cáo/bị đơn/nguyên đơn) = 1 phần tử trong danh_sach_doi_tuong.\n"
            "2. PHẢI điền ho_va_ten vào bi_cao_vn (nếu là người VN) hoặc bi_cao_nuoc_ngoai (nếu là người nước ngoài). "
            "KHÔNG để bi_cao_vn = null khi đã biết tên người.\n"
            "3. ten_toi_danh chỉ chứa TÊN HÀNH VI PHẠM TỘI/VI PHẠM (VD: 'Tranh chấp đất đai', 'Trộm cắp tài sản'). "
            "TUYỆT ĐỐI KHÔNG điền tên người vào ten_toi_danh.\n"
            "4. phap_nhan_pham_toi chỉ dùng cho DOANH NGHIỆP/TỔ CHỨC bị xét xử. "
            "KHÔNG điền vai trò tố tụng (Nguyên đơn, Bị đơn, Người có quyền lợi nghĩa vụ liên quan) vào đây.\n"
            "5. Nếu thông tin không có trong văn bản → null. Không bịa đặt. KHÔNG được ghi 'Không rõ', 'không xác định' hay bất kỳ chuỗi nào thay cho null.\n"
            "6. TUYỆT ĐỐI không chèn hội thoại, giải thích, lời chào (trả về đúng chuẩn JSON)."
        )
        logging.info(f"[LLMService] Connected to ollama at {host}, model: {model}")

    def _make_chain(self, structured_llm):
        """Tạo LangChain chain với system prompt chung."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._system_prompt),
            ("user", "Nội dung OCR của bản án:\n\n{ocr_text}"),
        ])
        return prompt | structured_llm

    # Sync pipeline - Full-document LLM
    def extract(self, ocr_text: str) -> dict:
        """Extract từ toàn bộ OCR text (full-document)."""
        chain  = self._make_chain(self.structured_llm)
        result = chain.invoke({"ocr_text": ocr_text})

        return result.model_dump() if hasattr(result, "model_dump") else result

    def extract_from_file(self, ocr_json_path: str) -> dict:
        """Chuyển file JSON OCR thành plain text rồi đưa vào LLM."""
        if not os.path.isfile(ocr_json_path):
            raise FileNotFoundError(f"OCR result file not found: {ocr_json_path}")

        with open(ocr_json_path, encoding="utf-8") as f:
            data = json.load(f)

        parts = []
        for page in data.get("pages", []):
            lines = [b["text"] for b in page.get("blocks", []) if b.get("text", "").strip()]
            if lines:
                parts.append("\n".join(lines))

        ocr_text = "\n".join(parts)

        if not ocr_text.strip():
            raise ValueError(f"No text extracted from: {ocr_json_path}")

        return self.extract(ocr_text)

    # Async pipeline - Sliding window LLM
    def extract_window(self, ocr_text: str) -> dict:
        """Extract từ một sliding window (default window_size = 2)."""
        chain  = self._make_chain(self.stream_structured_llm)
        result = chain.invoke({"ocr_text": ocr_text})

        return result.model_dump() if hasattr(result, "model_dump") else result

    async def extract_streaming(self, queue: asyncio.Queue, overlap: int = 1) -> dict:
        """
        Consumer: nhận page data từ queue, xử lý sliding window, merge kết quả incremental.
        """
        pages_buffer: list[dict] = []
        merged: dict | None = None

        while True:
            page_data = await queue.get()
            if page_data is None:         
                break

            pages_buffer.append(page_data)

            # Sliding window: overlap trang trước + trang hiện tại
            window_start = max(0, len(pages_buffer) - 1 - overlap)
            window_pages = pages_buffer[window_start:]

            window_text = self._pages_to_text(window_pages)
            if not window_text.strip():
                continue

            try:
                partial = await asyncio.to_thread(self.extract_window, window_text)
                merged  = self._merge_results(merged, partial)
                logging.info(
                    f"[LLMService] Processed window pages "
                    f"{[p['page'] for p in window_pages]}"
                )
            except Exception as e:
                logging.error(
                    f"[LLMService] Window pages "
                    f"{[p['page'] for p in window_pages]} error: {e}"
                )

        result = merged if merged is not None else self._empty_result()
        return self._cleanup_result(result)

    # Helpers                                                           
    @staticmethod
    def _pages_to_text(pages: list[dict]) -> str:
        """Chuyển list page dicts thành plain text."""
        parts = []
        for page in pages:
            lines = [
                b["text"]
                for b in page.get("blocks", [])
                if b.get("text", "").strip()
            ]
            if lines:
                parts.append("\n".join(lines))
        return "\n".join(parts)

    @staticmethod
    def _empty_result() -> dict:
        """Trả về template rỗng đúng schema."""
        return DuLieuBanAn(
            thong_tin_chung=ThongTinChungBanAn(),
            danh_sach_doi_tuong=[],
        ).model_dump()

    # Merge / Deduplicate

    # Các dạng dấu nháy Unicode mà OCR có thể trả về khác nhau
    _QUOTE_TABLE = str.maketrans({
        "\u2018": "'",   # '
        "\u2019": "'",   # '
        "\u201A": "'",   # ‚
        "\u02BC": "'",   # ʼ
        "\u0060": "'",   # `
        "\u00B4": "'",   # ´
    })

    # Các chuỗi LLM hay bịa khi không tìm thấy thông tin
    _PLACEHOLDER_STRINGS = {"null", "không rõ", "không xác định", "n/a", "none"}

    @classmethod
    def _normalize_name(cls, name: str) -> str:
        """Chuẩn hóa tên: thống nhất dấu nháy Unicode về ASCII, strip, collapse spaces."""
        name = name.translate(cls._QUOTE_TABLE)
        return " ".join(name.split())  # collapse multiple spaces

    @classmethod
    def _is_null_value(cls, val) -> bool:
        """Kiểm tra giá trị null (None, string 'null', 'Không rõ', v.v.)."""
        if val is None:
            return True
        if isinstance(val, str) and val.strip().lower() in cls._PLACEHOLDER_STRINGS:
            return True
        return False

    @classmethod
    def _is_empty_doi_tuong(cls, dt: dict) -> bool:
        """True nếu đối tượng chỉ chứa toàn giá trị null/rỗng."""
        def _all_empty(obj):
            if obj is None:
                return True
            if isinstance(obj, str):
                return not obj.strip() or obj.strip().lower() in cls._PLACEHOLDER_STRINGS
            if isinstance(obj, list):
                return len(obj) == 0
            if isinstance(obj, dict):
                return all(_all_empty(v) for v in obj.values())
            return False
        return _all_empty(dt)

    @classmethod
    def _get_person_name(cls, dt: dict) -> str | None:
        """Lấy tên CÁ NHÂN từ bi_cao_vn hoặc bi_cao_nuoc_ngoai."""
        bi_cao = dt.get("bi_cao_vn") or {}
        name = bi_cao.get("ho_va_ten")

        if not name:
            bi_cao_nn = dt.get("bi_cao_nuoc_ngoai") or {}
            name = bi_cao_nn.get("ho_ten")

        if cls._is_null_value(name):
            return None

        return cls._normalize_name(name)

    @classmethod
    def _get_doi_tuong_key(cls, dt: dict) -> tuple:
        """
        Composite key (person_name, ngay_sinh, so_giay_to).
        Chỉ dựa vào bi_cao_vn/bi_cao_nuoc_ngoai.
        """
        name = cls._get_person_name(dt)

        bi_cao = dt.get("bi_cao_vn") or {}
        ngay_sinh = (
            bi_cao.get("ngay_sinh")
            or (dt.get("bi_cao_nuoc_ngoai") or {}).get("ngay_sinh")
        )
        so_giay_to = (
            (bi_cao.get("giay_to_dinh_danh") or {}).get("so_giay_to")
            or (dt.get("bi_cao_nuoc_ngoai") or {}).get("so_giay_to")
        )

        return (name, ngay_sinh, so_giay_to)

    @classmethod
    def _keys_compatible(cls, key_a: tuple, key_b: tuple) -> bool:
        """Hai key tương thích nếu các trường phụ không mâu thuẫn."""
        _, ns_a, gt_a = key_a
        _, ns_b, gt_b = key_b
        if ns_a and ns_b and ns_a != ns_b:
            return False
        if gt_a and gt_b and gt_a != gt_b:
            return False
        return True

    @classmethod
    def _find_matching_doi_tuong(cls, existing_list: list[dict], new_dt: dict) -> int | None:
        """
        Tìm index đối tượng match.
        - Nếu new có person name → tìm entry trùng name, hoặc entry rỗng.
        - Nếu new KHÔNG có person name nhưng có thong_tin_ket_an →
          merge vào entry cuối cùng (info bổ sung từ page tiếp).
        """
        new_key = cls._get_doi_tuong_key(new_dt)
        new_name = new_key[0]

        if new_name is not None:
            # Ưu tiên: tìm entry có cùng person name
            for i, dt in enumerate(existing_list):
                ex_key = cls._get_doi_tuong_key(dt)
                if ex_key[0] == new_name and cls._keys_compatible(ex_key, new_key):
                    return i
            # Fallback: tìm entry rỗng để ghi đè
            for i, dt in enumerate(existing_list):
                if cls._is_empty_doi_tuong(dt):
                    return i
            return None  # entry mới thực sự

        # new không có person name → merge vào entry cuối
        if existing_list:
            return len(existing_list) - 1

        return None

    @classmethod
    def _deep_merge_dict(cls, base: dict, update: dict) -> None:
        """
        Merge update vào base:
        - Giữ nguyên tất cả key của base (không thêm, không xóa)
        - base[key] = null → ghi đè bằng giá trị mới
        - base[key] khác null, scalar → TUYỆT ĐỐI giữ nguyên
        - Dict lồng nhau: đệ quy
        - List: extend + deduplicate
        """
        for key in base:
            if key not in update:
                continue
            new_val = update[key]
            if cls._is_null_value(new_val):
                continue

            old_val = base[key]

            if cls._is_null_value(old_val):
                base[key] = new_val
            elif isinstance(old_val, dict) and isinstance(new_val, dict):
                cls._deep_merge_dict(old_val, new_val)
            elif isinstance(old_val, list) and isinstance(new_val, list):
                for item in new_val:
                    if item not in old_val:
                        old_val.append(item)
            # old_val khác null, không phải dict/list → giữ nguyên

    @classmethod
    def _merge_results(cls, base: dict | None, partial: dict) -> dict:
        """
        Merge partial vào base:
        1. thong_tin_chung: deep merge
        2. danh_sach_doi_tuong:
           - Có person name trùng → merge vào entry cũ
           - Có person name mới → append
           - Không có person name → merge ket_an vào entry cuối
        """
        if base is None:
            return partial

        # 1. Merge thong_tin_chung
        cls._deep_merge_dict(
            base.get("thong_tin_chung", {}),
            partial.get("thong_tin_chung", {}),
        )

        # 2. Merge danh_sach_doi_tuong
        for new_dt in partial.get("danh_sach_doi_tuong", []):
            if cls._is_empty_doi_tuong(new_dt):
                continue

            matched_idx = cls._find_matching_doi_tuong(
                base["danh_sach_doi_tuong"], new_dt
            )

            if matched_idx is not None:
                cls._deep_merge_dict(
                    base["danh_sach_doi_tuong"][matched_idx], new_dt
                )
            else:
                # Chỉ append nếu new có person name (bi_cao_vn/bi_cao_nuoc_ngoai)
                new_name = cls._get_person_name(new_dt)
                if new_name is not None:
                    base["danh_sach_doi_tuong"].append(new_dt)

        return base

    @classmethod
    def _cleanup_result(cls, result: dict) -> dict:
        """Loại bỏ đối tượng rỗng và dọn string 'null' khỏi kết quả cuối."""
        # Xóa đối tượng toàn null
        result["danh_sach_doi_tuong"] = [
            dt for dt in result.get("danh_sach_doi_tuong", [])
            if not cls._is_empty_doi_tuong(dt)
        ]

        # Dọn string "null" → None trong toàn bộ result
        cls._sanitize_null_strings(result)

        return result

    @classmethod
    def _sanitize_null_strings(cls, obj):
        """Đệ quy thay thế string 'null', 'Không rõ', v.v. bằng None."""
        if isinstance(obj, dict):
            for key in obj:
                if cls._is_null_value(obj[key]):
                    obj[key] = None
                else:
                    cls._sanitize_null_strings(obj[key])
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, str) and item.strip().lower() in cls._PLACEHOLDER_STRINGS:
                    obj[i] = None
                else:
                    cls._sanitize_null_strings(item)