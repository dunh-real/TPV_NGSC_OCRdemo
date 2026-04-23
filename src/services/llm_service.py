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
    """Án tích, Tội danh và Hình phạt - Riêng biệt cho từng người"""
    thong_tin_an_tich: Optional[str] = Field(None, description="Thông tin về án tích và tình trạng án tích")
    ten_toi_danh: List[str] = Field(
        default_factory=list,
        description="Danh sách các tội danh bị kết án"
    )
    dieu_khoan_luat_ap_dung: List[str] = Field(
        default_factory=list,
        description="Danh sách các điều, khoản, điểm luật được áp dụng"
    )
    ten_hinh_phat_chinh: Optional[str] = Field(None, description="Tên hình phạt chính")
    thoi_han_gia_tri_hinh_phat_chinh: Optional[str] = Field(None, description="Thời hạn/giá trị hình phạt chính")
    ten_hinh_phat_bo_sung: List[str] = Field(
        default_factory=list,
        description="Danh sách tên các hình phạt bổ sung"
    )
    thoi_han_gia_tri_hinh_phat_bo_sung: Optional[str] = Field(None, description="Thời hạn/giá trị hình phạt bổ sung")
    an_phi: Optional[str] = Field(None, description="Án phí phải nộp")
    mien_phi: Optional[str] = Field(None, description="Miễn phí (án phí/lệ phí)")

class ThongTinBiCao(BaseModel):
    ho_va_ten: Optional[str] = Field(None, description="Họ và tên bị cáo")
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
    so_dinh_danh: Optional[str] = Field(None, description="Số định danh cá nhân người nước ngoài")
    loai_giay_to_xnc: Optional[str] = Field(None, description="Loại giấy tờ Xuất nhập cảnh")
    so_giay_to: Optional[str] = Field(None, description="Số giấy tờ Xuất nhập cảnh")
    ho_ten: Optional[str] = Field(None, description="Họ tên người nước ngoài")
    quoc_tich: Optional[str] = Field(None, description="Quốc tịch")
    ngay_sinh: Optional[str] = Field(None, description="Ngày sinh")
    gioi_tinh: Optional[str] = Field(None, description="Giới tính")

class PhapNhanThuongMai(BaseModel):
    loai_co_quan_to_chuc: Optional[str] = Field(None, description="Loại cơ quan/tổ chức")
    ten_co_quan_to_chuc: Optional[str] = Field(None, description="Tên cơ quan/tổ chức")
    so_gcn_dang_ky_doanh_nghiep: Optional[str] = Field(None, description="Số Giấy chứng nhận đăng ký doanh nghiệp")
    ma_so_thue: Optional[str] = Field(None, description="Mã số thuế")
    ho_ten_nguoi_dai_dien: Optional[str] = Field(None, description="Họ và tên người đại diện pháp luật")
    so_dinh_danh_nguoi_dai_dien: Optional[str] = Field(None, description="Số định danh người đại diện")
    dia_chi_tru_so: Optional[DiaChi] = Field(None, description="Địa chỉ trụ sở chính")

class HoSoDoiTuong(BaseModel):
    """Đại diện cho 1 đối tượng bị xét xử"""
    thong_tin_ket_an: ThongTinKetAn = Field(..., description="Tội danh, hình phạt và án phí của riêng đối tượng này")
    bi_cao_vn: Optional[ThongTinBiCao] = Field(None, description="Điền nếu là công dân VN")
    bi_cao_nuoc_ngoai: Optional[ThongTinBiCaoNuocNgoai] = Field(None, description="Điền nếu là người nước ngoài")
    phap_nhan_pham_toi: Optional[PhapNhanThuongMai] = Field(None, description="Điền nếu là pháp nhân")

class DuLieuBanAn(BaseModel):
    """Class Root chứa toàn bộ thông tin bóc tách từ 1 văn bản Bản án/Quyết định"""
    thong_tin_chung: ThongTinChungBanAn = Field(..., description="Thông tin chung của bản án")
    danh_sach_doi_tuong: List[HoSoDoiTuong] = Field(
        default_factory=list,
        description="Danh sách những người/pháp nhân bị xét xử trong bản án này"
    )

class LLMService:
    def __init__(self, host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL):
        llm = ChatOllama(
            model=model,
            base_url=host,
            temperature=0,
            num_ctx=16384,
            num_predict=4096,
        )
        self.structured_llm = llm.with_structured_output(DuLieuBanAn)
        logging.info(f"[LLMService] Connected to ollama at {host}, model: {model}")

    def extract(self, ocr_text: str) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là Chuyên gia Trích xuất Dữ liệu Pháp lý với khả năng đọc hiểu, phân tích và trích xuất chính xác tuyệt đối thông tin từ các văn bản tố tụng hình sự Việt Nam (Bản án, Quyết định). 
        Nhiệm vụ của bạn là đọc nội dung văn bản OCR từ một bản án/quyết định của tòa án và trích xuất thông tin chính xác theo định dạng được yêu cầu.
        Nếu thông tin nào không có trong văn bản, hãy để trống (null). Không được tự bịa đặt dữ liệu.
        TUYỆT ĐỐI không chèn thêm bất kỳ hội thoại, giải thích hay lời chào nào (phải trả về đúng chuẩn parser JSON)."""),
            ("user", "Nội dung OCR của bản án:\n\n{ocr_text}")
        ])
        chain  = prompt | self.structured_llm
        result = chain.invoke({"ocr_text": ocr_text})

        return result.model_dump() if hasattr(result, "model_dump") else result

    def extract_from_file(self, ocr_json_path: str) -> dict:
        if not os.path.isfile(ocr_json_path):
            raise FileNotFoundError(f"OCR result file not found: {ocr_json_path}")
        
        """
        Chuyển file JSON OCR thành plain text để đưa vào LLM.
        """
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
        